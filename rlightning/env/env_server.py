"""Remote environment server for ZMQ-based environment control."""

import pickle
import queue
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Tuple

import ray
import zmq

from rlightning.env.base_env import BaseEnv
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.logger import get_logger
from rlightning.utils.ray import resolve_object
from rlightning.utils.registry import ENVS
from rlightning.utils.utils import InternalFlag
from rlightning.utils.zmq import communication

logger = get_logger(__name__)


@ENVS.register("env_server")
class RemoteEnvServer(BaseEnv):
    """ZMQ-backed environment server that brokers remote clients."""

    def __init__(
        self,
        config: Any,
        worker_index: Optional[int] = 0,
        preprocess_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the remote environment server."""
        super().__init__(config, worker_index, preprocess_fn)

        self._address = ray.util.get_node_ip_address()
        self._port = config.zmq_port if hasattr(config, "zmq_port") else communication.get_free_port()

        self.context, self.socket = None, None

        self.max_envs = 32

        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.event = threading.Event()
        self.env_ret_queue = queue.Queue(maxsize=self.max_envs)
        self.send_msg_queue = queue.Queue(maxsize=self.max_envs)

        if self._preprocess_fn is not None:
            warnings.warn(
                "ZMQEnvServer does not support preprocess_fn. The preprocess_fn will be ignored."
                "Please implement preprocessing on the client side."
            )

    def init(self) -> None:
        """Initialize ZMQ sockets and start background threads."""
        self.context, self.socket = communication.create_socket(self._port)
        logger.info(f"RemoteEnv Server initialized at {self._address}:{self._port}\nWaiting for clients to connect...")
        # recv thread
        self.thread_pool.submit(
            self._recv_thread,
            self.env_id,
            self.socket,
            self.env_ret_queue,
            self.event,
        )
        # send thread
        self.thread_pool.submit(
            self._send_thread,
            self.socket,
            self.send_msg_queue,
            self.event,
        )
        logger.info("RemoteEnv Server is ready to connect env clients.")

    def get_address_port(self) -> Tuple[str, int]:
        """Return the server address and port."""
        return self._address, self._port

    @staticmethod
    def _recv_thread(
        env_id: str,
        socket: zmq.Socket,
        env_ret_queue: queue.Queue,
        event: threading.Event,
    ) -> None:
        """Threaded request receiver
        Expect to receive an EnvRet object from each client

        Args:
            env_id (str): Environment ID used as the prefix for client env IDs.
            socket (zmq.Socket): Socket instance
            env_ret_queue (deque): EnvRet deque
            event (threading.Event): Threading event
        """

        while not event.is_set():
            try:
                zmq_identity, _, message = socket.recv_multipart(flags=zmq.NOBLOCK)
                message = pickle.loads(message)

                client_state = message["client_state"]
                env_ret = message["env_ret"]
                if client_state == "closed":
                    logger.info(f"Client {zmq_identity.decode()} has closed the connection.")
                    continue

                env_ret.env_id = env_id + "/" + zmq_identity.decode()
                env_ret_queue.put(env_ret)

            except zmq.Again:
                event.wait(0.02)
            except Exception as e:
                logger.error(f"Error in _recv_thread: {e}")
                import traceback

                traceback.print_exc()

    @staticmethod
    def _send_thread(
        socket: zmq.Socket,
        send_msg_queue: queue.Queue,
        event: threading.Event,
    ) -> None:
        """Threaded policy_response sender
            Send PolicyResponse to clients

        Args:
            socket (zmq.Socket): Socket instance
            send_msg_queue (deque): PolicyResponse queue
            event (threading.Event): Threading event instance
        """

        while not event.is_set():
            try:
                msg_send = send_msg_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            policy_resp = None
            for value in msg_send["args"]:
                if isinstance(value, PolicyResponse):
                    policy_resp = value
                    break
            for value in msg_send["kwargs"].values():
                if isinstance(value, PolicyResponse):
                    policy_resp = value
                    break

            if policy_resp is None:
                logger.error("No PolicyResponse found in msg_send")
                continue

            zmq_identity = policy_resp.env_id.split("/")[-1].encode()

            response = pickle.dumps(msg_send)
            socket.send_multipart([zmq_identity, b"", response])

    def reset(self, *args: Any, **kwargs: Any) -> List[EnvRet]:
        """Collect initial observations after environment initialization."""
        # msg_send = {"command": "reset", "args": args, "kwargs": kwargs}
        # self.send_msg_queue.put(msg_send)
        return self.collect_async()

    def _post_step_hook(self, env_ret):
        """Post-step hook to process EnvRet after reset.

        Do nothing on server side since post processing is done on client side.

        """
        if InternalFlag.REMOTE_STORAGE:
            env_ret = tuple(ret for ret in env_ret)  # use tuple for hashability
        if InternalFlag.REMOTE_ENV:
            env_ret = tuple(ray.put(ret) for ret in env_ret)

        return env_ret

    def step_async(self, policy_response_list: List[PolicyResponse]) -> None:
        """
        Step clients which env is in policy_response_list

        Args:
            policy_response_list (List[PolicyResponse]): list of PolicyResponse for clients

        Returns:
            None
        """
        policy_response_list = resolve_object(policy_response_list)
        for policy_response in policy_response_list:
            msg_send = {"command": "step", "args": (policy_response,), "kwargs": {}}
            self.send_msg_queue.put(msg_send)

    def collect_async(self, timeout: float | None = None) -> List[EnvRet]:
        """
        Collect clients' returns

        Args:
            timeout (float | None): timeout in seconds, None means blocking wait at least one return

        Returns:
            List[EnvRet]: list of EnvRet from clients
        """
        env_ret_list = []

        while True:
            try:
                first_ret = self.env_ret_queue.get(timeout=timeout)
                env_ret_list.append(first_ret)
                break
            except queue.Empty:
                break

        # try to get more env rets without blocking
        while True:
            try:
                item = self.env_ret_queue.get_nowait()
                env_ret_list.append(item)
            except queue.Empty:
                break

        return tuple(env_ret_list)  # for hashability

    def step(self, policy_resp: PolicyResponse) -> EnvRet:
        """Synchronous stepping is not supported on the server side."""
        raise NotImplementedError(
            "ZMQEnvServer does not support synchronous step. Please use step_async and collect_async."
        )

    def close(self) -> None:
        """Close the ZMQ server and background threads."""
        # clean up threads and zmq resources
        self.event.set()
        self.thread_pool.shutdown(wait=True)
        self.socket.close()
        self.context.term()
        logger.info("ZMQEnvServer closed.")

    def get_observation_space(self) -> Optional[Any]:
        """Remote server does not expose observation space directly."""
        return

    def get_action_space(self) -> Optional[Any]:
        """Remote server does not expose action space directly."""
        return
