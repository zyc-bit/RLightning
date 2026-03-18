"""Remote environment client for ZMQ-based environment control."""

import pickle
from typing import Any, Dict, Optional

import zmq

from rlightning.env.base_env import BaseEnv
from rlightning.types import EnvRet
from rlightning.utils.logger import get_logger
from rlightning.utils.zmq import communication

logger = get_logger(__name__)


class RemoteEnvClient:
    """ZMQ client that proxies commands to a local environment."""

    def __init__(self, env: BaseEnv, hostname: str, port: int) -> None:
        """Initialize the remote env client."""
        self.env: BaseEnv = env
        self.hostname = hostname
        self.port = port

        self.zmq_identity = None

        self.socket: Optional[zmq.Socket] = None

    def connect(self) -> None:
        """Connect to the remote env server."""

        localhost = communication.get_local_ip()
        context = zmq.Context()
        self.socket = context.socket(zmq.DEALER)
        self.zmq_identity = f"{localhost}-{self.env.get_env_id()}"

        self.socket.setsockopt(zmq.IDENTITY, self.zmq_identity.encode())

        logger.info(f"Connect to {self.hostname}:{self.port}")
        self.socket.connect(f"tcp://{self.hostname}:{self.port}")

        logger.debug(f"ZMQ DEALER socket connected with identity: {self.zmq_identity}")

    def close(self) -> None:
        """Close the connection to the env server."""
        msg_send = {
            "client_state": "closed",
            "env_ret": None,
        }
        self.socket.send_multipart([b"", pickle.dumps(msg_send)])

        logger.info(f"{communication.get_local_ip()}  Closing ZmqEnv connection...")
        self.socket.close()

    def waiting_response(self, block: bool = True) -> bytes:
        """Wait for a response from the server."""
        _, response = self.socket.recv_multipart()
        return response

    def reset(self, *args: Any, **kwargs: Any) -> EnvRet:
        """Placeholder reset (handled by resolve_command)."""
        return EnvRet()

    def step(self, *args: Any, **kwargs: Any) -> EnvRet:
        """Placeholder step (handled by resolve_command)."""
        return EnvRet()

    def resolve_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command and return a response payload."""
        valid_commands = ["step", "reset"]
        command, args, kwargs = message["command"], message["args"], message["kwargs"]
        if command not in valid_commands:
            logger.error(f"Invalid command received: {command}")
            raise ValueError(f"Invalid command: {command}")

        if command == "step":
            logger.debug("Received step command")
            env_ret: EnvRet = self.env._step(*args, **kwargs)
        elif command == "reset":
            logger.debug("Received reset command")
            env_ret: EnvRet = self.env._reset(*args, **kwargs)

        return {
            "client_state": "running",
            "env_ret": env_ret,
        }

    def run(self) -> None:
        """Run the client loop to service remote commands."""
        # initial reset
        msg_send = self.resolve_command(
            {
                "command": "reset",
                "args": (),
                "kwargs": {},
            }
        )
        self.socket.send_multipart([b"", pickle.dumps(msg_send)])

        while True:
            # recv
            _, msg_recv = self.socket.recv_multipart()  # pylint: disable=W0632
            msg_recv = pickle.loads(msg_recv)

            msg_send = self.resolve_command(msg_recv)

            self.socket.send_multipart([b"", pickle.dumps(msg_send)])
            # send
            if self.env.is_finish():
                self.close()
                break
