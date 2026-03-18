import socket
from typing import Tuple

import zmq


def get_local_ip() -> str:
    return [
        l
        for l in (
            [
                ip
                for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                if not ip.startswith("127.")
            ][:1],
            [
                [
                    (s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close())
                    for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]
                ][0][1]
            ],
        )
        if l
    ][0][0]


def get_free_port() -> int:
    """Return a free port for socket communication

    Returns:
        int: Port number
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to an available port
        return s.getsockname()[1]  # Get the assigned port number


def create_socket(port: int = 5566) -> Tuple[zmq.Context, zmq.Socket]:
    """Create a socket and return the tuple of context and socket

    Args:
        port (int, optional): Port number for zmq service. Defaults to 5566.

    Returns:
        Tuple[zmq.Context, zmq.Socket]: A tuple of context and socket
    """

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{port}")
    # print(f"Rollout service started at tcp://*:{port}")

    return context, socket
