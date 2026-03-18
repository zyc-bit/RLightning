"""Replay buffer implementation."""

from typing import Dict, List, Optional

from rlightning.utils.registry import BUFFERS

from .base_buffer import DataBuffer


@BUFFERS.register("ReplayBuffer")
class ReplayBuffer(DataBuffer):
    """Replay buffer usually for off-policy algorithms."""

    def sample(
        self, batch_size: Optional[int] = None, shuffle: Optional[bool] = True, drop_last: Optional[bool] = True
    ) -> List[Dict]:
        """
        Sample a batch of data from the buffer.
        """
        data_size_total = self.size()
        if batch_size is not None and batch_size > data_size_total:
            raise ValueError(
                "Not enough data in buffer to sample a batch of size "
                f"{batch_size}. Current buffer size: {data_size_total}."
                "Please increase the max rollout steps, or decrease the batch size."
            )

        data = super().sample(batch_size, shuffle, drop_last)
        return data
