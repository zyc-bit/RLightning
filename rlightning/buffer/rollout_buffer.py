"""Rollout buffer implementation for on-policy data collection.

Defines a buffer that samples transitions/episodes, validates batch sizing,
and clears stored data after sampling. Integrates with the buffer registry
and supports sharded/distributed sampling via the base buffer.
"""

from typing import Dict, List, Optional

import ray

from rlightning.utils.logger import get_logger
from rlightning.utils.registry import BUFFERS

from .base_buffer import DataBuffer
from .sampler import AllDataSampler

logger = get_logger(__name__)


@BUFFERS.register("RolloutBuffer")
class RolloutBuffer(DataBuffer):
    """Rollout Buffer for on-policy algorithms"""

    def sample(
        self,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = True,
    ) -> List[Dict] | List[ray.ObjectRef]:
        """
        Sample a batch of data (transitions/truncated_episodes) from the buffer.

        Sharded storage support:
        - Each storage samples indices based on its own size.
        - DistributedSampler then splits those indices to the workers bound to that storage.

        Returns:
            dict mapping worker rank -> {"storage_idx": int, "indices": np.ndarray}.
        """

        if batch_size is not None:
            data_size_total = self.size()
            if not isinstance(self.sampler, AllDataSampler):
                if batch_size > data_size_total:
                    raise ValueError(
                        "Not enough data in buffer to sample a batch of size "
                        f"{batch_size}. Current buffer size: {data_size_total}."
                        "Please increase the max rollout steps, or decrease the batch size."
                    )
                elif batch_size < data_size_total:
                    logger.warning(
                        f"batch_size {batch_size} is smaller than data_size {data_size_total}. "
                        "Since you are using RolloutBuffer, the remaining unsampled data will be "
                        "discarded after sampling. If you want to use all data, please set "
                        "batch_size to None."
                    )

        sample_data = super().sample(batch_size, shuffle=shuffle, drop_last=drop_last)
        self.clear()
        return sample_data
