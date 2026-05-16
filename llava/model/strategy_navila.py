import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

class SimpleFSDPWrapper:
    def __init__(self, model):
        self.model = model

    def wrap(self):
        """最简单的包装方法"""
        return FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=torch.cuda.current_device(),
        )

# 使用：
# wrapper = SimpleFSDPWrapper(model)
# fsdp_model = wrapper.wrap()