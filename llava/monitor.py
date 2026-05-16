import torch
import torch.distributed as dist
import time
import psutil
import os
from datetime import datetime


class DDPMemoryMonitor:
    def __init__(self, log_interval=100, log_file=None):
        self.log_interval = log_interval
        self.log_file = log_file
        self.step_count = 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

    def get_gpu_memory_info(self):
        """获取单个GPU的显存信息"""
        if not torch.cuda.is_available():
            return {}

        torch.cuda.synchronize()
        device = torch.device(f'cuda:{self.local_rank}')

        # 当前显存使用
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3

        # 缓存分配器统计
        stats = torch.cuda.memory_stats(device)
        active_bytes = stats.get('active_bytes.all.current', 0) / 1024 ** 3
        inactive_bytes = stats.get('inactive_bytes.all.current', 0) / 1024 ** 3

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'active_gb': active_bytes,
            'inactive_gb': inactive_bytes
        }

    def get_system_memory_info(self):
        """获取系统内存信息"""
        memory = psutil.virtual_memory()
        return {
            'system_memory_used_gb': memory.used / 1024 ** 3,
            'system_memory_total_gb': memory.total / 1024 ** 3,
            'system_memory_percent': memory.percent
        }

    def gather_distributed_info(self, local_info):
        """在分布式环境中收集所有rank的信息"""
        if not dist.is_initialized() or self.world_size == 1:
            return {0: local_info}

        # 将所有rank的信息收集到rank 0
        gathered_info = [None] * self.world_size
        dist.all_gather_object(gathered_info, local_info)

        return {i: info for i, info in enumerate(gathered_info)}

    def log_memory_info(self, phase="training"):
        """记录内存信息"""
        self.step_count += 1

        # 只在特定步骤记录
        if self.step_count % self.log_interval != 0 and phase != "start" and phase != "end":
            return

        gpu_info = self.get_gpu_memory_info()
        system_info = self.get_system_memory_info()

        # 收集所有rank的信息
        all_rank_info = self.gather_distributed_info({
            'gpu': gpu_info,
            'system': system_info,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'phase': phase,
            'step': self.step_count
        })

        # 只在rank 0打印和记录
        if self.local_rank == 0:
            self._print_memory_report(all_rank_info, phase)
            # if self.log_file:
                # self._write_to_log(all_rank_info, phase)

    def _print_memory_report(self, all_rank_info, phase):
        """打印内存报告"""
        print(f"\n{'=' * 60}")
        print(f"内存监控报告 - Phase: {phase} - Step: {self.step_count}")
        print(f"{'=' * 60}")

        for rank, info in all_rank_info.items():
            gpu_info = info['gpu']
            system_info = info['system']

            print(f"Rank {rank}:")
            if gpu_info:
                print(f"  GPU显存: {gpu_info['allocated_gb']:.2f}GB / "
                      f"保留: {gpu_info['reserved_gb']:.2f}GB / "
                      f"峰值: {gpu_info['max_allocated_gb']:.2f}GB")
                print(f"  活跃内存: {gpu_info['active_gb']:.2f}GB / "
                      f"非活跃: {gpu_info['inactive_gb']:.2f}GB")
            print(f"  系统内存: {system_info['system_memory_used_gb']:.2f}GB/"
                  f"{system_info['system_memory_total_gb']:.2f}GB "
                  f"({system_info['system_memory_percent']:.1f}%)")
        print(f"{'=' * 60}\n")

    def _write_to_log(self, all_rank_info, phase):
        """写入日志文件"""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{datetime.now().isoformat()}, Phase: {phase}, Step: {self.step_count}\n")
            for rank, info in all_rank_info.items():
                gpu_info = info['gpu']
                if gpu_info:
                    f.write(f"Rank{rank}: GPU{rank}: {gpu_info['allocated_gb']:.2f}GB, "
                            f"Peak: {gpu_info['max_allocated_gb']:.2f}GB\n")