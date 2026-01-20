"""
分布式工具函数（参考 VBench 的 distributed.py 实现）
提供视频列表分片和结果汇总功能
"""

import torch
import torch.distributed as dist
import pickle
from logging import getLogger

logger = getLogger(__name__)


def is_dist_initialized():
    """检查分布式是否已初始化"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """获取当前 rank"""
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size():
    """获取总 rank 数"""
    return dist.get_world_size() if is_dist_initialized() else 1


def print0(*args, **kwargs):
    """只在 rank0 打印（参考 VBench）"""
    if get_rank() == 0:
        print(*args, **kwargs)


def distribute_list_to_rank(data_list):
    """
    将列表按 rank 分片（参考 VBench 的实现）
    rank0 处理 [0, world_size, 2*world_size, ...]
    rank1 处理 [1, world_size+1, 2*world_size+1, ...]
    
    这样可以保证负载均衡
    """
    rank = get_rank()
    world_size = get_world_size()
    return data_list[rank::world_size]


def all_gather_object(data):
    """
    收集所有 rank 的数据（参考 VBench 的 all_gather）
    
    Args:
        data: 任意可 pickle 的对象
    
    Returns:
        list: 所有 rank 的数据列表
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    # 序列化为 tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).cuda()
    
    # 获取各 rank 的 tensor 大小
    local_size = torch.LongTensor([tensor.numel()]).cuda()
    size_list = [torch.LongTensor([0]).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    
    # Padding 到相同大小（all_gather 要求）
    if tensor.numel() < max_size:
        padding = torch.zeros(max_size - tensor.numel(), dtype=torch.uint8, device='cuda')
        tensor = torch.cat([tensor, padding])
    
    # All gather
    tensor_list = [torch.zeros(max_size, dtype=torch.uint8, device='cuda') for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    # 反序列化
    data_list = []
    for size, t in zip(size_list, tensor_list):
        buffer = t.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    
    return data_list


def merge_list_of_list(results):
    """展平嵌套列表（参考 VBench）"""
    return [item for sublist in results for item in sublist]


def all_gather(results):
    """
    收集并展平各 rank 的字典列表结果（参考 VBench）
    
    Args:
        results: list[dict]，当前 rank 的结果列表
    
    Returns:
        list[dict]: 所有 rank 的结果合并后的列表
    """
    results = all_gather_object(results)
    results = merge_list_of_list(results)
    return results


def gather_list_of_dict(results):
    """
    收集并合并各 rank 的字典列表（VBench 兼容函数名）
    与 all_gather 功能相同，只是函数名不同
    
    Args:
        results: list[dict]，当前 rank 的结果列表
    
    Returns:
        list[dict]: 所有 rank 的结果合并后的列表
    """
    return all_gather(results)


def barrier():
    """同步所有 rank"""
    if is_dist_initialized():
        dist.barrier()

