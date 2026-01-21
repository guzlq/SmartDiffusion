"""
分布式工具函数（参考 VBench 的 distributed.py 实现）
提供视频列表分片和结果汇总功能
支持：评测子进程组（subgroup），用于 num_videos < world_size 的情况
"""

import torch
import torch.distributed as dist
import pickle
from logging import getLogger

logger = getLogger(__name__)

# -----------------------
# NEW: 可切换的默认通信 group
# -----------------------
_CURRENT_GROUP = None  # None 表示使用 dist.group.WORLD（默认全局组）

def set_group(group):
    """设置当前 distributed 工具函数使用的进程组（用于评测子组）"""
    global _CURRENT_GROUP
    _CURRENT_GROUP = group

def clear_group():
    """清空当前进程组，回退到默认全局组"""
    global _CURRENT_GROUP
    _CURRENT_GROUP = None

def _resolve_group(group=None):
    """解析实际使用的 group：优先显式传入，其次 _CURRENT_GROUP，最后 WORLD"""
    if group is not None:
        return group
    if _CURRENT_GROUP is not None:
        return _CURRENT_GROUP
    return dist.group.WORLD


def is_dist_initialized():
    """检查分布式是否已初始化"""
    return dist.is_available() and dist.is_initialized()


def get_rank(group=None):
    """获取当前 rank（基于指定/当前 group）"""
    if not is_dist_initialized():
        return 0
    g = _resolve_group(group)
    return dist.get_rank(group=g)


def get_world_size(group=None):
    """获取总 rank 数（基于指定/当前 group）"""
    if not is_dist_initialized():
        return 1
    g = _resolve_group(group)
    return dist.get_world_size(group=g)


def print0(*args, **kwargs):
    """只在 rank0 打印（基于当前 group）"""
    if get_rank() == 0:
        print(*args, **kwargs)


def distribute_list_to_rank(data_list):
    """
    将列表按 rank 分片（基于当前 group 的 rank/world_size）
    rank0 处理 [0, world_size, 2*world_size, ...]
    """
    rank = get_rank()
    world_size = get_world_size()
    return data_list[rank::world_size]


def _default_comm_device():
    # all_gather_object 需要一个 device 来放 byte tensor
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def all_gather_object(data, group=None, device=None):
    """
    收集所有 rank 的数据（可 pickle），支持 subgroup
    """
    g = _resolve_group(group)
    world_size = get_world_size(group=g)
    if world_size == 1:
        return [data]

    if device is None:
        device = _default_comm_device()

    # 序列化为 byte tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # 获取各 rank 的 tensor 大小
    local_size = torch.LongTensor([tensor.numel()]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=g)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # padding 到相同大小
    if tensor.numel() < max_size:
        padding = torch.zeros(max_size - tensor.numel(), dtype=torch.uint8, device=device)
        tensor = torch.cat([tensor, padding], dim=0)

    # all_gather
    tensor_list = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=g)

    # 反序列化
    data_list = []
    for size, t in zip(size_list, tensor_list):
        buf = t.detach().cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buf))
    return data_list


def merge_list_of_list(results):
    """展平嵌套列表"""
    return [item for sublist in results for item in sublist]


def all_gather(results):
    """
    收集并展平各 rank 的字典列表结果（支持 subgroup）
    """
    results = all_gather_object(results)
    results = merge_list_of_list(results)
    return results


def gather_list_of_dict(results):
    """VBench 兼容函数名"""
    return all_gather(results)


def barrier(group=None):
    """同步所有 rank（支持 subgroup）"""
    if is_dist_initialized():
        g = _resolve_group(group)
        dist.barrier(group=g)


# -----------------------
# NEW: 全局 barrier（永远用 WORLD）
# 用于评测前后跟生成流程对齐，避免 subgroup barrier 影响主流程
# -----------------------
def barrier_world():
    if is_dist_initialized():
        dist.barrier(group=dist.group.WORLD)
