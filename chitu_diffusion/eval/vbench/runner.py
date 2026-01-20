"""
VBench Custom 评测启动器
- 所有 rank 参与（多卡并行）
- rank0 收集 video_prompt/videos_dir 并广播
- VBench 内部各维度 compute_xxx 负责 distribute/gather
- VBench.evaluate 内部负责：rank0 写 meta_json + barrier（你已改好）
"""

import torch
from logging import getLogger
from datetime import datetime

from .cal_function.distributed import get_rank, get_world_size, barrier
from .video_collector import collect_videos_and_prompts
from .cal_function import VBench

logger = getLogger(__name__)


def run_vbench_evaluation(args):
    """
    检查是否启用 VBench，如果启用则执行 custom 评测
    """
    rank = get_rank()
    world_size = get_world_size()

    # 1) 等待视频生成完成
    barrier()

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"Starting VBench CUSTOM evaluation (world_size={world_size})")
        logger.info("=" * 60)

    try:
        _run_vbench_custom(args)
    except Exception as e:
        logger.error(f"[Rank {rank}] VBench evaluation failed: {e}", exc_info=True)
        # 不阻断主流程

    # 2) 等待评测结束
    barrier()

    if rank == 0:
        logger.info("VBench evaluation completed.")


def _run_vbench_custom(args):
    rank = get_rank()
    world_size = get_world_size()
    output_dir = "./vbench_out"
    # (可选) 如果你还需要把 vbench repo 加到 sys.path，就保留
    # setup_vbench_path(vbench_config.vbench_repo)

    # 1) rank0 收集 video_prompt/videos_dir
    if rank == 0:
        video_prompt, videos_dir = collect_videos_and_prompts(args)
        # name 统一：所有 rank 必须一致
        name = "vbench_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        video_prompt, videos_dir, name = None, None, None

    # 2) 广播给所有 rank
    if world_size > 1:
        obj = [video_prompt, videos_dir, name]
        torch.distributed.broadcast_object_list(obj, src=0)
        video_prompt, videos_dir, name = obj

    if not video_prompt or not videos_dir:
        logger.error(f"[Rank {rank}] Missing video_prompt/videos_dir, skip.")
        return

    # 3) 调用 VBench custom
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vb = VBench(device=device, output_path=output_dir)

    # 这些 kwargs 会透传到 compute_xxx（比如 imaging_quality 的预处理模式）
    vb.evaluate(
        videos_path=videos_dir,
        name=name,
        video_prompt=video_prompt,
        # dimension_list=None -> 默认所有 custom 可测维度
        dimension_list=None,
        local= True,
        read_frame=False,
        
    )
