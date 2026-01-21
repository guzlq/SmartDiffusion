import torch
from logging import getLogger
from datetime import datetime

from .cal_function.distributed import (
    get_rank, get_world_size,
    barrier_world,  # NEW
    set_group, clear_group,  # NEW
)
from .video_collector import collect_videos_and_prompts
from .cal_function import VBench

logger = getLogger(__name__)


def run_vbench_evaluation(args):
    rank = get_rank()
    world_size = get_world_size()

    # 1) 等待视频生成完成：必须全局同步
    barrier_world()

    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"Starting VBench CUSTOM evaluation (world_size={world_size})")
        logger.info("=" * 60)

    try:
        _run_vbench_custom(args)
    except Exception as e:
        logger.error(f"[Rank {rank}] VBench evaluation failed: {e}", exc_info=True)

    # 2) 等待评测结束：必须全局同步
    barrier_world()

    if rank == 0:
        logger.info("VBench evaluation completed.")


def _run_vbench_custom(args):
    import torch.distributed as dist

    rank = get_rank()
    world_size = get_world_size()
    output_dir = "./vbench_out"

    # 1) rank0 收集 video_prompt/videos_dir
    if rank == 0:
        video_prompt, videos_dir = collect_videos_and_prompts(args)
        name = "vbench_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        num_videos = len(video_prompt) if video_prompt is not None else 0
    else:
        video_prompt, videos_dir, name, num_videos = None, None, None, 0

    # 2) 广播给所有 rank（全局组广播即可）
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        obj = [video_prompt, videos_dir, name, num_videos]
        dist.broadcast_object_list(obj, src=0)
        video_prompt, videos_dir, name, num_videos = obj

    if not video_prompt or not videos_dir or num_videos == 0:
        logger.error(f"[Rank {rank}] Missing video_prompt/videos_dir, skip.")
        return

    # 3) 关键：评测只用前 eval_world_size 个 rank，保证每个 rank 至少一个视频
    eval_world_size = min(world_size, num_videos)

    if rank == 0 and eval_world_size < world_size:
        logger.warning(
            f"VBench eval: num_videos={num_videos} < world_size={world_size}. "
            f"Create eval subgroup with eval_world_size={eval_world_size}; "
            f"ranks [{eval_world_size}, {world_size-1}] will skip evaluation."
        )

    eval_group = None
    if world_size > 1 and eval_world_size < world_size:
        # 注意：new_group 最稳妥做法是所有 rank 都调用（即使不在组内）
        eval_ranks = list(range(eval_world_size))
        eval_group = dist.new_group(ranks=eval_ranks)

    # 不在 eval 子组的 rank：直接跳过评测，但仍会在 run_vbench_evaluation 末尾 barrier_world 对齐
    if rank >= eval_world_size:
        logger.info(f"[Rank {rank}] Skip VBench (no video assigned in eval subgroup).")
        return

    # 4) 让 VBench 内部的 distribute/gather/barrier 都走 eval_group
    if eval_group is not None:
        set_group(eval_group)
    else:
        clear_group()  # world_size==eval_world_size 或单卡时，回到默认 WORLD

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vb = VBench(device=device, output_path=output_dir)

        vb.evaluate(
            videos_path=videos_dir,
            name=name,
            video_prompt=video_prompt,
            dimension_list=None,
            local=True,
            read_frame=False,
        )
    finally:
        # 评测结束，务必恢复默认 group，避免影响后续全局 barrier_world / 其他 collectives
        clear_group()
