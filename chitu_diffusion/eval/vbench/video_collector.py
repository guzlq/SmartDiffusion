"""
收集生成的视频文件，构建 VBench 评测所需的 video_prompt 字典
直接从 DiffusionTaskPool 获取任务信息，使用 SmartDiffusion 的命名规则生成视频路径

输出:
- video_prompt: { "vid1.mp4": "prompt", ... }
- videos_dir:   视频所在目录（要求所有 task.save_dir 一致）
"""

import os
from pathlib import Path
from logging import getLogger
from typing import Dict, Tuple

logger = getLogger(__name__)


def collect_videos_and_prompts(args) -> Tuple[Dict[str, str], str]:
    """
    从 DiffusionTaskPool 收集已完成任务的视频，并返回 VBench 需要的字典形式

    SmartDiffusion 命名规则（generator.py:360-361）：
      save_name = prompt[:20].replace(" ", "_").replace(".", "") + f"_{task_id}.mp4"
      保存目录 = task.req.params.save_dir

    Returns:
        video_prompt: dict, key 是视频文件名（含后缀），value 是对应 prompt
        videos_dir: str, 视频所在目录（绝对路径）
    """
    completed_tasks = _get_completed_tasks_from_pool()
    if not completed_tasks:
        raise ValueError("No completed tasks found in DiffusionTaskPool")

    logger.info(f"Found {len(completed_tasks)} completed tasks")

    # 收集所有 save_dir，确保一致（否则 VBench 的 videos_path + filename 定位会失效）
    save_dirs = {str(Path(task.req.params.save_dir).resolve()) for task in completed_tasks}
    if len(save_dirs) != 1:
        raise ValueError(
            f"Completed tasks have different save_dir values: {sorted(save_dirs)}. "
            "VBench expects a single videos_dir. Please unify save_dir or copy videos into one folder."
        )
    videos_dir = next(iter(save_dirs))

    video_prompt: Dict[str, str] = {}
    missing_files = 0

    for task in completed_tasks:
        prompt = task.req.get_prompt()
        task_id = task.task_id
        save_dir = str(Path(task.req.params.save_dir).resolve())

        # 命名逻辑：严格与 generator.py 一致
        save_name = prompt[:20].replace(" ", "_").replace(".", "") + f"_{task_id}.mp4"
        video_path = os.path.join(save_dir, save_name)

        if not os.path.exists(video_path):
            missing_files += 1
            logger.warning(f"Video file not found: {video_path}")
            continue

        # key 用“文件名（含后缀）”，适配 VBench: videos_path(目录) + video_name(key)
        if save_name in video_prompt:
            # 理论上 task_id 不同不会撞名，这里防呆
            logger.warning(f"Duplicate video name detected: {save_name}. Overwriting previous prompt.")
        video_prompt[save_name] = prompt

        logger.debug(f"Collected: {save_name} -> {prompt[:50]}...")

    if not video_prompt:
        raise ValueError("No video files found for completed tasks (all missing on disk?)")

    logger.info(
        f"Successfully collected {len(video_prompt)} videos. "
        f"Missing files: {missing_files}. videos_dir={videos_dir}"
    )
    return video_prompt, videos_dir


def _get_completed_tasks_from_pool():
    """
    从 DiffusionTaskPool 获取所有已完成的任务
    """
    from chitu_diffusion.task import DiffusionTaskPool, DiffusionTaskStatus

    completed_tasks = []
    for task_id, task in DiffusionTaskPool.pool.items():
        if task.status == DiffusionTaskStatus.Completed:
            completed_tasks.append(task)
        else:
            logger.debug(f"Skipping task {task_id} with status {task.status}")

    return completed_tasks
