import os
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Union

from .utils import init_submodules, save_json
from .distributed import get_rank, print0


class VBench:
    """
    Custom-only VBench runner (directory-only, strict video_prompt keys).

    You provide:
      - videos_path: a DIRECTORY path (root)
      - video_prompt: dict { "vid1.mp4": "prompt text", ... }  # exact filenames with suffix

    Behavior:
      - ONLY evaluates videos listed in video_prompt
      - DEFAULT evaluates all dimensions supported in custom mode
    """

    _FULL_DIMS = [
        "subject_consistency",
        "background_consistency",
        "aesthetic_quality",
        "imaging_quality",
        "object_class",
        "multiple_objects",
        "color",
        "spatial_relationship",
        "scene",
        "temporal_style",
        "overall_consistency",
        "human_action",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "appearance_style",
    ]

    _CUSTOM_DIMS = [
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
    ]

    def __init__(self, device: str, output_path: str):
        self.device = device
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    @classmethod
    def default_custom_dimensions(cls) -> List[str]:
        return [d for d in cls._CUSTOM_DIMS]

    @staticmethod
    def _resolve_videos(
        videos_path: Union[str, Path],
        video_prompt: Dict[str, str],
    ) -> List[Path]:
     

        base = Path(videos_path)
        vids: List[Path] = []
        missing: List[str] = []

        for video_name in sorted(video_prompt.keys()):
            p = (base / video_name)
            if not p.exists() or not p.is_file():
                missing.append(video_name)
            else:
                vids.append(p.resolve())

        if missing:
            raise FileNotFoundError(
                f"These videos listed in video_prompt were not found under '{base}': {missing}"
            )

        return vids

    def build_custom_full_info_json(
        self,
        videos_path: Union[str, Path],
        name: str,
        video_prompt: Dict[str, str],
        dimension_list: Optional[List[str]] = None,
    ) -> str:
        if dimension_list is None:
            dimension_list = self.default_custom_dimensions()

        bad = [d for d in dimension_list if d not in self._CUSTOM_DIMS]
        if bad:
            raise ValueError(f"These dimensions are NOT supported in custom mode: {sorted(bad)}")

        videos = self._resolve_videos(videos_path, video_prompt)

        cur_full_info_list = []
        for vp in videos:
            prompt = video_prompt[vp.name]
            if not isinstance(prompt, str):
                raise TypeError(f"video_prompt['{vp.name}'] must be a str prompt")

            cur_full_info_list.append(
                {
                    "prompt_en": prompt,
                    "dimension": dimension_list,
                    "video_list": [str(vp)],
                }
            )

        out_json = os.path.join(self.output_path, f"{name}_full_info.json")
        save_json(cur_full_info_list, out_json)
        print0(f"Custom evaluation meta saved to {out_json}")
        return out_json

    def evaluate(
        self,
        videos_path: Union[str, Path],
        name: str,
        video_prompt: Dict[str, str],
        dimension_list: Optional[List[str]] = None,
        local: bool = False,
        read_frame: bool = False,
        **kwargs,
    ):
        # 需要 barrier + world_size
        from .distributed import get_world_size, barrier

        if dimension_list is None:
            dimension_list = self.default_custom_dimensions()

        # 初始化子模块：每个 rank 都需要各自加载模型（正常）
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        # 关键：meta_json 路径所有 rank 一致
        meta_json = os.path.join(self.output_path, f"{name}_full_info.json")

        # 只让 rank0 写 meta json，避免并发写
        if get_rank() == 0:
            self.build_custom_full_info_json(
                videos_path=videos_path,
                name=name,
                video_prompt=video_prompt,
                dimension_list=dimension_list,
            )

        # 等待 rank0 写完，所有 rank 再开始读 meta_json 评测
        if get_world_size() > 1:
            barrier()

        # 开始逐维度评测（compute_xxx 内部会 distribute/gather）
        results_dict = {}
        for dim in dimension_list:
            try:
                # 使用相对导入，因为维度模块在同一包下
                dim_module = importlib.import_module(f".{dim}", package="chitu_diffusion.eval.vbench.cal_function")
                compute_fn = getattr(dim_module, f"compute_{dim}")
            except Exception as e:
                raise NotImplementedError(f"Dimension {dim} not implemented/importable: {e}")

            submods = submodules_dict[dim]
            print0(f"[VBench custom] running {dim} ...")
            results = compute_fn(meta_json, self.device, submods, **kwargs)
            results_dict[dim] = results

        # 只在 rank0 保存最终结果（合理）
        out_path = os.path.join(self.output_path, f"{name}_eval_results.json")
        if get_rank() == 0:
            save_json(results_dict, out_path)
            print0(f"Evaluation results saved to {out_path}")

        # 可选：再 barrier 一下，确保 rank0 保存完（不是必须）
        # if get_world_size() > 1:
        #     barrier()

        return results_dict

