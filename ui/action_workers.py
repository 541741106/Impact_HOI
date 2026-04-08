import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from utils.feature_env import build_runner_cmd, load_feature_env_defaults
from utils.optional_deps import (
    MissingOptionalDependency,
    format_missing_dependency_message,
    import_optional_module,
)


ACTION_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_feature_env_defaults(repo_root=ACTION_REPO_ROOT)


def load_feature_extractor_module():
    return import_optional_module(
        "tools.feature_extractors",
        feature_name="Feature extraction / ASOT pre-labeling",
        install_hint=(
            "Install the optional feature-extraction dependencies first, "
            "for example: pip install torch torchvision"
        ),
    )


def _run_streaming_command(cmd, progress_cb):
    log_lines = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip()
                log_lines.append(line + "\n")
                progress_cb(line)
    finally:
        proc.wait()
    return proc.returncode, log_lines


def _write_worker_log(log_path: str, log_lines, progress_cb) -> None:
    if not log_path:
        return
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_lines)
        progress_cb(f"[LOG] Saved to {log_path}")
    except Exception as ex:
        progress_cb(f"[WARN] failed to write log: {ex}")


class ASOTInferWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(str, str)  # (txt_path, json_path)

    def __init__(
        self,
        features_dir,
        ckpt,
        class_names=None,
        smooth_k=1,
        standardize=False,
        out_prefix="pred_asot",
        tool_path="asot_infer_adapter.py",
        allow_standardize=True,
        extra_args=None,
        log_path=None,
    ):
        super().__init__()
        self.features_dir = features_dir
        self.ckpt = ckpt
        self.class_names = class_names
        self.smooth_k = smooth_k
        self.standardize = standardize
        self.allow_standardize = allow_standardize
        self.out_prefix = out_prefix
        self.tool_path = tool_path
        self.extra_args = extra_args or []
        self.log_path = log_path or os.path.join(
            features_dir, f"{out_prefix}_infer.log"
        )

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="asot",
                script_path=self.tool_path,
                script_args=[
                    "--features_dir",
                    self.features_dir,
                    "--ckpt",
                    self.ckpt,
                    "--out_prefix",
                    self.out_prefix,
                ],
                python_executable=sys.executable,
            )
            if self.class_names:
                cmd += ["--class_names", self.class_names]
            if self.standardize and self.allow_standardize:
                cmd.append("--standardize")
            if self.smooth_k and int(self.smooth_k) > 1:
                cmd += ["--smooth_k", str(int(self.smooth_k))]
            if self.extra_args:
                cmd += list(self.extra_args)

            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            txt_path = os.path.join(
                self.features_dir, f"{self.out_prefix}_segments.txt"
            )
            json_path = os.path.join(
                self.features_dir, f"{self.out_prefix}_segments.json"
            )
            ok = (returncode == 0) and os.path.isfile(txt_path)
            self.done.emit(txt_path if ok else "", json_path if ok else "")
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit("", "")
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)


class ASOTRemapBuildWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)  # (ok, output_json)

    def __init__(
        self,
        roots: Sequence[str],
        *,
        output_json: str,
        class_names: str = "",
        feature_search_roots: Optional[Sequence[str]] = None,
        pred_prefix: str = "pred_asot",
        tool_path: str = "",
        log_path: Optional[str] = None,
    ):
        super().__init__()
        self.roots = [str(x) for x in (roots or []) if str(x or "").strip()]
        self.output_json = str(output_json or "")
        self.class_names = str(class_names or "")
        self.feature_search_roots = [
            str(x) for x in (feature_search_roots or []) if str(x or "").strip()
        ]
        self.pred_prefix = str(pred_prefix or "pred_asot")
        self.tool_path = str(tool_path or "")
        self.log_path = log_path or (
            os.path.join(os.path.dirname(self.output_json), "asot_label_remap_build.log")
            if self.output_json
            else ""
        )

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="current",
                script_path=self.tool_path,
                script_args=[
                    *self.roots,
                    "--pred-prefix",
                    self.pred_prefix,
                    "--output-json",
                    self.output_json,
                ],
                python_executable=sys.executable,
            )
            if self.class_names:
                cmd += ["--class-names", self.class_names]
            for path in self.feature_search_roots:
                cmd += ["--feature-search-root", path]
            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            ok = (returncode == 0) and os.path.isfile(self.output_json)
            self.done.emit(ok, self.output_json)
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit(False, self.output_json)
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)






class FactBatchWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)  # (ok, output_dir)

    def __init__(
        self,
        video_dir,
        output_dir,
        fact_repo,
        ckpt,
        fact_cfg,
        tool_path,
        class_names=None,
        log_path=None,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.fact_repo = fact_repo
        self.ckpt = ckpt
        self.fact_cfg = fact_cfg
        self.tool_path = tool_path
        self.class_names = class_names
        self.log_path = log_path or os.path.join(output_dir, "pred_fact_batch.log")

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="current",
                script_path=self.tool_path,
                script_args=[
                    "--video_dir",
                    self.video_dir,
                    "--output_dir",
                    self.output_dir,
                    "--fact_repo",
                    self.fact_repo,
                    "--fact_cfg",
                    self.fact_cfg,
                    "--ckpt",
                    self.ckpt,
                ],
                python_executable=sys.executable,
            )
            if self.class_names:
                cmd += ["--class_names", self.class_names]

            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            self.done.emit(returncode == 0, self.output_dir)
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit(False, self.output_dir)
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)




class FeatureExtractWorker(QObject):
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int, int)
    done = pyqtSignal(object, bool)  # (features_dir, ok)

    def __init__(
        self,
        video_path: str,
        features_dir: str,
        batch_size: int = 128,
        frame_stride: int = 1,
        use_fp16: bool = True,
        backbone: Optional[str] = None,
    ):
        super().__init__()
        self.video_path = video_path
        self.features_dir = features_dir
        self.batch_size = batch_size
        self.frame_stride = frame_stride
        self.use_fp16 = use_fp16
        self.backbone = str(backbone or '').strip()

    def run(self):
        try:
            feat_path = os.path.join(self.features_dir, 'features.npy')
            os.makedirs(self.features_dir, exist_ok=True)
            backbone = self.backbone or os.environ.get(
                'FEATURE_BACKBONE', 'dinov2_vitb14'
            )
            self.progress.emit(f'[FEATS] Extracting {backbone} features to {feat_path}')
            feature_api = load_feature_extractor_module()

            def _emit_progress(done: int, total: int):
                self.progress_value.emit(done, total)

            feats, meta = feature_api.extract_video_features(
                self.video_path,
                backbone=backbone,
                batch_size=self.batch_size,
                frame_stride=max(1, self.frame_stride),
                use_fp16=self.use_fp16,
                progress_cb=_emit_progress,
            )
            if bool(meta.get('model_cached', False)):
                self.progress.emit(f'[FEATS] Reused cached {backbone} model.')
            else:
                self.progress.emit(f'[FEATS] Loaded {backbone} model.')
            feature_api.save_features(self.features_dir, feats, meta=meta)
            self.progress.emit(f'[FEATS] Saved features {feats.shape} to {feat_path}')
            self.done.emit(self.features_dir, True)
        except MissingOptionalDependency as ex:
            self.progress.emit(f'[FEATS][ERROR] {format_missing_dependency_message(ex)}')
            self.done.emit(None, False)
        except Exception as ex:
            self.progress.emit(f'[FEATS][ERROR] {ex}')
            self.done.emit(None, False)




