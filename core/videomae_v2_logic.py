import os
from typing import List, Optional


class VideoMAEHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.labels: List[str] = []
        self.weights_path: Optional[str] = None
        self.verb_list_path: Optional[str] = None
        self._yaml = None
        self._torch = None
        self._Image = None
        self._np = None
        self._decord = None
        self._VideoMAEImageProcessor = None
        self._VideoMAEForVideoClassification = None
        self._VideoMAEConfig = None

    def _ensure_runtime(self):
        if (
            self._yaml is not None
            and self._torch is not None
            and self._Image is not None
            and self._np is not None
            and self._decord is not None
            and self._VideoMAEImageProcessor is not None
            and self._VideoMAEForVideoClassification is not None
        ):
            if self.device is None:
                self.device = self._torch.device(
                    "cuda" if self._torch.cuda.is_available() else "cpu"
                )
            return True, None
        try:
            import yaml
            import torch
            from PIL import Image
            import numpy as np
            import decord
            from transformers import (
                VideoMAEForVideoClassification,
                VideoMAEImageProcessor,
                VideoMAEConfig,
            )
        except Exception as ex:
            return (
                False,
                "VideoMAE optional dependencies are missing. "
                "Install Pillow, PyYAML, transformers, and decord to enable action ranking.\n"
                f"{ex}",
            )
        self._yaml = yaml
        self._torch = torch
        self._Image = Image
        self._np = np
        self._decord = decord
        self._VideoMAEImageProcessor = VideoMAEImageProcessor
        self._VideoMAEForVideoClassification = VideoMAEForVideoClassification
        self._VideoMAEConfig = VideoMAEConfig
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return True, None

    def load_model(self, weights_path, verb_list_path):
        """Load VideoMAE V2 model and custom labels."""
        ok, err = self._ensure_runtime()
        if not ok:
            self.model = None
            self.processor = None
            return False, err

        self.weights_path = weights_path
        self.verb_list_path = verb_list_path

        with open(verb_list_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        data = self._yaml.safe_load(raw_text)
        if isinstance(data, dict) and "names" in data:
            names = data["names"]
            try:
                ordered_keys = sorted(names.keys(), key=lambda x: int(x))
            except Exception:
                ordered_keys = sorted(names.keys(), key=lambda x: str(x))
            self.labels = [str(names[i]) for i in ordered_keys]
        elif isinstance(data, list):
            self.labels = [str(v) for v in data]
        elif isinstance(data, str):
            self.labels = [line.strip() for line in raw_text.splitlines() if line.strip()]
        else:
            self.labels = []

        num_labels = len(self.labels)
        
        # Absolute path discovery for project root (IMPACT_HOI(1)/)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_dir = os.path.join(root_dir, "weights", "videomae")
        
        if os.path.exists(os.path.join(local_dir, "config.json")):
            model_id = local_dir
            print(f"VideoMAE: Using local config from {model_id}")
        else:
            model_id = "OpenGVLab/video-mae-v2-base-kinetics"
            print(f"VideoMAE: Local config not found at {local_dir}, attempting HF Hub...")

        try:
            self.processor = self._VideoMAEImageProcessor.from_pretrained(model_id)
            # Offline fix: Load config only, weights will be loaded from .pth subsequently
            config = self._VideoMAEConfig.from_pretrained(model_id, num_labels=num_labels)
            self.model = self._VideoMAEForVideoClassification(config)
            if weights_path and os.path.exists(weights_path):
                state_dict = self._torch.load(weights_path, map_location=self.device)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                # Check for key mismatches between official repo and transformers
                new_state_dict = {}
                for k, v in state_dict.items():
                    nk = k
                    if k.startswith("encoder.blocks."):
                        nk = k.replace("encoder.blocks.", "videomae.encoder.layer.")
                    elif k.startswith("encoder."):
                        nk = k.replace("encoder.", "videomae.encoder.")
                    elif not k.startswith("videomae.") and not k.startswith("classifier."):
                        # Assume it's a backbone weight missing the prefix
                        nk = f"videomae.{k}"
                    new_state_dict[nk] = v
                
                self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            return True, f"Loaded model with {num_labels} labels."
        except Exception as ex:
            self.model = None
            self.processor = None
            return False, str(ex)

    def predict(self, video_path, start_frame, end_frame, top_k=5):
        """Predict action labels for a given segment."""
        ok, err = self._ensure_runtime()
        if not ok:
            return None, err
        if self.model is None or self.processor is None:
            return None, "Model not loaded."

        try:
            vr = self._decord.VideoReader(video_path, num_threads=1)
            num_frames_available = len(vr)
            start = max(0, int(start_frame))
            end = min(num_frames_available - 1, int(end_frame))
            num_samples = 16
            indices = self._np.linspace(start, end, num_samples).astype(int)
            video = vr.get_batch(indices).asnumpy()
            frames = [self._Image.fromarray(frame) for frame in video]
            inputs = self.processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = self._torch.nn.functional.softmax(logits, dim=-1)
                top_probs, top_indices = self._torch.topk(
                    probs, k=min(top_k, len(self.labels))
                )

            results = []
            for i in range(top_probs.size(1)):
                idx = top_indices[0, i].item()
                results.append(
                    {
                        "label": self.labels[idx]
                        if idx < len(self.labels)
                        else f"Unknown_{idx}",
                        "score": top_probs[0, i].item(),
                    }
                )
            return results, None
        except Exception as ex:
            return None, str(ex)
