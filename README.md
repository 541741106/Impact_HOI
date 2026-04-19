# IMPACT-HOI: Supervisory Control for Onset-Anchored Partial HOI Event Construction

IMPACT-HOI is a mixed-initiative annotation system for hand-object interaction (HOI) event construction in egocentric procedural video. Instead of predicting a full event in one shot, the system maintains a partially specified per-hand event state, proposes onset-anchored local completions, protects confirmed fields from overwrite, and chooses between manual query, human confirmation, and conservative local completion using empirical trust calibration.

> Teaser placeholder  
> Add one screenshot of the HOI GUI with:
> 1. one clip loaded,
> 2. Left/Right event rows visible on the timeline,
> 3. the assist panel visible,
> 4. onset grounding boxes rendered on the video frame.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Quickstart](#quickstart)
- [Method-to-Code Map](#method-to-code-map)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Citation](#citation)

## Installation

Status: this release does **not** ship a fully pinned environment file. The commands below install the declared minimum requirements plus the extra packages imported by the optional action-assist path.

The shell commands below assume Linux or macOS from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install Pillow PyYAML transformers decord
```

Notes:
- `requirements.txt` declares `PyQt5`, `opencv-python`, `numpy`, `ultralytics`, `mediapipe`, `torch`, `torchvision`, and `torchaudio`.
- `Action Assist` additionally imports `Pillow`, `PyYAML`, `transformers`, and `decord`.
- CUDA is optional. The code uses `torch.cuda.is_available()` and falls back to CPU when needed.
- Exact tested Python/CUDA/OS versions are not documented in the current release. TODO: add a pinned environment file for the public release.

To launch the application:

```bash
python app.py --oplog
```

## Dataset Preparation

This repository does **not** bundle public data.

The intended evaluation setting uses a curated subset of procedural HOI clips with reference annotations, restricted to:
- single-person clips,
- default `Left_hand` / `Right_hand` entities,
- mostly single-active-hand interactions,
- 15-second clips per condition in the user study.

This release targets the IMPACT dataset setting, but the exact public clip list and split manifest are **not** included.

### Recommended per-clip folder layout

The GUI auto-discovers assets placed next to the video file. The easiest layout is one folder per clip:

```text
data/
└── clip_0001/
    ├── clip_0001.mp4
    ├── nouns.txt
    ├── verbs.txt
    ├── verb_noun_ontology.csv
    ├── data.yaml
    ├── yolo_model.pt
    ├── videomae_cache.npz
    ├── videomae_verbs.yaml
    ├── semantic_adapter.pt
    ├── hands.xml
    └── labels/
        ├── frame_000001.txt
        ├── frame_000002.txt
        └── ...
```

Minimum assets for `Manual` mode:
- one video file,
- `nouns.txt`,
- `verbs.txt`,
- `verb_noun_ontology.csv`.

Additional assets for `Full Assist`:
- either `videomae_cache.npz` or VideoMAE weights,
- optionally `videomae_verbs.yaml` / `.json` / `.txt`,
- optionally `semantic_adapter.pt`,
- optionally `data.yaml` and a YOLO model for object grounding.

### Required text / CSV formats

#### `nouns.txt`

One noun/object category per line:

```text
gear
screwdriver
bolt
housing
```

#### `verbs.txt`

One verb per line:

```text
pick
place
tighten
inspect
```

#### `verb_noun_ontology.csv`

Supported format 1: long form

```csv
verb,noun,allowed
pick,gear,1
pick,screwdriver,1
tighten,bolt,1
inspect,,1
```

Supported format 2: matrix form

```csv
verb,gear,screwdriver,bolt,__NO_NOUN__
pick,1,1,0,0
tighten,0,0,1,0
inspect,0,0,0,1
```

`__NO_NOUN__` is the code path for verbs that can legally omit a noun.

#### `data.yaml`

Minimal YOLO class map:

```yaml
names:
  0: gear
  1: screwdriver
  2: bolt
```

The GUI matches YOLO classes to the noun library by normalized name.

#### YOLO label directory

The GUI can import a directory of per-frame YOLO TXT labels:

```text
labels/
├── frame_000123.txt
├── frame_000124.txt
└── ...
```

Each file should contain standard YOLO rows:

```text
0 0.512500 0.471875 0.125000 0.153125
2 0.665625 0.550000 0.093750 0.118750
```

#### Hand boxes XML

The GUI accepts CVAT XML in both:
- CVAT-for-images style, and
- CVAT video-track style.

Expected hand labels are normalized from variants of `left` / `right`.

#### Existing HOI annotation JSON

The current loader only accepts the native HOI schema with top-level `tracks` and `hoi_events`.

On save, the GUI writes:
- `*.json`
- `*.event_graph.json`
- `*.ops.log.csv`
- `*.validation.json` if validation is enabled

## Pretrained Checkpoints

This release does **not** ship pretrained checkpoints.

### What the GUI expects

#### YOLO model
- File type: `.pt` or `.pth`
- Usage: current-frame and event-keyframe object grounding
- Placement: next to the video for auto-discovery, or load manually from `... -> Detection -> Load YOLO Model...`

#### VideoMAE weights
- File types: `.pt`, `.pth`, `.ckpt`, `.bin`, `.safetensors`
- Usage: frozen video encoder for verb ranking and event-local semantic features
- Alternative: a precomputed `videomae_cache.npz` file

#### VideoMAE cache
- File type: `.npz`
- Expected arrays: `window_starts`, `window_ends`, `window_centers`, `segment_features`
- Optional array: `verb_scores`
- Expected metadata: `meta_json`

#### Semantic adapter
- File type: `semantic_adapter.pt`
- Usage: local onset-semantics completion on top of event-local features
- Search order:
  1. participant-specific runtime workspace,
  2. manually loaded base model,
  3. shared `runtime_artifacts/semantic_adapter.pt`

#### MediaPipe hand landmarker
- File name: `hand_landmarker.task`
- If missing, the application may try to download it automatically on first use.
- Default location:
  - `<video_dir>/runtime_artifacts/mediapipe_models/hand_landmarker.task`
  - or `./runtime_artifacts/mediapipe_models/hand_landmarker.task` if no video is loaded yet

TODO:
- publish checkpoint download links,
- provide file hashes,
- provide one public demo clip with a ready-to-run asset bundle.

## Quickstart

### 1. GUI smoke test

```bash
python app.py --oplog
```

This should open the `IMPACT HOI` window.

### 2. Minimal manual annotation workflow

1. Click `... -> Load Video...` and open a clip.
2. If `nouns.txt`, `verbs.txt`, and `verb_noun_ontology.csv` are in the same folder, the GUI may auto-load them.
3. Otherwise import them manually from `... -> Import`.
4. Start in `Manual` mode.
5. Create an event, set rough start/end, assign verb, assign noun, and save from `... -> Save / Export -> Save HOI Annotations...`.

### 3. Save behavior

By default, the app starts in user-study mode and requires `Participant No` before saving. You have two options:
- fill `Participant No`, or
- press `Ctrl+Alt+U` once to disable user-study mode for local debugging.

## Method-to-Code Map

This section maps the paper's main components to the released code.

- Lock-aware Partial Event Completion (LPEC):
  - `ui/hoi_window.py`
  - `core/hoi_runtime_kernel.py`
  - `core/structured_event_graph.py`

- Hand-guided Onset Prior (HOP):
  - hand-track motion computation in `ui/hoi_window.py`
  - onset band helpers in `core/onset_guidance.py`

- Event-local onset / semantics completion:
  - `core/semantic_adapter.py`
  - `core/hoi_runtime_kernel.py`

- Statistics-guided Cooperative Refinement (SCR):
  - structured semantic reweighting and refinement logic in `core/semantic_adapter.py`

- Trust-Calibrated Supervisory Controller (TSC):
  - `core/hoi_query_controller.py`
  - `core/hoi_empirical_calibration.py`

- Safe execution / rollback / lock preservation:
  - `ui/hoi_window.py`
  - `core/structured_event_graph.py`

## Repository Structure

```text
.
├── app.py                         # PyQt application entry point
├── README.md                      # top-level repository README
├── requirements.txt               # declared minimum Python dependencies
├── LICENSE                        # Apache License 2.0
├── feature_defaults.json          # local default env settings
├── runner_envs.json               # runner profile mapping
├── configs/
│   └── psr_models.json            # auxiliary config, not central to HOI release
├── core/
│   ├── hoi_completion.py          # onset-centric local completion heuristics
│   ├── hoi_empirical_calibration.py
│   ├── hoi_eval_utils.py          # helpers for HOI annotation parsing
│   ├── hoi_ontology.py            # verb-noun ontology loader and constraints
│   ├── hoi_query_controller.py    # supervisory controller and authority policy
│   ├── hoi_runtime_kernel.py      # event-local semantic decode runtime
│   ├── onset_guidance.py          # onset band / local temporal windows
│   ├── semantic_adapter.py        # semantic adapter package and training code
│   ├── structured_event_graph.py  # event-graph export sidecar
│   └── videomae_v2_logic.py       # VideoMAE loading, inference, cache support
├── ui/
│   ├── hoi_window.py              # main HOI annotation GUI
│   ├── hoi_timeline.py            # HOI timeline widgets
│   ├── main_window.py             # top-level host window
│   ├── video_player.py            # video display and frame interaction
│   ├── label_panel.py             # verb/noun library panel
│   └── widgets.py                 # reusable PyQt widgets
├── tools/
│   ├── extract_resnet50_feats.py  # ResNet-50 feature extraction
│   ├── fact_batch_infer.py        # FACT batch inference, needs external FACT repo
│   ├── fact_infer_adapter.py      # FACT single-feature-dir adapter
│   ├── asot_infer_adapter.py      # ASOT MLP inference, needs external ASOT repo
│   ├── asot_full_infer_adapter.py # ASOT full-model inference, needs external ASOT repo
│   ├── asot_ckpt_selector.py      # ASOT checkpoint selection helper
│   ├── boundary_eval.py           # action-seg boundary evaluation utility
│   ├── convert_legacy_action_json.py
│   ├── convert_fine_labels.py
│   ├── repair_action_segments.py
│   └── runners/run_in_env.py      # helper for conda profile execution
└── utils/
    ├── config.py
    ├── constants.py
    ├── feature_env.py
    ├── op_logger.py
    ├── optional_deps.py
    └── shortcut_settings.py
```

## License

This repository includes an Apache License 2.0 file in `LICENSE`.

## Acknowledgments

This release interfaces with the following external projects and model ecosystems:
- Ultralytics YOLO
- MediaPipe Hands / Hand Landmarker
- Hugging Face VideoMAE
- ASOT
- FACT

Please check the original licenses of those dependencies before redistributing weights or derived assets.

## Contact

For questions about the paper or release, contact Di Wen at `di.wen@kit.edu`.

## Citation

```bibtex
@inproceedings{zhang2026impacthoi,
  title={IMPACT-HOI: Supervisory Control for Onset-Anchored Partial HOI Event Construction},
  author={Zhang, Haoshen and Wen, Di and Peng, Kunyu and Schneider, David and Zhong, Zeyun and Jaus, Alexander and Marinov, Zdravko and Wei, Jiale and Liu, Ruiping and Zheng, Junwei and Chen, Yufan and Zhang, Yufeng and Luo, Yuanhao and Qi, Lei and Stiefelhagen, Rainer},
  booktitle={IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year={2026},
  note={Under review}
}
```
