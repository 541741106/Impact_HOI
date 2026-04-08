# IMPACT HOI

IMPACT HOI is a PyQt5 desktop annotation tool for HandOI / HOI labeling. The current shipped entrypoint is the standalone HOI application in [app.py](app.py), with object import/detection utilities, HOI timeline editing, VideoMAE-assisted verb ranking, validation logging, and structured event-graph sidecars.

## Current Scope

- `python app.py` launches the **HandOI / HOI Detection** window only.
- The repository still contains Action Segmentation / PSR / ASR / ASD code and helper scripts, but those modules are **not** mounted by the current main window.
- The README below documents the application state that actually matches the current entrypoint and UI.

## Installation

Install the core GUI and HOI dependencies:

```bash
pip install -r requirements.txt
```

Optional VideoMAE extras are loaded lazily. Install them only if you want action-ranking support inside the HOI UI:

```bash
pip install Pillow PyYAML transformers decord
```

## Launch

Normal launch:

```bash
python app.py
```

Launch with operation logging enabled:

```bash
python app.py --oplog
```

`--oplog` enables `*.ops.log.csv` output when you save annotations.

## What The Current App Does

The HOI window supports these main workflows:

- Load a video and crop the active working frame range.
- Use a cleaner two-row top bar: session controls above, transport/edit controls below.
- Open project, import/export, detection, model, and settings actions from the compact `...` menu.
- Work inside three inspector tabs: `Event`, `Objects`, and `Review`, each with a task-specific status card and vertical scrolling when the inspector content is taller than the viewport.
- Adjust `UI Scale` from `... -> Settings...` or `Ctrl+,` to tune font and control density for different screens.
- Create and edit HOI events on the timeline.
- Assign verbs, instrument objects, target objects, and anomaly labels per actor.
- Rename anomaly groups and anomaly labels inline by double-clicking them.
- Rank the verb library with VideoMAE suggestions for the selected event.
- Track per-field `confirmed / suggested / source` state for HOI event supervision.
- Track onset-centered sparse evidence at `start / onset / end` for instrument/target grounding, with per-slot provenance derived from the current boxes.
- Surface a `Next Best Query` card in `Review` that points to the most valuable next temporal / semantic / object / review action and exposes `VOI / propagation / cost / risk`.
- Save HOI annotations plus a structured `event_graph` sidecar.
- Run validation sessions and emit validation summaries / logs with query-level instrumentation.

## Quickstart

1. Run `python app.py`.
2. Open the top-right `...` menu and choose **Load Video...**.
3. Import supporting data from the same `...` menu as needed:
   - **Import Instrument List...**
   - **Import Target List...**
   - **Load Class Map (data.yaml)...**
   - **Import YOLO Boxes...** or **Detect Current Frame**
   - **Import Verb List...**
4. Create HOI events by dragging on the actor rows in the timeline.
5. Use the inspector tabs on the left:
   - `Event`: selected-event status, Start/Onset/End jump buttons, actor selection, and action labeling
   - `Objects`: instrument/target linking, current-frame object list, local detection shortcuts, and box-edit controls
   - `Review`: anomaly labels, validation state, incomplete-event summary / navigation, and the controller's `Next Best Query`
6. Use the **Action** panel to choose or type the verb.
7. Open `... -> Settings...` if you want to change `UI Scale`, operation logging, or validation-summary logging.
8. Save from the top-right `...` menu.

## Action Panel Behavior

The current `Action` panel is no longer based on a manual `Suggest -> choose -> apply` loop.

- Selecting an event automatically tries to rank the `Verb Library` with cached VideoMAE suggestions.
- Retiming an event (`start / onset / end`) invalidates the stale ranking and schedules a delayed refresh.
- The `Refresh` button forces a fresh VideoMAE re-ranking for the selected event.
- `Detect All` is intentionally treated as a low-frequency batch action and is kept inside `Objects -> ...` instead of the primary toolbar row.
- VideoMAE top-1 results can now exist as `suggested` verb state instead of silently overwriting already confirmed labels.
- The action `...` menu contains low-frequency actions such as `Manage Verb Library` and `Auto-Apply Top-1 to All Events`.
- The Event tab also exposes a compact status card so missing fields are visible without parsing a long status string.
- `Manage Verb Library` reveals the add/remove/color controls only when you need them.
- **... -> Action Assist -> Review Selected Action Label...** opens the explicit chooser dialog for the selected event.

## Query Controller And Sparse Evidence

- The HOI window now carries per-field supervision state for `start / onset / end / verb / instrument / target`.
- `Review -> Next Best Query` is a lightweight query controller: it ranks the next most valuable supervision step across `Event`, `Objects`, and `Review`.
- The controller is onset-centered: missing or suggested `onset` supervision is treated as high-leverage because it contracts the feasible event graph and reduces downstream review.
- Sparse object evidence is organized as six key slots: `instrument@start`, `instrument@onset`, `instrument@end`, `target@start`, `target@onset`, and `target@end`.
- Object-evidence queries are specific rather than generic: the Review panel now points to the exact missing keyframe slot that still needs grounding.
- The `Next Best Query` card now exposes the controller's explanation terms:
  - `VOI`: overall priority score
  - `Prop`: expected downstream supervision saved by resolving the query
  - `Cost`: estimated human effort
  - `Risk`: overwrite / rework risk
- Safe local completion is deliberately conservative:
  - it only fills missing or unconfirmed fields inside the current event
  - it never overwrites a field already marked `confirmed`
  - it is currently used for low-risk completions such as midpoint onset suggestions or model-suggested verbs
- Sparse object evidence still centers on `start / onset / end`, and missing boxes on those keyframes are surfaced as object-evidence queries.
- Box provenance is preserved where possible:
  - imported YOLO boxes are tagged as `yolo_import`
  - detector proposals are tagged as `yolo_detect`
  - MediaPipe hand boxes are tagged as `mediapipe_hands`
  - CVAT XML hand boxes are tagged as `hands_xml`
  - manual edits are tagged as `manual_box_add` / `manual_box_edit`

If the optional VideoMAE packages are not installed, the HOI window still opens; only action-ranking features are unavailable.

## Settings

The current app exposes a compact settings entry at `... -> Settings...` and `Ctrl+,`.

The current Settings dialog includes:

- `UI Scale`: scales fonts, button density, and toolbar icon sizes together.
- `Reset Layout`: restores the default inspector / timeline splitter proportions.
- `Write operations CSV`: toggles `*.ops.log.csv` output.
- `Write validation summary`: toggles validation-summary export when validation mode is used.

`UI Scale` is persisted across launches in the user settings directory, not next to individual annotation files.

## Saved Files

When you save HOI annotations, the application may write these files next to the chosen output path:

- `<name>.json`: main HOI annotation file.
- `<name>.event_graph.json`: structured event-graph sidecar generated from HOI events.
- `<name>.ops.log.csv`: operation log when logging is enabled. Rows now include `session_id`, `elapsed_ms`, `assist_mode`, and query metadata such as `query_id`, `query_type`, `VOI`, and `query_latency_ms` when applicable.
- `<name>.validation.json`: validation summary when validation mode was active.
- `<name>.validation.ops.log.csv`: validation operation log when validation mode and oplog are both enabled.

## User Settings

Application-level settings are stored in the per-user settings directory used by `utils/shortcut_settings.py` (by default `~/.cvhci_video_annotation_suite`). This currently includes:

- `shortcuts.json` / `shortcuts.backup.json`: shortcut bindings
- `logging_policy.json` / `logging_policy.backup.json`: logging toggles
- `ui_preferences.json` / `ui_preferences.backup.json`: UI scale preference

## Repository Layout

The files most relevant to the current shipped UI are:

- [app.py](app.py): application entrypoint.
- [ui/main_window.py](ui/main_window.py): current standalone HOI host.
- [ui/hoi_window.py](ui/hoi_window.py): main HOI application window.
- [ui/hoi_timeline.py](ui/hoi_timeline.py): HOI timeline editor.
- [ui/label_panel.py](ui/label_panel.py): verb library and label helpers.
- [core/structured_event_graph.py](core/structured_event_graph.py): HOI event-graph sidecar generation / loading.
- [core/videomae_v2_logic.py](core/videomae_v2_logic.py): optional VideoMAE action-ranking helper.
- [utils/op_logger.py](utils/op_logger.py): operation logging.

Auxiliary scripts and legacy modules still present in the repo include:

- [tools](tools): ASOT / FACT / conversion / evaluation helpers.
- [ui/action_window.py](ui/action_window.py): Action Segmentation window code, not launched by `app.py`.
- [ui/psr_window.py](ui/psr_window.py): legacy PSR UI, not launched by `app.py`.
- [docs/psr_asr_asd_code_map.md](docs/psr_asr_asd_code_map.md): legacy PSR/ASR/ASD note.

## Notes

- This README intentionally reflects the **current runnable app**, not every historical module in the repository.
- If you want to expose Action Segmentation or PSR again from the GUI, that needs to be re-wired in [ui/main_window.py](ui/main_window.py).
- The HOI UI currently treats VideoMAE as an optional assistant, not as a required dependency for startup.
- `Assist Mode` now changes both behavior and visible affordances: `Manual`, `Assist`, and `Full Assist` are intended for side-by-side workflow evaluation.
