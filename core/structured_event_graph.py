from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence


SCHEMA_VERSION = 2


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_annotation_state(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        out[str(key)] = dict(value)
    return out


def _normalize_sparse_evidence_state(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        out[str(key)] = dict(value)
    return out


def event_graph_sidecar_path(annotation_path: str) -> str:
    text = str(annotation_path or "").strip()
    if not text:
        return ""
    base, _ext = os.path.splitext(text)
    if not base:
        base = text
    return f"{base}.event_graph.json"


def _normalize_actor_id(actor_id: str) -> str:
    text = _safe_text(actor_id)
    if not text:
        return "Unknown_actor"
    norm = text.replace(" ", "_")
    if norm.lower() == "left":
        return "Left_hand"
    if norm.lower() == "right":
        return "Right_hand"
    return norm


def _iter_actor_defs(
    events: Sequence[Dict[str, Any]], actors_config: Optional[Sequence[Dict[str, Any]]] = None
) -> List[Dict[str, str]]:
    ordered: List[Dict[str, str]] = []
    seen = set()

    def _append(actor_id: str, label: str = "", short: str = "") -> None:
        aid = _normalize_actor_id(actor_id)
        if not aid or aid in seen:
            return
        seen.add(aid)
        ordered.append(
            {
                "id": aid,
                "label": _safe_text(label) or aid.replace("_", " "),
                "short": _safe_text(short) or aid[:1].upper(),
            }
        )

    for actor in list(actors_config or []):
        if not isinstance(actor, dict):
            continue
        _append(actor.get("id", ""), actor.get("label", ""), actor.get("short", ""))

    for event in list(events or []):
        if not isinstance(event, dict):
            continue
        hoi_data = event.get("hoi_data", {}) or {}
        if not isinstance(hoi_data, dict):
            continue
        for actor_id in hoi_data.keys():
            _append(actor_id)

    if not ordered:
        _append("Left_hand", "Left Hand", "L")
        _append("Right_hand", "Right Hand", "R")
    return ordered


def _has_event_content(entry: Dict[str, Any]) -> bool:
    return any(
        entry.get(key) is not None
        for key in (
            "start_frame",
            "contact_onset_frame",
            "end_frame",
            "instrument_object_id",
            "target_object_id",
        )
    ) or bool(_safe_text(entry.get("verb", "")))


def _defined_frames(entry: Dict[str, Any]) -> List[int]:
    frames: List[int] = []
    for key in ("start_frame", "contact_onset_frame", "end_frame"):
        value = _safe_int(entry.get(key))
        if value is not None:
            frames.append(int(value))
    return frames

def _event_consistency_flags(entry: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    start = _safe_int(entry.get("start_frame"))
    onset = _safe_int(entry.get("contact_onset_frame"))
    end = _safe_int(entry.get("end_frame"))
    verb = _safe_text(entry.get("verb", ""))
    has_objects = entry.get("instrument_object_id") is not None or entry.get("target_object_id") is not None

    if start is not None and end is not None and start > end:
        flags.append("start_after_end")
    if onset is not None and start is not None and onset < start:
        flags.append("onset_before_start")
    if onset is not None and end is not None and onset > end:
        flags.append("onset_after_end")
    if (start is not None or onset is not None or end is not None or has_objects) and not verb:
        flags.append("missing_verb")
    if verb and start is None and onset is None and end is None:
        flags.append("missing_temporal_support")
    return flags


def build_hoi_event_graph(
    events: List[Dict[str, Any]],
    *,
    video_path: str = "",
    annotation_path: str = "",
    actors_config: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    actor_defs = _iter_actor_defs(events, actors_config)
    graph_events: List[Dict[str, Any]] = []
    onset_anchors: List[Dict[str, Any]] = []
    locked_regions: List[Dict[str, Any]] = []
    consistency_flags: List[Dict[str, Any]] = []

    for idx, event in enumerate(list(events or [])):
        if not isinstance(event, dict):
            continue
        event_id = _safe_int(event.get("event_id"))
        if event_id is None:
            event_id = idx
        hoi_data = event.get("hoi_data", {}) or {}
        if not isinstance(hoi_data, dict):
            hoi_data = {}

        for actor in actor_defs:
            actor_id = actor["id"]
            hand = hoi_data.get(actor_id, {}) or {}
            if not isinstance(hand, dict):
                continue
            start_frame = _safe_int(hand.get("interaction_start"))
            onset_frame = _safe_int(hand.get("functional_contact_onset"))
            end_frame = _safe_int(hand.get("interaction_end"))
            entry = {
                "event_id": int(event_id),
                "hand": actor_id,
                "actor_id": actor_id,
                "actor_label": actor.get("label", actor_id),
                "actor_short": actor.get("short", actor_id[:1].upper()),
                "start_frame": start_frame,
                "contact_onset_frame": onset_frame,
                "end_frame": end_frame,
                "verb": _safe_text(hand.get("verb", "")),
                "instrument_object_id": _safe_int(hand.get("instrument_object_id")),
                "target_object_id": _safe_int(hand.get("target_object_id")),
                "anomaly_label": _safe_text(hand.get("anomaly_label", "")) or "Normal",
                "source_task": "hoi",
                "confirmed_kind": "confirmed",
                "locked": True,
                "field_state": _normalize_annotation_state(hand.get("_field_state")),
                "field_suggestions": _normalize_annotation_state(
                    hand.get("_field_suggestions")
                ),
                "sparse_evidence_state": _normalize_sparse_evidence_state(
                    hand.get("_sparse_evidence_state")
                ),
            }
            if not _has_event_content(entry):
                continue

            sparse_evidence = list((entry.get("sparse_evidence_state") or {}).values())
            sparse_expected = 0
            sparse_confirmed = 0
            sparse_missing = 0
            for item in sparse_evidence:
                status = _safe_text((item or {}).get("status")).lower()
                if status == "blocked":
                    continue
                sparse_expected += 1
                if status == "confirmed":
                    sparse_confirmed += 1
                elif status == "missing":
                    sparse_missing += 1
            entry["sparse_evidence_expected_count"] = int(sparse_expected)
            entry["sparse_evidence_confirmed_count"] = int(sparse_confirmed)
            entry["sparse_evidence_missing_count"] = int(sparse_missing)

            flags = _event_consistency_flags(entry)
            entry["consistency_flags"] = list(flags)
            entry["has_consistency_issue"] = bool(flags)
            graph_events.append(entry)

            if onset_frame is not None:
                onset_anchors.append(
                    {
                        "event_id": int(event_id),
                        "hand": actor_id,
                        "actor_id": actor_id,
                        "actor_label": actor.get("label", actor_id),
                        "frame": int(onset_frame),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "verb": entry["verb"],
                        "instrument_object_id": entry["instrument_object_id"],
                        "target_object_id": entry["target_object_id"],
                        "locked": True,
                        "anchor_type": "contact_onset",
                        "source_task": "hoi",
                    }
                )

            frames = _defined_frames(entry)
            if frames:
                locked_regions.append(
                    {
                        "event_id": int(event_id),
                        "hand": actor_id,
                        "actor_id": actor_id,
                        "start_frame": int(min(frames)),
                        "end_frame": int(max(frames)),
                        "anchor_frames": sorted(frames),
                        "verb": entry["verb"],
                        "locked": True,
                        "region_type": "confirmed_temporal_support",
                        "source_task": "hoi",
                    }
                )

            for flag in flags:
                consistency_flags.append(
                    {
                        "event_id": int(event_id),
                        "hand": actor_id,
                        "actor_id": actor_id,
                        "flag": flag,
                        "start_frame": start_frame,
                        "contact_onset_frame": onset_frame,
                        "end_frame": end_frame,
                        "source_task": "hoi",
                    }
                )

    return {
        "schema_version": int(SCHEMA_VERSION),
        "graph_type": "hoi_event_graph",
        "source_task": "hoi",
        "video_path": _safe_text(video_path),
        "annotation_path": _safe_text(annotation_path),
        "actors_config": actor_defs,
        "events": graph_events,
        "onset_anchors": onset_anchors,
        "locked_regions": locked_regions,
        "consistency_flags": consistency_flags,
        "stats": {
            "event_count": int(len(graph_events)),
            "actor_count": int(len(actor_defs)),
            "onset_anchor_count": int(len(onset_anchors)),
            "locked_region_count": int(len(locked_regions)),
            "consistency_issue_count": int(len(consistency_flags)),
            "sparse_evidence_expected_count": int(
                sum(int(item.get("sparse_evidence_expected_count", 0) or 0) for item in graph_events)
            ),
            "sparse_evidence_confirmed_count": int(
                sum(int(item.get("sparse_evidence_confirmed_count", 0) or 0) for item in graph_events)
            ),
            "sparse_evidence_missing_count": int(
                sum(int(item.get("sparse_evidence_missing_count", 0) or 0) for item in graph_events)
            ),
        },
    }


def save_event_graph_sidecar(annotation_path: str, graph: Dict[str, Any]) -> str:
    path = event_graph_sidecar_path(annotation_path)
    if not path:
        raise ValueError("annotation_path is required")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return path


def load_event_graph_sidecar(annotation_path: str) -> Dict[str, Any]:
    path = event_graph_sidecar_path(annotation_path)
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def extract_onset_anchors(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(graph, dict):
        return []
    anchors = graph.get("onset_anchors", [])
    if not isinstance(anchors, list):
        anchors = []
    out: List[Dict[str, Any]] = []
    for item in anchors:
        if not isinstance(item, dict):
            continue
        frame = _safe_int(item.get("frame"))
        if frame is None:
            continue
        out.append(
            {
                "event_id": _safe_int(item.get("event_id")),
                "hand": _normalize_actor_id(item.get("hand", "")),
                "actor_id": _normalize_actor_id(item.get("actor_id", item.get("hand", ""))),
                "frame": int(frame),
                "start_frame": _safe_int(item.get("start_frame")),
                "end_frame": _safe_int(item.get("end_frame")),
                "verb": _safe_text(item.get("verb", "")),
                "instrument_object_id": _safe_int(item.get("instrument_object_id")),
                "target_object_id": _safe_int(item.get("target_object_id")),
                "locked": bool(item.get("locked", True)),
                "anchor_type": _safe_text(item.get("anchor_type", "contact_onset")) or "contact_onset",
                "source_task": _safe_text(item.get("source_task", "hoi")) or "hoi",
            }
        )
    out.sort(key=lambda row: (int(row.get("frame", 0)), str(row.get("hand", ""))))
    return out


def extract_locked_regions(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(graph, dict):
        return []
    rows = graph.get("locked_regions", [])
    if not isinstance(rows, list):
        rows = []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        start = _safe_int(item.get("start_frame"))
        end = _safe_int(item.get("end_frame"))
        if start is None or end is None:
            continue
        anchors = []
        for raw in list(item.get("anchor_frames", []) or []):
            frame = _safe_int(raw)
            if frame is not None:
                anchors.append(int(frame))
        out.append(
            {
                "event_id": _safe_int(item.get("event_id")),
                "hand": _normalize_actor_id(item.get("hand", "")),
                "actor_id": _normalize_actor_id(item.get("actor_id", item.get("hand", ""))),
                "start_frame": int(min(start, end)),
                "end_frame": int(max(start, end)),
                "anchor_frames": sorted(anchors),
                "verb": _safe_text(item.get("verb", "")),
                "locked": bool(item.get("locked", True)),
                "region_type": _safe_text(item.get("region_type", "confirmed_temporal_support")) or "confirmed_temporal_support",
                "source_task": _safe_text(item.get("source_task", "hoi")) or "hoi",
            }
        )
    out.sort(key=lambda row: (int(row.get("start_frame", 0)), int(row.get("end_frame", 0))))
    return out


def extract_consistency_flags(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(graph, dict):
        return []
    rows = graph.get("consistency_flags", [])
    if not isinstance(rows, list):
        rows = []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "event_id": _safe_int(item.get("event_id")),
                "hand": _normalize_actor_id(item.get("hand", "")),
                "actor_id": _normalize_actor_id(item.get("actor_id", item.get("hand", ""))),
                "flag": _safe_text(item.get("flag", "")),
                "start_frame": _safe_int(item.get("start_frame")),
                "contact_onset_frame": _safe_int(item.get("contact_onset_frame")),
                "end_frame": _safe_int(item.get("end_frame")),
                "source_task": _safe_text(item.get("source_task", "hoi")) or "hoi",
            }
        )
    return out
