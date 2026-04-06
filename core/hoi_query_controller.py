from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


FIELD_NAMES = (
    "interaction_start",
    "functional_contact_onset",
    "interaction_end",
    "verb",
    "instrument_object_id",
    "target_object_id",
)

FIELD_LABELS = {
    "interaction_start": "start",
    "functional_contact_onset": "onset",
    "interaction_end": "end",
    "verb": "verb",
    "instrument_object_id": "instrument",
    "target_object_id": "target",
}

TEMPORAL_FIELDS = (
    "interaction_start",
    "functional_contact_onset",
    "interaction_end",
)

OBJECT_FIELDS = ("instrument_object_id", "target_object_id")

EVIDENCE_TIME_FIELDS = (
    ("start", "interaction_start"),
    ("onset", "functional_contact_onset"),
    ("end", "interaction_end"),
)

EVIDENCE_ROLE_FIELDS = (
    ("instrument", "instrument_object_id"),
    ("target", "target_object_id"),
)

DEFAULT_COST = {
    "temporal": 0.28,
    "semantic": 0.22,
    "object": 0.52,
    "review": 0.34,
}

DEFAULT_PROPAGATION = {
    "temporal": 0.90,
    "semantic": 0.72,
    "object": 0.66,
    "review": 0.78,
}

DEFAULT_RISK = {
    "temporal": 0.10,
    "semantic": 0.14,
    "object": 0.42,
    "review": 0.16,
}

_UNSET = object()


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def evidence_slot_name(role: str, point: str) -> str:
    role_txt = _safe_text(role).lower() or "object"
    point_txt = _safe_text(point).lower() or "frame"
    return f"{role_txt}_{point_txt}"


def normalize_sparse_evidence_state(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        slot = _safe_text(key)
        if not slot:
            continue
        out[slot] = {
            "status": _safe_text(value.get("status")) or "unknown",
            "source": _safe_text(value.get("source")) or "unknown",
            "frame": _safe_int(value.get("frame")),
            "role": _safe_text(value.get("role")),
            "time_key": _safe_text(value.get("time_key")),
            "time_label": _safe_text(value.get("time_label")),
            "object_id": _safe_int(value.get("object_id")),
            "object_name": _safe_text(value.get("object_name")),
            "note": _safe_text(value.get("note")),
        }
    return out


def _field_is_empty(field_name: str, value: Any) -> bool:
    if field_name == "verb":
        return not _safe_text(value)
    return value is None


def _normalize_state_entry(field_name: str, entry: Any, current_value: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        entry = {}
    status = _safe_text(entry.get("status"))
    source = _safe_text(entry.get("source"))
    if not status:
        status = "empty" if _field_is_empty(field_name, current_value) else "confirmed"
    if not source:
        source = "unknown"
    return {
        "status": status,
        "source": source,
        "updated_at": _safe_text(entry.get("updated_at")) or _now_text(),
        "note": _safe_text(entry.get("note")),
        "value": entry.get("value", current_value),
    }


def _normalize_suggestion_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    if entry.get("value") is None and not _safe_text(entry.get("value")):
        return None
    confidence = entry.get("confidence")
    try:
        confidence = None if confidence is None else float(confidence)
    except Exception:
        confidence = None
    return {
        "value": entry.get("value"),
        "source": _safe_text(entry.get("source")) or "unknown",
        "status": _safe_text(entry.get("status")) or "suggested",
        "created_at": _safe_text(entry.get("created_at")) or _now_text(),
        "reason": _safe_text(entry.get("reason")),
        "confidence": confidence,
        "safe_to_apply": bool(entry.get("safe_to_apply", True)),
    }


def ensure_hand_annotation_state(hand_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(hand_data, dict):
        return {}
    raw_state = hand_data.get("_field_state")
    if not isinstance(raw_state, dict):
        raw_state = {}
    raw_suggestions = hand_data.get("_field_suggestions")
    if not isinstance(raw_suggestions, dict):
        raw_suggestions = {}

    state: Dict[str, Dict[str, Any]] = {}
    for field_name in FIELD_NAMES:
        state[field_name] = _normalize_state_entry(
            field_name, raw_state.get(field_name), hand_data.get(field_name)
        )

    suggestions: Dict[str, Dict[str, Any]] = {}
    for field_name in FIELD_NAMES:
        row = _normalize_suggestion_entry(raw_suggestions.get(field_name))
        if row is not None:
            suggestions[field_name] = row

    hand_data["_field_state"] = state
    hand_data["_field_suggestions"] = suggestions
    return hand_data


def hydrate_existing_field_state(
    hand_data: Dict[str, Any], default_source: str = "loaded_annotation"
) -> Dict[str, Any]:
    hand_data = ensure_hand_annotation_state(hand_data)
    state = hand_data.get("_field_state", {})
    for field_name in FIELD_NAMES:
        value = hand_data.get(field_name)
        row = dict(state.get(field_name, {}))
        if _field_is_empty(field_name, value):
            if row.get("status") not in ("suggested", "confirmed"):
                row["status"] = "empty"
                row["source"] = row.get("source") or "empty"
        else:
            if row.get("status") in ("", None, "empty"):
                row["status"] = "confirmed"
                row["source"] = default_source
                row["updated_at"] = _now_text()
                row["value"] = value
        state[field_name] = _normalize_state_entry(field_name, row, value)
    hand_data["_field_state"] = state
    return hand_data


def get_field_state(hand_data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    ensure_hand_annotation_state(hand_data)
    return dict((hand_data.get("_field_state") or {}).get(field_name, {}))


def get_field_suggestion(hand_data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    ensure_hand_annotation_state(hand_data)
    return dict((hand_data.get("_field_suggestions") or {}).get(field_name, {}))


def set_field_confirmation(
    hand_data: Dict[str, Any],
    field_name: str,
    *,
    source: str,
    value: Any = _UNSET,
    status: str = "confirmed",
    note: str = "",
) -> Dict[str, Any]:
    ensure_hand_annotation_state(hand_data)
    if value is not _UNSET:
        hand_data[field_name] = value
    current_value = hand_data.get(field_name)
    hand_data["_field_state"][field_name] = {
        "status": _safe_text(status) or "confirmed",
        "source": _safe_text(source) or "unknown",
        "updated_at": _now_text(),
        "note": _safe_text(note),
        "value": current_value,
    }
    if status == "confirmed":
        hand_data["_field_suggestions"].pop(field_name, None)
    return dict(hand_data["_field_state"][field_name])


def set_field_suggestion(
    hand_data: Dict[str, Any],
    field_name: str,
    value: Any,
    *,
    source: str,
    confidence: Optional[float] = None,
    reason: str = "",
    safe_to_apply: bool = True,
) -> Dict[str, Any]:
    ensure_hand_annotation_state(hand_data)
    hand_data["_field_suggestions"][field_name] = {
        "value": value,
        "source": _safe_text(source) or "unknown",
        "status": "suggested",
        "created_at": _now_text(),
        "reason": _safe_text(reason),
        "confidence": confidence,
        "safe_to_apply": bool(safe_to_apply),
    }
    return dict(hand_data["_field_suggestions"][field_name])


def clear_field_suggestion(hand_data: Dict[str, Any], field_name: str) -> None:
    ensure_hand_annotation_state(hand_data)
    hand_data["_field_suggestions"].pop(field_name, None)


def clear_field_value(
    hand_data: Dict[str, Any], field_name: str, *, source: str = "manual_clear"
) -> None:
    ensure_hand_annotation_state(hand_data)
    hand_data[field_name] = "" if field_name == "verb" else None
    hand_data["_field_state"][field_name] = {
        "status": "empty",
        "source": _safe_text(source) or "manual_clear",
        "updated_at": _now_text(),
        "note": "",
        "value": hand_data[field_name],
    }
    hand_data["_field_suggestions"].pop(field_name, None)


def apply_field_suggestion(
    hand_data: Dict[str, Any],
    field_name: str,
    *,
    source: Optional[str] = None,
    as_status: str = "suggested",
) -> bool:
    ensure_hand_annotation_state(hand_data)
    state = get_field_state(hand_data, field_name)
    if state.get("status") == "confirmed":
        return False
    suggestion = get_field_suggestion(hand_data, field_name)
    if not suggestion:
        return False
    set_field_confirmation(
        hand_data,
        field_name,
        source=source or suggestion.get("source") or "suggested_apply",
        value=suggestion.get("value"),
        status=as_status,
        note=suggestion.get("reason", ""),
    )
    return True


def _score_query(
    *,
    base_priority: float,
    surface: str,
    selected_bonus: float = 0.0,
    propagation_gain: Optional[float] = None,
    human_cost: Optional[float] = None,
    overwrite_risk: Optional[float] = None,
) -> Dict[str, float]:
    propagation = (
        DEFAULT_PROPAGATION.get(surface, 0.5)
        if propagation_gain is None
        else max(0.0, min(1.0, float(propagation_gain)))
    )
    cost = (
        DEFAULT_COST.get(surface, 0.4)
        if human_cost is None
        else max(0.0, min(1.0, float(human_cost)))
    )
    risk = (
        DEFAULT_RISK.get(surface, 0.2)
        if overwrite_risk is None
        else max(0.0, min(1.0, float(overwrite_risk)))
    )
    voi = float(base_priority + 0.24 * propagation - 0.14 * cost - 0.22 * risk + selected_bonus)
    return {
        "base_priority": float(base_priority),
        "propagation_gain": float(propagation),
        "human_cost_est": float(cost),
        "overwrite_risk": float(risk),
        "voi_score": float(voi),
    }


def build_query_candidates(
    hand_rows: Sequence[Dict[str, Any]],
    *,
    selected_event_id: Optional[int] = None,
    selected_hand: Optional[str] = None,
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []

    for row in list(hand_rows or []):
        if not isinstance(row, dict):
            continue
        event_id = row.get("event_id")
        hand = _safe_text(row.get("hand"))
        selected_bonus = 0.0
        if event_id == selected_event_id:
            selected_bonus += 0.04
        if hand and hand == _safe_text(selected_hand):
            selected_bonus += 0.03

        field_state = dict(row.get("field_state") or {})
        field_suggestions = dict(row.get("field_suggestions") or {})
        bbox_errors = list(row.get("bbox_errors") or [])
        sparse_evidence_state = normalize_sparse_evidence_state(
            row.get("sparse_evidence_state")
        )
        consistency_flags = list(row.get("consistency_flags") or [])
        videomae_candidates = list(row.get("videomae_candidates") or [])

        start = row.get("interaction_start")
        onset = row.get("functional_contact_onset")
        end = row.get("interaction_end")
        verb = _safe_text(row.get("verb"))
        inst = row.get("instrument_object_id")
        target = row.get("target_object_id")

        def _append(query: Dict[str, Any]) -> None:
            query.setdefault("event_id", event_id)
            query.setdefault("hand", hand)
            query.setdefault("selected_bonus", selected_bonus)
            query.setdefault(
                "sparse_evidence_summary",
                dict(row.get("sparse_evidence_summary") or {}),
            )
            query.setdefault(
                "query_id",
                "|".join(
                    [
                        str(event_id),
                        hand or "",
                        str(query.get("query_type") or ""),
                        str(query.get("field_name") or ""),
                        str(query.get("target_frame") or ""),
                        str(query.get("target_slot") or ""),
                    ]
                ),
            )
            queries.append(query)

        onset_state = dict(field_state.get("functional_contact_onset") or {})
        onset_suggestion = dict(field_suggestions.get("functional_contact_onset") or {})
        if onset_state.get("status") == "suggested" and onset is not None:
            score = _score_query(
                base_priority=0.93,
                surface="temporal",
                selected_bonus=selected_bonus,
                overwrite_risk=0.06,
            )
            _append(
                {
                    "query_type": "confirm_suggested_onset",
                    "surface": "event",
                    "field_name": "functional_contact_onset",
                    "summary": f"Confirm onset at frame {int(onset)}.",
                    "reason": onset_suggestion.get("reason")
                    or "Onset was suggested and should be confirmed before downstream review.",
                    "target_frame": int(onset),
                    "safe_apply": True,
                    "apply_mode": "confirm_current",
                    "suggested_value": onset,
                    "suggested_source": onset_state.get("source") or onset_suggestion.get("source"),
                    **score,
                }
            )
        elif onset is None and start is not None and end is not None:
            midpoint = int(round((int(start) + int(end)) / 2.0))
            score = _score_query(
                base_priority=0.95,
                surface="temporal",
                selected_bonus=selected_bonus,
                propagation_gain=0.96,
                overwrite_risk=0.05,
            )
            _append(
                {
                    "query_type": "fill_missing_onset",
                    "surface": "event",
                    "field_name": "functional_contact_onset",
                    "summary": f"Add onset between start {int(start)} and end {int(end)}.",
                    "reason": "Onset anchors the sparse-evidence protocol and contracts the feasible event graph.",
                    "target_frame": midpoint,
                    "safe_apply": True,
                    "apply_mode": "apply_suggestion",
                    "suggested_value": midpoint,
                    "suggested_source": "midpoint_inference",
                    **score,
                }
            )

        verb_state = dict(field_state.get("verb") or {})
        verb_suggestion = dict(field_suggestions.get("verb") or {})
        if verb_state.get("status") == "suggested" and verb:
            score = _score_query(
                base_priority=0.89,
                surface="semantic",
                selected_bonus=selected_bonus,
                overwrite_risk=0.07,
            )
            _append(
                {
                    "query_type": "confirm_suggested_verb",
                    "surface": "event",
                    "field_name": "verb",
                    "summary": f"Confirm suggested verb '{verb}'.",
                    "reason": verb_suggestion.get("reason")
                    or "The current action label was suggested by the model and still needs confirmation.",
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": True,
                    "apply_mode": "confirm_current",
                    "suggested_value": verb,
                    "suggested_source": verb_state.get("source") or verb_suggestion.get("source"),
                    **score,
                }
            )
        elif not verb:
            top_candidate = videomae_candidates[0] if videomae_candidates else {}
            suggested_label = _safe_text(top_candidate.get("label"))
            confidence = top_candidate.get("score")
            score = _score_query(
                base_priority=0.84 if suggested_label else 0.70,
                surface="semantic",
                selected_bonus=selected_bonus,
                propagation_gain=0.74 if suggested_label else 0.48,
                overwrite_risk=0.11 if suggested_label else 0.18,
            )
            _append(
                {
                    "query_type": "fill_missing_verb",
                    "surface": "event",
                    "field_name": "verb",
                    "summary": (
                        f"Confirm verb '{suggested_label}'."
                        if suggested_label
                        else "Choose a verb for this HOI event."
                    ),
                    "reason": (
                        "VideoMAE already produced a candidate and confirming it will reduce semantic ambiguity."
                        if suggested_label
                        else "This event has temporal support but no action label yet."
                    ),
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": bool(suggested_label),
                    "apply_mode": "apply_suggestion" if suggested_label else "",
                    "suggested_value": suggested_label or None,
                    "suggested_source": "videomae_top1" if suggested_label else "",
                    "suggested_confidence": confidence,
                    **score,
                }
            )

        if inst is None:
            score = _score_query(
                base_priority=0.76,
                surface="object",
                selected_bonus=selected_bonus,
                propagation_gain=0.63,
                overwrite_risk=0.20,
            )
            _append(
                {
                    "query_type": "assign_instrument",
                    "surface": "objects",
                    "field_name": "instrument_object_id",
                    "summary": "Assign an instrument object.",
                    "reason": "Instrument identity is still missing for this event.",
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": False,
                    **score,
                }
            )
        if target is None:
            score = _score_query(
                base_priority=0.74,
                surface="object",
                selected_bonus=selected_bonus,
                propagation_gain=0.60,
                overwrite_risk=0.20,
            )
            _append(
                {
                    "query_type": "assign_target",
                    "surface": "objects",
                    "field_name": "target_object_id",
                    "summary": "Assign a target object.",
                    "reason": "Target identity is still missing for this event.",
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": False,
                    **score,
                }
            )

        if bbox_errors:
            time_rank = {"Onset": 0, "Start": 1, "End": 2}
            role_rank = {"Instrument": 0, "Target": 1}
            ordered_bbox_errors = sorted(
                [dict(err) for err in bbox_errors],
                key=lambda err: (
                    time_rank.get(_safe_text(err.get("time_label")).title(), 3),
                    role_rank.get(_safe_text(err.get("role")).title(), 2),
                    _safe_int(err.get("frame")) if _safe_int(err.get("frame")) is not None else 10**9,
                ),
            )
            for err in ordered_bbox_errors:
                time_label = _safe_text(err.get("time_label")).title() or "Key"
                role_label = _safe_text(err.get("role")).title() or "Object"
                frame = _safe_int(err.get("frame"))
                time_key = _safe_text(err.get("time_key")).lower()
                target_slot = evidence_slot_name(role_label, time_label)
                base_priority = {
                    "onset": 0.92,
                    "start": 0.84,
                    "end": 0.80,
                }.get(time_key, 0.78)
                if role_label.lower() == "instrument":
                    base_priority += 0.02
                score = _score_query(
                    base_priority=base_priority,
                    surface="object",
                    selected_bonus=selected_bonus,
                    propagation_gain=0.94 if time_key == "onset" else 0.82,
                    human_cost=0.56,
                    overwrite_risk=0.26,
                )
                object_name = _safe_text(err.get("object_name"))
                object_text = f" '{object_name}'" if object_name else ""
                _append(
                    {
                        "query_type": "verify_sparse_evidence",
                        "surface": "objects",
                        "field_name": "bbox",
                        "summary": (
                            f"Acquire {role_label.lower()}{object_text} evidence at {time_label.lower()} "
                            f"(frame {frame})."
                            if frame is not None
                            else f"Acquire {role_label.lower()}{object_text} evidence at {time_label.lower()}."
                        ),
                        "reason": (
                            "Sparse evidence is organized around start / onset / end; resolving this slot improves event grounding."
                        ),
                        "target_frame": frame,
                        "target_role": role_label.lower(),
                        "target_time_key": time_key,
                        "target_time_label": time_label,
                        "target_slot": target_slot,
                        "safe_apply": False,
                        **score,
                    }
                )

        if consistency_flags:
            first_flag = dict(consistency_flags[0])
            flag_text = _safe_text(first_flag.get("flag")).replace("_", " ") or "consistency issue"
            score = _score_query(
                base_priority=0.83,
                surface="review",
                selected_bonus=selected_bonus,
                propagation_gain=0.79,
                overwrite_risk=0.12,
            )
            _append(
                {
                    "query_type": "resolve_consistency_flag",
                    "surface": "review",
                    "field_name": first_flag.get("flag") or "consistency",
                    "summary": f"Resolve consistency issue: {flag_text}.",
                    "reason": "The event graph contains an inconsistent temporal or semantic state.",
                    "target_frame": first_flag.get("contact_onset_frame")
                    or first_flag.get("start_frame")
                    or first_flag.get("end_frame"),
                    "safe_apply": False,
                    **score,
                }
            )

        if sparse_evidence_state:
            for slot_name, slot_state in sparse_evidence_state.items():
                if _safe_text(slot_state.get("status")) != "suggested":
                    continue
                frame = _safe_int(slot_state.get("frame"))
                role_label = _safe_text(slot_state.get("role")).title() or "Object"
                time_label = _safe_text(slot_state.get("time_label")).title() or "Key"
                source = _safe_text(slot_state.get("source")) or "suggested_evidence"
                score = _score_query(
                    base_priority=0.86 if time_label.lower() == "onset" else 0.80,
                    surface="object",
                    selected_bonus=selected_bonus,
                    propagation_gain=0.90,
                    human_cost=0.38,
                    overwrite_risk=0.10,
                )
                _append(
                    {
                        "query_type": "confirm_sparse_evidence",
                        "surface": "objects",
                        "field_name": "bbox",
                        "summary": (
                            f"Confirm {role_label.lower()} evidence at {time_label.lower()} "
                            f"(frame {frame})."
                            if frame is not None
                            else f"Confirm {role_label.lower()} evidence at {time_label.lower()}."
                        ),
                        "reason": _safe_text(slot_state.get("note"))
                        or "Suggested sparse evidence should be confirmed before finalization.",
                        "target_frame": frame,
                        "target_role": role_label.lower(),
                        "target_time_key": _safe_text(slot_state.get("time_key")).lower(),
                        "target_time_label": time_label,
                        "target_slot": slot_name,
                        "safe_apply": False,
                        "suggested_source": source,
                        **score,
                    }
                )

    queries.sort(
        key=lambda row: (
            float(row.get("voi_score", 0.0) or 0.0),
            float(row.get("base_priority", 0.0) or 0.0),
            -float(row.get("human_cost_est", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return queries
