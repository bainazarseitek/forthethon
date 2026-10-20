import importlib
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.frames import decode_at, to_jpeg_base64

log = logging.getLogger("pipeline")


@dataclass(frozen=True)
class Rules:
    documents: dict
    types: dict

    @classmethod
    def load(cls, path: str | Path) -> "Rules":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["documents"], {t["type_id"]: t for t in data["types"]})

    def resolve(self, type_id: str | None) -> dict:
        return self.types.get(type_id or "") or self.types["other"]

    def _grouped(self, refs: list[dict], with_clause: bool = True) -> list[tuple[str, list[str]]]:
        groups: dict[str, list[str]] = {}
        for ref in refs:
            clauses = groups.setdefault(ref["doc"], [])
            if with_clause and ref.get("clause") and ref["doc"] != "BANK":
                clauses.append(ref["clause"])
        return list(groups.items())

    def citation(self, refs: list[dict], with_clause: bool = True) -> str:
        parts = []
        for doc, clauses in self._grouped(refs, with_clause):
            if doc == "BANK":
                parts.append("внутренние правила банка")
            else:
                parts.append(", ".join([self.documents[doc]["title"], *clauses]))
        return "; ".join(parts)

    def primary_ref(self, refs: list[dict]) -> dict:
        grouped = self._grouped(refs)
        doc, clauses = next(((d, c) for d, c in grouped if d != "BANK"), grouped[0])
        if doc == "BANK":
            return {"document": "внутренние правила банка", "clause": None, "url": None}
        meta = self.documents[doc]
        return {"document": meta["title"], "clause": ", ".join(clauses) or None, "url": meta.get("url")}

    def describe(self, event: dict) -> tuple[dict, str, dict]:
        known = event.get("type_id") in self.types
        entry = self.resolve(event.get("type_id"))
        observation = (event.get("observation") or "").strip().rstrip(".")
        observation = observation[:1].upper() + observation[1:]
        if entry["type_id"] == "other":
            text = f"{observation}. {_sentence(entry['rule'])} ({self.citation(entry['refs'], with_clause=False)})."
            closest = self.types.get(event.get("closest_type_id") or "") if known else None
            if closest and closest["type_id"] != "other":
                text += f" По аналогии: {_sentence(closest['rule'])} ({self.citation(closest['refs'])})."
            return entry, text, self.primary_ref(entry["refs"])
        text = f"{observation}. {_sentence(entry['rule'])} ({self.citation(entry['refs'])})."
        return entry, text, self.primary_ref(entry["refs"])


def _sentence(rule: str) -> str:
    return rule.strip().rstrip(".")


def load_detector(name: str) -> Callable[[str], dict]:
    module = importlib.import_module("app.detector_stub" if name == "stub" else name)
    if hasattr(module, "load"):
        module.load()
    return module.analyze


def threshold_for(event: dict, rules: Rules, settings: Settings) -> float:
    is_other = rules.resolve(event.get("type_id"))["type_id"] == "other"
    return settings.threshold_other if is_other else settings.threshold


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clean_event(event: dict) -> dict | None:
    confidence = finite(event.get("confidence"))
    if confidence is None:
        return None
    frames = [dict(f, t_sec=t) for f in event.get("frames") or [] if (t := finite(f.get("t_sec"))) is not None and t >= 0]
    start, end = finite(event.get("t_start_sec")), finite(event.get("t_end_sec"))
    if not frames and start is None:
        return None
    return dict(event, confidence=confidence, frames=frames, t_start_sec=start, t_end_sec=end)


def build_violation(event: dict, video_path: str, rules: Rules, settings: Settings) -> dict:
    entry, description, rule = rules.describe(event)
    frames = event.get("frames") or []
    times = [float(f["t_sec"]) for f in frames][: settings.max_frames_per_event]
    if not times:
        times = [float(event["t_start_sec"])]
    boxes_by_time = {float(f["t_sec"]): f.get("boxes") or [] for f in frames}
    decoded = decode_at(video_path, times)
    evidence, boxes, seen = [], [], set()
    for requested, frame in zip(times, decoded):
        if frame.t_sec in seen:
            continue
        seen.add(frame.t_sec)
        evidence.append({"timestamp_sec": frame.t_sec, "image_base64": to_jpeg_base64(frame.image, settings.jpeg_quality)})
        boxes.append({"timestamp_sec": frame.t_sec, "boxes": boxes_by_time.get(requested, [])})
    return {
        "type": entry["name"],
        "timecode_sec": evidence[0]["timestamp_sec"],
        "description": description,
        "evidence_frames": evidence,
        "type_id": entry["type_id"],
        "confidence": float(event.get("confidence", 0)),
        "t_start_sec": event.get("t_start_sec"),
        "t_end_sec": event.get("t_end_sec"),
        "rule": rule,
        "boxes": boxes,
    }


def run(
    video_path: str,
    detector: Callable[[str], dict],
    rules: Rules,
    settings: Settings,
    progress: Callable[[float], None] = lambda _: None,
) -> tuple[dict, dict]:
    raw = detector(video_path)
    progress(0.8)
    events = []
    for event in raw.get("events") or []:
        cleaned = clean_event(event)
        if cleaned is None:
            log.warning("dropping event with non-finite confidence or time: %r", event)
        elif cleaned["confidence"] >= threshold_for(cleaned, rules, settings):
            events.append(cleaned)
    violations = [build_violation(e, video_path, rules, settings) for e in events]
    progress(0.95)
    result = {"verdict": "violation" if violations else "no_violation", "violations": violations}
    return result, raw
