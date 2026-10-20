def _event(confidence, t_sec=1.0):
    return {
        "type_id": "cash_unattended",
        "closest_type_id": None,
        "t_start_sec": 1.0,
        "t_end_sec": 2.0,
        "confidence": confidence,
        "observation": "кассир отошёл от стола",
        "frames": [{"t_sec": t_sec, "boxes": []}],
    }


def analyze(video_path: str) -> dict:
    return {
        "events": [
            _event(float("nan")),
            _event(float("inf")),
            _event(float("-inf")),
            _event("high"),
            _event(None),
            _event(0.9, t_sec=float("nan")) | {"t_start_sec": float("inf")},
            _event(0.9),
        ]
    }
