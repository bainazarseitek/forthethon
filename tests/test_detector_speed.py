from PIL import Image, ImageDraw

from app.detector_vlm import state
from app.detector_vlm.detector import plan

FULL = {"step": 1.0, "detail": True, "fine": True}


def test_long_video_keeps_full_plan_when_real_cost_fits():
    assert plan(143, None) == FULL
    assert plan(143, 130) == FULL
    assert plan(143, 80)["step"] == 2.0
    assert plan(143, 40) == {"step": 3.0, "detail": False, "fine": False}


def test_short_video_degrades_one_step_first():
    assert plan(20, 80) == FULL
    assert plan(20, 50)["step"] == 2.0


def frame(box=None):
    image = Image.new("RGB", (480, 848), (90, 90, 90))
    if box:
        ImageDraw.Draw(image).rectangle(box, fill=(250, 250, 250))
    return image


def test_static_frames_reuse_the_last_asked_frame():
    frames = [(0.0, frame()), (1.0, frame()), (2.0, frame((0, 0, 200, 200))), (3.0, frame((0, 0, 200, 200))), (4.0, frame((0, 0, 200, 200)))]
    assert state.anchors(frames) == [0, 0, 2, 2, 4]


def test_gated_states_keep_their_own_timestamps():
    calls = []

    def post(body):
        calls.append(body)
        content = "{" + ", ".join(f'"{k}": false' for k in state.FIELDS) + "}"
        return {"choices": [{"message": {"content": content}, "logprobs": None}]}

    frames = [(float(t), frame()) for t in range(4)]
    out = state.frame_states(post, lambda image: "data:", frames)
    assert [s["t"] for s in out] == [0.0, 1.0, 2.0, 3.0]
    assert len(calls) == 2
    assert all(set(s) == {"t", *state.FIELDS} for s in out)
