import subprocess

from PIL import Image, ImageChops

from app.frames import decode_at, frame_times, probe


def corner_index(image: Image.Image) -> int:
    luma = image.convert("L").crop((8, 8, 56, 56))
    values = sorted(luma.get_flattened_data())
    return round(values[len(values) // 2] * 219 / 255 / 8)


def test_synthetic_video_is_vfr(videos):
    times = frame_times(videos["vfr"])
    gaps = {round(b - a, 3) for a, b in zip(times, times[1:])}
    assert gaps == {0.033, 0.067}
    assert any(abs(t - i / 30) > 0.02 for i, t in enumerate(times))


def test_decoded_time_matches_pts(videos):
    times = frame_times(videos["vfr"])
    picks = [t for t, prev in zip(times[1:], times) if t - prev > 0.05][:5] + [times[0], times[-1]]
    frames = decode_at(videos["vfr"], picks)
    for target, frame in zip(picks, frames):
        assert frame.t_sec == round(target, 3)
        assert corner_index(frame.image) == round(target * 30) % 25


def test_index_based_time_would_pick_wrong_frame(videos):
    times = frame_times(videos["vfr"])
    i = next(i for i, t in enumerate(times) if abs(t - i / 30) > 0.02)
    frame = decode_at(videos["vfr"], [times[i]])[0]
    assert corner_index(frame.image) == round(times[i] * 30) % 25
    assert corner_index(frame.image) != i % 25


def test_rounded_request_snaps_to_nearest_frame(videos):
    times = frame_times(videos["vfr"])
    target = times[20]
    for requested in (round(target, 3), target - 0.01, target + 0.01):
        assert decode_at(videos["vfr"], [requested])[0].t_sec == round(target, 3)


def test_rotation_applied(videos, tmp_path):
    info = probe(videos["rotated"])
    assert (info.width, info.height, info.rotation) == (240, 320, 90)
    reference = tmp_path / "ref.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(videos["rotated"]), "-frames:v", "1", str(reference)], check=True)
    frame = decode_at(videos["rotated"], [0.0])[0]
    expected = Image.open(reference).convert("RGB")
    assert frame.image.size == expected.size
    diff = ImageChops.difference(frame.image, expected).convert("L")
    assert max(diff.get_flattened_data()) <= 2
