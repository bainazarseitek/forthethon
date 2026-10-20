import base64
import io
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image


class VideoError(ValueError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    width: int
    height: int
    rotation: int


@dataclass(frozen=True)
class Frame:
    t_sec: float
    image: Image.Image


def probe(path: str | Path) -> VideoInfo:
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoError("no video stream")
            stream = container.streams.video[0]
            first = next(container.decode(stream), None)
            if first is None:
                raise VideoError("no decodable frames")
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            else:
                duration = 0.0
            rotation = int(first.rotation or 0)
            width, height = first.width, first.height
            if rotation % 180:
                width, height = height, width
            return VideoInfo(duration, width, height, rotation)
    except VideoError:
        raise
    except (av.FFmpegError, StopIteration, UnicodeDecodeError) as exc:
        raise VideoError(str(exc)) from exc


def frame_times(path: str | Path) -> list[float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        times = [float(p.pts * stream.time_base) for p in container.demux(stream) if p.pts is not None]
    return sorted(times)


def _upright(frame: av.VideoFrame) -> Image.Image:
    image = frame.to_image()
    rotation = int(frame.rotation or 0) % 360
    if rotation:
        image = image.rotate(rotation, expand=True)
    return image


def decode_at(path: str | Path, times: list[float]) -> list[Frame]:
    result: list[Frame] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for target in times:
            container.seek(max(0, int(target / stream.time_base)), stream=stream, backward=True, any_frame=False)
            previous = None
            chosen = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if frame.time >= target:
                    if previous is not None and target - previous.time < frame.time - target:
                        chosen = previous
                    else:
                        chosen = frame
                    break
                previous = frame
            chosen = chosen or previous
            if chosen is None:
                raise VideoError(f"no frame at {target}")
            result.append(Frame(round(float(chosen.time), 3), _upright(chosen)))
    return result


def to_jpeg_base64(image: Image.Image, quality: int = 85) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
