"""Serve the local review with byte ranges so browsers can seek in videos."""
import argparse
import errno
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "review"


def byte_range(header, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError("Invalid range")
    first, last = match.groups()
    if not first:
        if int(last) <= 0:
            raise ValueError("Invalid suffix")
        return max(0, size - int(last)), size - 1
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable range")
    return start, end


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path))
        header = self.headers.get("Range")
        if not header or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        try:
            start, end = byte_range(header, size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        stream = path.open("rb")
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining:
                chunk = source.read(min(self.remaining, 64 * 1024))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Seeking or changing clips cancels the previous download.


if __name__ == "__main__":
    assert byte_range("bytes=0-9", 100) == (0, 9)
    assert byte_range("bytes=90-", 100) == (90, 99)
    assert byte_range("bytes=-20", 100) == (80, 99)
    assert byte_range("bytes=90-200", 100) == (90, 99)
    for invalid in ["bytes=100-", "bytes=8-3", "bytes=-0", "bytes=-", "bytes=0-1,3-4"]:
        try:
            byte_range(invalid, 100)
        except ValueError:
            continue
        raise AssertionError(invalid)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766, help="Local port (default: 8766)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    url = f"http://127.0.0.1:{args.port}"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            parser.exit(2, f"Port {args.port} is already in use.\n"
                        f"If the review server is already running, open {url}\n"
                        "Otherwise choose another port, for example: python3 scripts/serve_review.py --port 8767\n")
        raise
    print(f"Clip review: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
