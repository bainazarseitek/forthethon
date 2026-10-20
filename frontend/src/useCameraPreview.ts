import { useCallback, useEffect, useRef, useState } from "react";
import { previewSocketUrl } from "./cameraApi";

// One WebSocket carries every camera's preview as "<camera id>\n<jpeg>". Frames go
// straight into the <img> elements, so React does not re-render eight times a second.
export function useCameraPreview(enabled: boolean) {
  const images = useRef(new Map<string, HTMLImageElement>());
  const urls = useRef(new Map<string, string>());
  const lastFrame = useRef(new Map<string, number>());
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let socket: WebSocket | undefined;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let closed = false;
    const connect = () => {
      socket = new WebSocket(previewSocketUrl(window.location));
      socket.binaryType = "arraybuffer";
      socket.onopen = () => setConnected(true);
      socket.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        const bytes = new Uint8Array(e.data);
        const split = bytes.indexOf(10);
        if (split < 1) return;
        const id = new TextDecoder().decode(bytes.subarray(0, split));
        const url = URL.createObjectURL(
          new Blob([bytes.subarray(split + 1)], { type: "image/jpeg" }),
        );
        const img = images.current.get(id);
        if (img) img.src = url;
        const previous = urls.current.get(id);
        if (previous) URL.revokeObjectURL(previous);
        urls.current.set(id, url);
        lastFrame.current.set(id, Date.now());
      };
      socket.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 1500);
      };
    };
    connect();
    const all = urls.current;
    return () => {
      closed = true;
      clearTimeout(retry);
      socket?.close();
      for (const url of all.values()) URL.revokeObjectURL(url);
      all.clear();
    };
  }, [enabled]);

  // Ref callback for a camera's <img>: shows the last frame at once when it mounts.
  const bind = useCallback(
    (id: string) => (img: HTMLImageElement | null) => {
      if (!img) {
        images.current.delete(id);
        return;
      }
      images.current.set(id, img);
      const url = urls.current.get(id);
      if (url) img.src = url;
    },
    [],
  );
  const hasSignal = useCallback(
    (id: string) => Date.now() - (lastFrame.current.get(id) ?? 0) < 3000,
    [],
  );
  return { bind, connected, hasSignal };
}
