import { useEffect, useState } from "react";

export function useFilmstrip(count: number, src?: string) {
  const [thumbs, setThumbs] = useState<string[]>([]);
  useEffect(() => {
    if (!src) return;
    let alive = true;
    const v = document.createElement("video");
    v.muted = true;
    v.preload = "auto";
    v.src = src;
    const canvas = document.createElement("canvas");
    const out: string[] = [];
    v.addEventListener(
      "loadeddata",
      async () => {
        const d = v.duration;
        if (!Number.isFinite(d) || d <= 0 || !v.videoWidth) return;
        const scale = 200 / Math.max(v.videoWidth, v.videoHeight);
        canvas.width = Math.round(v.videoWidth * scale);
        canvas.height = Math.round(v.videoHeight * scale);
        const ctx = canvas.getContext("2d");
        for (let i = 0; i < count && alive; i++) {
          v.currentTime = (d * (i + 0.5)) / count;
          await new Promise((r) =>
            v.addEventListener("seeked", r, { once: true }),
          );
          ctx?.drawImage(v, 0, 0, canvas.width, canvas.height);
          out.push(canvas.toDataURL("image/jpeg", 0.7));
          if (alive) setThumbs([...out]);
        }
      },
      { once: true },
    );
    return () => {
      alive = false;
      v.removeAttribute("src");
      v.load();
    };
  }, [count, src]);
  return thumbs;
}
