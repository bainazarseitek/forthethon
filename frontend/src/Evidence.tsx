import type { Frame } from "./domain";
import { evidenceSrc } from "./domain";

type Box = NonNullable<Frame["boxes"]>[number];

export function Boxes({ boxes }: { boxes?: Box[] }) {
  if (!boxes?.length) return null;
  return (
    <div className="boxes" aria-hidden="true">
      {boxes.map((b, i) => (
        <span
          key={i}
          className="box"
          style={{
            left: `${clamp(b.x) * 100}%`,
            top: `${clamp(b.y) * 100}%`,
            width: `${clamp(b.w, 1 - clamp(b.x)) * 100}%`,
            height: `${clamp(b.h, 1 - clamp(b.y)) * 100}%`,
          }}
        >
          {b.label && <span className="box-label">{b.label}</span>}
        </span>
      ))}
    </div>
  );
}

function clamp(v: number, max = 1) {
  return Math.min(Math.max(v, 0), max);
}

export function Evidence({
  frame,
  alt,
  boxes = true,
  onLoad,
}: {
  frame: Frame;
  alt: string;
  boxes?: boolean;
  onLoad?: (img: HTMLImageElement) => void;
}) {
  const src = evidenceSrc(frame);
  return (
    <span className="evidence">
      {src ? (
        <img
          src={src}
          alt={alt}
          draggable={false}
          onLoad={onLoad && ((e) => onLoad(e.currentTarget))}
        />
      ) : (
        <span className="evidence-missing">Кадр недоступен</span>
      )}
      {boxes && <Boxes boxes={frame.boxes} />}
    </span>
  );
}
