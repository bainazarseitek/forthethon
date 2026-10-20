import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  CirclePlus,
  FileText,
  Maximize,
  Pause,
  Play,
  Plus,
  RotateCcw,
  ScanSearch,
  X,
  ZoomIn,
} from "lucide-react";
import {
  confidenceLevel,
  confidencePercent,
  episodeRange,
  eventKey,
  safeMediaUrl,
  safeSourceUrl,
  story,
  timecode,
  type Added,
  type CatalogType,
  type CompletedJob,
  type Decision,
  type Event,
} from "./domain";
import { useReviews } from "./useReviews";
import { useAddedViolations } from "./useAddedViolations";
import { AddViolationPanel, AddedList, typeName } from "./AddViolation";
import { useFilmstrip } from "./useFilmstrip";
import { Boxes, Evidence } from "./Evidence";
import { Shell, caseNumber } from "./Shell";

const LEAD = 3;
const FPS = 30;
const STRIP = 24;
const BOX_WINDOW = 0.25;

export const verdictText = {
  violation: "Обнаружены признаки возможных нарушений",
  no_violation: "Нарушение не обнаружено",
} as const;

export function episodes(n: number) {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return `${n} эпизод`;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return `${n} эпизода`;
  return `${n} эпизодов`;
}

export function Review({ job }: { job: CompletedJob }) {
  const events = job.violations;
  const video = useRef<HTMLVideoElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const ambient = useRef<HTMLCanvasElement>(null);
  const reduce = useReducedMotion();
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(job.duration_sec ?? 0);
  const [ratio, setRatio] = useState<number>();
  const [playing, setPlaying] = useState(false);
  const [active, setActive] = useState(0);
  const [direction, setDirection] = useState(1);
  const [rate, setRate] = useState(1);
  const [videoError, setVideoError] = useState(false);
  const [zoom, setZoom] = useState<{ event: number; frame: number }>();
  const [help, setHelp] = useState(false);
  const [flash, setFlash] = useState(0);
  const [toast, setToast] = useState("");
  const loopRange = useRef<{ start: number; end: number }>(undefined);
  const { snapshot, change } = useReviews(job, false);
  const feedback = useAddedViolations(job, snapshot.decisions);
  const [adding, setAdding] = useState(false);

  const src = safeMediaUrl(
    job.video_url ?? `/app/jobs/${encodeURIComponent(job.job_id)}/video`,
    window.location.origin,
  );

  const strip = useFilmstrip(STRIP, src);

  useEffect(() => {
    document.title = `Дело ${caseNumber(job.job_id)} · ${job.filename ?? "видео"}`;
  }, [job.job_id, job.filename]);

  const toastTimer = useRef(0);
  const say = useCallback((text: string) => {
    setToast(text);
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(""), 1600);
  }, []);

  const paintAmbient = useCallback(() => {
    const v = video.current;
    const c = ambient.current;
    if (!v || !c || v.readyState < 2) return;
    try {
      c.getContext("2d")?.drawImage(v, 0, 0, c.width, c.height);
    } catch {
      return;
    }
  }, []);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = 0;
    const step = (now: number) => {
      const v = video.current;
      if (v) {
        const range = loopRange.current;
        if (range && v.currentTime >= range.end) {
          v.pause();
          loopRange.current = undefined;
        }
        setTime(v.currentTime);
        if (now - last > 200) {
          paintAmbient();
          last = now;
        }
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, paintAmbient]);

  useEffect(() => {
    const i = events.findIndex((e) => {
      const r = episodeRange(e, duration);
      return time >= r.start - 0.3 && time <= r.end + 0.3;
    });
    if (i >= 0 && i !== active && playing) {
      setDirection(i > active ? 1 : -1);
      setActive(i);
    }
  }, [time, events, duration, active, playing]);

  const seek = useCallback(
    (t: number) => {
      const v = video.current;
      const next = Math.min(Math.max(t, 0), duration || t);
      if (v) v.currentTime = next;
      setTime(next);
      setFlash((n) => n + 1);
    },
    [duration],
  );

  const goTo = useCallback(
    (i: number) => {
      if (!events.length) return;
      const next = (i + events.length) % events.length;
      setDirection(next >= active ? 1 : -1);
      setActive(next);
      loopRange.current = undefined;
      video.current?.pause();
      seek(events[next].timecode_sec);
    },
    [events, active, seek],
  );

  const replay = useCallback(
    (i = active) => {
      const e = events[i];
      if (!e || !video.current) return;
      const r = episodeRange(e, duration);
      loopRange.current = {
        start: Math.max(0, e.timecode_sec - LEAD),
        end: Math.min(
          Math.max(r.end, e.timecode_sec) + 1,
          duration || r.end + 1,
        ),
      };
      seek(loopRange.current.start);
      video.current.play().catch(() => undefined);
    },
    [active, events, duration, seek],
  );

  const toggle = useCallback(() => {
    const v = video.current;
    if (!v) return;
    if (v.paused) {
      loopRange.current = undefined;
      v.play().catch(() => undefined);
    } else v.pause();
  }, []);

  const startAdding = useCallback(() => {
    video.current?.pause();
    loopRange.current = undefined;
    setAdding(true);
  }, []);

  const stepFrame = useCallback(
    (n: number) => {
      const v = video.current;
      v?.pause();
      seek((Math.round((v?.currentTime ?? time) * FPS) + n) / FPS);
    },
    [seek, time],
  );

  const fullscreen = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen();
    else stage.current?.requestFullscreen?.().catch(() => undefined);
  }, []);

  const setSpeed = useCallback(
    (r: number) => {
      setRate(r);
      if (video.current) video.current.playbackRate = r;
      say(`Скорость ${r}×`);
    },
    [say],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey || zoom || help) return;
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, [contenteditable]")) return;
      const onButton = t.closest("button, a");
      const v = video.current;

      const k = e.key;
      const map: Record<string, () => void> = {
        " ": toggle,
        k: toggle,
        ArrowLeft: () =>
          e.shiftKey ? stepFrame(-1) : seek((v?.currentTime ?? time) - 1),
        ArrowRight: () =>
          e.shiftKey ? stepFrame(1) : seek((v?.currentTime ?? time) + 1),
        ",": () => stepFrame(-1),
        ".": () => stepFrame(1),
        j: () => seek((v?.currentTime ?? time) - 5),
        l: () => seek((v?.currentTime ?? time) + 5),
        n: () => goTo(active + 1),
        p: () => goTo(active - 1),
        r: () => replay(),
        f: fullscreen,
        "0": () => seek(0),
        Home: () => seek(0),
        "?": () => setHelp(true),
        a: startAdding,
        s: () => setSpeed(rate === 1 ? 0.5 : rate === 0.5 ? 0.25 : 1),
      };
      const lower = k.length === 1 ? k.toLowerCase() : k;
      const latin: Record<string, string> = {
        л: "k",
        т: "n",
        о: "j",
        з: "p",
        к: "r",
        д: "l",
        а: "f",
        ы: "s",
        б: ",",
        ю: ".",
        ф: "a",
      };
      const key = latin[lower] ?? lower;
      if (/^[1-9]$/.test(key) && Number(key) <= events.length) {
        e.preventDefault();
        goTo(Number(key) - 1);
        return;
      }
      if (key === " " && onButton) return;
      const action = map[key];
      if (action) {
        e.preventDefault();
        action();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    toggle,
    seek,
    goTo,
    replay,
    stepFrame,
    time,
    fullscreen,
    setSpeed,
    rate,
    active,
    events.length,
    zoom,
    help,
    startAdding,
  ]);

  const liveBoxes = useMemo(() => {
    for (const e of events)
      for (const f of e.evidence_frames)
        if (Math.abs(f.timestamp_sec - time) <= BOX_WINDOW && f.boxes?.length)
          return f.boxes;
    return undefined;
  }, [events, time]);

  const current = events[active];
  const currentRange = current
    ? episodeRange(current, duration)
    : { start: 0, end: 0 };
  const isViolation = job.verdict === "violation";
  const decisions = snapshot.decisions;

  return (
    <Shell
      trail={
        <span className="trail-case">
          Дело {caseNumber(job.job_id)}
          <span className="trail-file">{job.filename}</span>
        </span>
      }
      actions={
        <>
          <Link className="btn btn-quiet" to="/" aria-label="Новая проверка">
            <Plus aria-hidden="true" />
            <span className="hide-sm">Новая проверка</span>
          </Link>
          <Link className="btn btn-primary" to={`/jobs/${job.job_id}/report`}>
            <FileText aria-hidden="true" />
            Отчёт
          </Link>
        </>
      }
    >
      <main
        className="review"
        id="main"
        data-orientation={
          ratio === undefined ? undefined : ratio < 1 ? "portrait" : "landscape"
        }
      >
        <div className={`review-grid ${isViolation ? "" : "is-clean"}`}>
          <div className="side">
            <motion.header
              className={`verdict ${isViolation ? "is-alert" : "is-clean"}`}
              initial={reduce ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45 }}
            >
              <p className="overline">Заключение модели</p>
              <h1 className="verdict-title">
                <span className="verdict-mark" aria-hidden="true" />
                {verdictText[job.verdict]}
              </h1>
              <p className="verdict-sub">
                {isViolation
                  ? `${episodes(events.length)} для проверки. `
                  : "Модель не нашла эпизодов с признаками нарушения. "}
                Это подсказка для проверяющего, а не юридическое заключение.
              </p>
            </motion.header>
            {events.length > 0 && (
              <nav className="log" aria-label="Журнал эпизодов">
                <h2 className="rail-title">Журнал эпизодов</h2>
                <ol className="index">
                  {events.map((e, i) => (
                    <li key={eventKey(e, i)}>
                      <button
                        className={`index-item ${i === active ? "is-active" : ""}`}
                        aria-current={i === active ? "true" : undefined}
                        onClick={() => goTo(i)}
                      >
                        <span className="index-body">
                          <span className="index-type">{e.type}</span>
                          <span className="index-meta">
                            <span className="mono">
                              {timecode(e.timecode_sec)}
                            </span>
                            {e.confidence !== undefined && (
                              <span>{confidencePercent(e.confidence)}</span>
                            )}
                          </span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ol>
              </nav>
            )}
            <AddedList
              added={feedback.added}
              catalog={feedback.catalog}
              onSeek={(t) => {
                video.current?.pause();
                seek(t);
              }}
              onRemove={async (id) => {
                await feedback.remove(id);
                say("Нарушение удалено");
              }}
            />
          </div>
          <div className="work">
            <section className="player" aria-label="Видеозапись">
              <div
                className="stage"
                ref={stage}
                style={{ "--ratio": ratio ?? 9 / 16 } as React.CSSProperties}
              >
                <canvas
                  ref={ambient}
                  className="ambient"
                  width={18}
                  height={32}
                  aria-hidden="true"
                />
                <div
                  className="screen"
                  style={{ aspectRatio: String(ratio ?? 9 / 16) }}
                >
                  {src && !videoError ? (
                    <video
                      ref={video}
                      src={src}
                      preload="auto"
                      playsInline
                      muted
                      onClick={toggle}
                      onLoadedMetadata={(e) => {
                        const v = e.currentTarget;
                        if (v.videoWidth)
                          setRatio(v.videoWidth / v.videoHeight);
                        if (Number.isFinite(v.duration))
                          setDuration(v.duration);
                      }}
                      onLoadedData={paintAmbient}
                      onSeeked={(e) => {
                        setTime(e.currentTarget.currentTime);
                        paintAmbient();
                      }}
                      onPlay={() => setPlaying(true)}
                      onPause={(e) => {
                        setPlaying(false);
                        setTime(e.currentTarget.currentTime);
                      }}
                      onEnded={() => setPlaying(false)}
                      onError={() => setVideoError(true)}
                      aria-label={`Видео ${job.filename ?? ""}`}
                    />
                  ) : (
                    <p className="screen-error">
                      Видео недоступно в браузере. Кадры-доказательства ниже
                      остаются в силе.
                    </p>
                  )}
                  <Boxes boxes={liveBoxes} />
                  {flash > 0 && (
                    <span
                      key={flash}
                      className="seek-flash"
                      aria-hidden="true"
                    />
                  )}
                  <span className="stage-tc" aria-hidden="true">
                    {timecode(time)}
                  </span>
                  <AnimatePresence>
                    {toast && (
                      <motion.span
                        key="toast"
                        className="toast"
                        role="status"
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                      >
                        {toast}
                      </motion.span>
                    )}
                  </AnimatePresence>
                  {!playing && !videoError && time === 0 && (
                    <button
                      className="big-play"
                      onClick={toggle}
                      aria-label="Воспроизвести"
                      tabIndex={-1}
                    >
                      <Play aria-hidden="true" />
                    </button>
                  )}
                </div>
              </div>

              {current && (
                <div className="episode-bar">
                  <IconButton
                    label="Предыдущее нарушение (P)"
                    onClick={() => goTo(active - 1)}
                  >
                    <ChevronLeft />
                  </IconButton>
                  <button
                    className="episode-bar-main"
                    onClick={() => goTo(active)}
                    aria-label={`Перейти к эпизоду ${active + 1}: ${current.type}, ${timecode(current.timecode_sec, true)}`}
                  >
                    <span className="episode-bar-n mono">
                      {String(active + 1).padStart(2, "0")}/
                      {String(events.length).padStart(2, "0")}
                    </span>
                    <span className="episode-bar-type">{current.type}</span>
                    <span className="episode-bar-tc mono">
                      {timecode(current.timecode_sec, true)}
                      <span>
                        {current.t_start_sec === current.t_end_sec
                          ? currentRange.start.toFixed(1)
                          : `${currentRange.start.toFixed(1)}–${currentRange.end.toFixed(1)}`}{" "}
                        с
                      </span>
                    </span>
                  </button>
                  <IconButton
                    label="Следующее нарушение (N)"
                    onClick={() => goTo(active + 1)}
                  >
                    <ChevronRight />
                  </IconButton>
                </div>
              )}
              <Timeline
                strip={strip}
                events={events}
                duration={duration}
                time={time}
                active={active}
                playing={playing}
                decisions={decisions}
                added={feedback.added}
                catalog={feedback.catalog}
                onSeek={seek}
                onPick={goTo}
              />

              <div
                className="transport"
                role="toolbar"
                aria-label="Управление видео"
              >
                <IconButton
                  label={playing ? "Пауза (пробел)" : "Воспроизвести (пробел)"}
                  onClick={toggle}
                  primary
                >
                  {playing ? <Pause /> : <Play />}
                </IconButton>
                {events.length > 0 && (
                  <>
                    <button
                      className="btn btn-quiet transport-btn transport-step"
                      onClick={() => goTo(active - 1)}
                      title="Предыдущее нарушение (P)"
                    >
                      <ChevronLeft aria-hidden="true" />
                      <span className="transport-label">
                        Предыдущее нарушение
                      </span>
                    </button>
                    <button
                      className="btn btn-quiet transport-btn transport-step"
                      onClick={() => goTo(active + 1)}
                      title="Следующее нарушение (N)"
                    >
                      <span className="transport-label">
                        Следующее нарушение
                      </span>
                      <ChevronRight aria-hidden="true" />
                    </button>
                    <button
                      className="btn btn-quiet transport-btn"
                      onClick={() => replay()}
                      title={`Повторить эпизод с ${LEAD} с до начала (R)`}
                    >
                      <RotateCcw aria-hidden="true" />
                      <span className="transport-label">Повторить эпизод</span>
                    </button>
                  </>
                )}
                <span className="transport-gap" />
                <button
                  className="btn transport-btn transport-add"
                  onClick={startAdding}
                  aria-pressed={adding}
                  title="Модель пропустила нарушение? Добавьте его с текущего кадра (A)"
                >
                  <CirclePlus aria-hidden="true" />
                  <span className="transport-label">Добавить нарушение</span>
                </button>
                <IconButton label="Во весь экран (F)" onClick={fullscreen}>
                  <Maximize />
                </IconButton>
              </div>
            </section>
            <section className="finding-col" aria-label="Текущий эпизод">
              {adding ? (
                <AddViolationPanel
                  catalog={feedback.catalog}
                  loadError={feedback.loadError}
                  time={time}
                  duration={duration}
                  onSeek={(t) => {
                    video.current?.pause();
                    seek(t);
                  }}
                  onCancel={() => setAdding(false)}
                  onSave={async (item) => {
                    await feedback.add(item);
                    setAdding(false);
                    say("Нарушение добавлено");
                  }}
                />
              ) : current ? (
                <AnimatePresence mode="wait" custom={direction} initial={false}>
                  <motion.div
                    key={active}
                    custom={direction}
                    initial={
                      reduce
                        ? { opacity: 0 }
                        : { opacity: 0, x: 24 * direction }
                    }
                    animate={{ opacity: 1, x: 0 }}
                    exit={
                      reduce
                        ? { opacity: 0 }
                        : { opacity: 0, x: -24 * direction }
                    }
                    transition={{ duration: 0.28, ease: [0.2, 0.7, 0.2, 1] }}
                  >
                    <FindingCard
                      event={current}
                      decision={decisions[eventKey(current, active)]}
                      onDecide={(d) => change(eventKey(current, active), d)}
                      onSeek={seek}
                      onZoom={(frame) => setZoom({ event: active, frame })}
                    />
                  </motion.div>
                </AnimatePresence>
              ) : (
                <CleanCard onAdd={startAdding} />
              )}
            </section>
          </div>
        </div>
      </main>

      {zoom && events[zoom.event] && (
        <Lightbox
          event={events[zoom.event]}
          start={zoom.frame}
          onClose={() => setZoom(undefined)}
          onShow={(t) => {
            setZoom(undefined);
            video.current?.pause();
            seek(t);
          }}
        />
      )}
      {help && <Help onClose={() => setHelp(false)} />}
    </Shell>
  );
}

function IconButton({
  label,
  onClick,
  children,
  pressed,
  primary,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
  pressed?: boolean;
  primary?: boolean;
}) {
  return (
    <button
      className={`icon-btn ${primary ? "is-primary" : ""}`}
      onClick={onClick}
      aria-label={label}
      aria-pressed={pressed}
      title={label}
    >
      {children}
    </button>
  );
}

function Timeline({
  strip,
  events,
  duration,
  time,
  active,
  playing,
  decisions,
  added,
  catalog,
  onSeek,
  onPick,
}: {
  strip: string[];
  events: Event[];
  duration: number;
  time: number;
  active: number;
  playing: boolean;
  decisions: Record<string, Decision>;
  added: Added[];
  catalog: CatalogType[];
  onSeek: (t: number) => void;
  onPick: (i: number) => void;
}) {
  const track = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number>();
  const pct = (t: number) =>
    `${duration ? Math.min(100, (t / duration) * 100) : 0}%`;
  function at(e: ReactPointerEvent) {
    const r = track.current!.getBoundingClientRect();
    return Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1) * duration;
  }
  const ticks = Math.max(1, Math.round(duration));
  return (
    <div className="timeline">
      <div className="pins">
        {events.map((e, i) => {
          const d = decisions[eventKey(e, i)]?.status;
          return (
            <button
              key={eventKey(e, i)}
              className={`pin ${e.timecode_sec > duration / 2 ? "tip-left" : "tip-right"} ${i === active ? "is-active" : ""} ${d && d !== "unreviewed" ? `pin-${d}` : ""}`}
              style={{ left: pct(e.timecode_sec) }}
              onClick={() => onPick(i)}
              aria-label={`Эпизод ${i + 1}: ${e.type}, ${timecode(e.timecode_sec)}`}
            >
              <span className="pin-head">{i + 1}</span>
              <span className="pin-tip" role="tooltip">
                <span className="mono">{timecode(e.timecode_sec)}</span>
                {e.type}
              </span>
            </button>
          );
        })}
        {added.map((a) => (
          <button
            key={a.id}
            className={`pin pin-added ${a.t_start_sec > duration / 2 ? "tip-left" : "tip-right"}`}
            style={{ left: pct(a.t_start_sec) }}
            onClick={() => onSeek(a.t_start_sec)}
            aria-label={`Добавлено проверяющим: ${typeName(catalog, a.type_id)}, ${timecode(a.t_start_sec)}`}
          >
            <span className="pin-head">+</span>
            <span className="pin-tip" role="tooltip">
              <span className="mono">{timecode(a.t_start_sec)}</span>
              {typeName(catalog, a.type_id)} · проверяющий
            </span>
          </button>
        ))}
      </div>
      <div
        ref={track}
        className={`track ${playing ? "is-playing" : ""}`}
        role="slider"
        tabIndex={0}
        aria-label="Позиция в видео"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(time)}
        aria-valuetext={timecode(time)}
        style={{ ["--ticks" as string]: ticks }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          onSeek(at(e));
        }}
        onPointerMove={(e) => {
          setHover(at(e));
          if (e.buttons === 1) onSeek(at(e));
        }}
        onPointerLeave={() => setHover(undefined)}
      >
        {strip.length > 0 && (
          <span className="filmstrip" aria-hidden="true">
            {strip.map((t, i) => (
              <img key={i} src={t} alt="" />
            ))}
          </span>
        )}
        <span className="track-fill" style={{ width: pct(time) }} />
        {events.map((e, i) => {
          const r = episodeRange(e, duration);
          return (
            <span
              key={eventKey(e, i)}
              className={`chapter ${i === active ? "is-active" : ""}`}
              style={{
                left: pct(Math.max(0, e.timecode_sec - LEAD)),
                width: `calc(${pct(Math.max(r.end, e.timecode_sec) - Math.max(0, e.timecode_sec - LEAD))} + 2px)`,
              }}
            >
              <span
                className="chapter-core"
                style={{
                  left: `${((Math.min(r.start, e.timecode_sec) - Math.max(0, e.timecode_sec - LEAD)) / Math.max(0.01, Math.max(r.end, e.timecode_sec) - Math.max(0, e.timecode_sec - LEAD))) * 100}%`,
                }}
              />
            </span>
          );
        })}
        <span className="playhead" style={{ left: pct(time) }} />
        {hover !== undefined && (
          <span className="hover-time" style={{ left: pct(hover) }}>
            {timecode(hover)}
          </span>
        )}
      </div>
      <div className="track-scale">
        <span aria-hidden="true">00:00</span>
        <span className="time" aria-label="Текущее время">
          {timecode(time, true)}
        </span>
        <span aria-hidden="true">{timecode(duration)}</span>
      </div>
    </div>
  );
}

const decisionOptions: { status: Decision["status"]; label: string }[] = [
  { status: "accepted", label: "Подтверждаю" },
  { status: "uncertain", label: "Нужна проверка" },
  { status: "rejected", label: "Отклоняю" },
];

const FindingCard = memo(function FindingCard({
  event,
  decision,
  onDecide,
  onSeek,
  onZoom,
}: {
  event: Event;
  decision?: Decision;
  onDecide: (d: Decision) => void;
  onSeek: (t: number) => void;
  onZoom: (frame: number) => void;
}) {
  const s = story(event);
  const url = safeSourceUrl(event.rule?.url);
  const steps = [
    {
      title: "Норма",
      body: event.rule ? (
        <div className="citation">
          {event.rule.clause && (
            <p className="citation-clause">{event.rule.clause}</p>
          )}
          <p className="citation-doc">{event.rule.document}</p>
          {!url && (
            <p className="citation-doc">
              Внутренний документ банка, ссылки на adilet нет.
            </p>
          )}
          {url && (
            <a
              className="citation-link"
              href={url}
              target="_blank"
              rel="noreferrer noopener"
            >
              Открыть на adilet.zan.kz <ArrowUpRight aria-hidden="true" />
            </a>
          )}
        </div>
      ) : (
        <p className="muted">
          Норма не указана моделью. Сверьтесь с внутренними правилами банка.
        </p>
      ),
    },
    { title: "Что видно на записи", body: <p>{s.seen}</p> },
    ...(s.why
      ? [{ title: "Почему это может быть нарушением", body: <p>{s.why}</p> }]
      : []),
  ];
  return (
    <article className="finding">
      <header className="finding-head">
        <div className="finding-id">
          <Confidence value={event.confidence} />
          {event.type_id === "other" && (
            <p className="note">
              Тип вне каталога правил, требует уточнения проверяющим.
            </p>
          )}
        </div>
        <fieldset className="decision">
          <legend>Решение проверяющего</legend>
          <div className="segmented">
            {decisionOptions.map((o) => (
              <button
                key={o.status}
                className={`seg seg-${o.status}`}
                aria-pressed={decision?.status === o.status}
                onClick={() =>
                  onDecide({
                    ...decision,
                    status:
                      decision?.status === o.status ? "unreviewed" : o.status,
                  })
                }
              >
                {o.label}
              </button>
            ))}
          </div>
        </fieldset>
      </header>
      <div className="finding-body">
        <ol className="story">
          {steps.map((step, i) => (
            <li key={step.title}>
              <h3>{step.title}</h3>
              {step.body}
            </li>
          ))}
        </ol>
        {event.evidence_frames.length > 0 && (
          <div className="frames">
            <h3 className="frames-title">Кадры-доказательства</h3>
            <div className="frames-row">
              {event.evidence_frames.map((f, i) => (
                <button
                  key={i}
                  className="frame-btn"
                  onClick={() => onZoom(i)}
                  aria-label={`Увеличить кадр ${i + 1}, ${timecode(f.timestamp_sec, true)}`}
                >
                  <Evidence frame={f} alt="" />
                  <span className="frame-cap">
                    <span className="mono">{f.timestamp_sec.toFixed(1)} с</span>
                    <ZoomIn aria-hidden="true" />
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </article>
  );
});

export function Confidence({ value }: { value?: number }) {
  if (value === undefined) return null;
  return (
    <p className="confidence">
      Уверенность модели <strong>{confidencePercent(value)}</strong> ·{" "}
      {confidenceLevel(value)}
    </p>
  );
}

function CleanCard({ onAdd }: { onAdd: () => void }) {
  return (
    <article className="finding finding-clean">
      <p className="overline">Результат</p>
      <h2 className="finding-title">Эпизодов для проверки нет</h2>
      <ol className="story">
        <li>
          <div>
            <h3>
              <span className="story-n" aria-hidden="true">
                ✓
              </span>
              Что это значит
            </h3>
            <p>
              Модель просмотрела всю запись и не нашла моментов, похожих на
              нарушение порядка кассовых операций из каталога правил.
            </p>
          </div>
        </li>
        <li>
          <div>
            <h3>
              <span className="story-n" aria-hidden="true">
                !
              </span>
              Что сделать проверяющему
            </h3>
            <p>
              Модель может ошибаться. При сомнениях просмотрите запись на
              ускоренной скорости и зафиксируйте результат в отчёте.
            </p>
          </div>
        </li>
        <li>
          <div>
            <h3>
              <span className="story-n" aria-hidden="true">
                +
              </span>
              Видите нарушение, которое модель пропустила?
            </h3>
            <p>
              Остановите видео на кадре, где оно начинается, и добавьте его.
            </p>
            <button className="btn" onClick={onAdd}>
              <CirclePlus aria-hidden="true" />
              Добавить нарушение
            </button>
          </div>
        </li>
      </ol>
    </article>
  );
}

function useDialog(onClose: () => void) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    const opener = document.activeElement as HTMLElement | null;
    d.showModal();
    const close = () => onClose();
    d.addEventListener("close", close);
    return () => {
      d.removeEventListener("close", close);
      opener?.focus?.();
    };
  }, [onClose]);
  return ref;
}

function Lightbox({
  event,
  start,
  onClose,
  onShow,
}: {
  event: Event;
  start: number;
  onClose: () => void;
  onShow: (t: number) => void;
}) {
  const ref = useDialog(onClose);
  const [i, setI] = useState(start);
  const [boxes, setBoxes] = useState(true);
  const [zoomAt, setZoomAt] = useState<{ x: number; y: number }>();
  const frames = event.evidence_frames;
  const f = frames[i];
  const hasBoxes = frames.some((frame) => frame.boxes?.length);
  const [ratio, setRatio] = useState(9 / 16);
  const frameRef = useRef<HTMLDivElement>(null);
  useEffect(() => setZoomAt(undefined), [i]);
  const pointAt = (e: React.MouseEvent<HTMLElement>) => {
    const el = frameRef.current!;
    const r = e.currentTarget.getBoundingClientRect();
    const pct = (v: number, size: number) =>
      Math.min(100, Math.max(0, (v / size) * 100));
    return {
      x: pct(e.clientX - r.left - el.offsetLeft, el.offsetWidth),
      y: pct(e.clientY - r.top - el.offsetTop, el.offsetHeight),
    };
  };
  return (
    <dialog
      ref={ref}
      className="lightbox"
      style={{ "--ratio": ratio } as React.CSSProperties}
      aria-label={`Кадры эпизода: ${event.type}`}
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") setI((n) => (n + 1) % frames.length);
        if (e.key === "ArrowLeft")
          setI((n) => (n - 1 + frames.length) % frames.length);
        if (e.key.toLowerCase() === "b" || e.key.toLowerCase() === "и")
          setBoxes((b) => !b);
      }}
      onClick={(e) => e.target === e.currentTarget && ref.current?.close()}
    >
      <div className="lightbox-inner">
        <header className="lightbox-head">
          <div>
            <p className="overline">
              Кадр {i + 1} из {frames.length} ·{" "}
              <span className="mono">{timecode(f.timestamp_sec, true)}</span>
            </p>
            <h2>{event.type}</h2>
          </div>
          <button
            className="icon-btn"
            onClick={() => ref.current?.close()}
            aria-label="Закрыть (Esc)"
            autoFocus
          >
            <X />
          </button>
        </header>
        <div
          className={`lightbox-stage ${zoomAt ? "is-zoomed" : ""}`}
          onClick={(e) => setZoomAt(zoomAt ? undefined : pointAt(e))}
          onPointerMove={(e) => zoomAt && setZoomAt(pointAt(e))}
          onWheel={(e) => {
            if (e.deltaY < 0) setZoomAt(pointAt(e));
            else if (e.deltaY > 0) setZoomAt(undefined);
          }}
          title={
            zoomAt ? "Нажмите, чтобы уменьшить" : "Нажмите, чтобы увеличить"
          }
        >
          <div
            ref={frameRef}
            className="lightbox-frame"
            style={
              zoomAt
                ? {
                    transform: "scale(2)",
                    transformOrigin: `${zoomAt.x}% ${zoomAt.y}%`,
                  }
                : undefined
            }
          >
            <Evidence
              frame={f}
              alt={`Кадр ${i + 1}: ${event.type}`}
              boxes={boxes}
              onLoad={(img) =>
                img.naturalHeight &&
                setRatio(img.naturalWidth / img.naturalHeight)
              }
            />
          </div>
        </div>
        <footer className="lightbox-foot">
          <div className="row">
            <button
              className="icon-btn"
              onClick={() =>
                setI((n) => (n - 1 + frames.length) % frames.length)
              }
              aria-label="Предыдущий кадр (←)"
              disabled={frames.length < 2}
            >
              <ChevronLeft />
            </button>
            <div className="dots" aria-hidden="true">
              {frames.map((_, n) => (
                <span key={n} className={n === i ? "on" : ""} />
              ))}
            </div>
            <button
              className="icon-btn"
              onClick={() => setI((n) => (n + 1) % frames.length)}
              aria-label="Следующий кадр (→)"
              disabled={frames.length < 2}
            >
              <ChevronRight />
            </button>
          </div>
          <div className="row">
            {hasBoxes && (
              <button
                className="btn btn-quiet"
                aria-pressed={boxes}
                onClick={() => setBoxes((b) => !b)}
              >
                <ScanSearch aria-hidden="true" />{" "}
                {boxes ? "Скрыть рамки" : "Показать рамки"}
              </button>
            )}
            <button className="btn" onClick={() => onShow(f.timestamp_sec)}>
              <Play aria-hidden="true" /> Показать в видео
            </button>
          </div>
        </footer>
      </div>
    </dialog>
  );
}

export const shortcuts: [string[], string][] = [
  [["Пробел", "K"], "Воспроизвести / пауза"],
  [["J", "L"], "Назад / вперёд на 5 с"],
  [["←", "→"], "Назад / вперёд на 1 с"],
  [["Shift", "← →"], "Кадр назад / вперёд"],
  [[",", "."], "Кадр назад / вперёд"],
  [["N", "P"], "Следующий / предыдущий эпизод"],
  [["1–9"], "Перейти к эпизоду по номеру"],
  [["R"], "Повторить эпизод"],
  [["0"], "В начало записи"],
  [["S"], "Скорость 1× / 0,5× / 0,25×"],
  [["F"], "Во весь экран"],
  [["A"], "Добавить нарушение, пропущенное моделью"],
  [["?"], "Эта подсказка"],
];

function Help({ onClose }: { onClose: () => void }) {
  const ref = useDialog(onClose);
  return (
    <dialog
      ref={ref}
      className="help"
      aria-labelledby="help-title"
      onClick={(e) => e.target === e.currentTarget && ref.current?.close()}
    >
      <header className="lightbox-head">
        <h2 id="help-title">Горячие клавиши</h2>
        <button
          className="icon-btn"
          onClick={() => ref.current?.close()}
          aria-label="Закрыть (Esc)"
          autoFocus
        >
          <X />
        </button>
      </header>
      <dl className="keys">
        {shortcuts.map(([keys, text]) => (
          <div key={text}>
            <dt>
              {keys.map((k) => (
                <kbd key={k}>{k}</kbd>
              ))}
            </dt>
            <dd>{text}</dd>
          </div>
        ))}
      </dl>
      <p className="fineprint">Работают и в русской раскладке.</p>
    </dialog>
  );
}
