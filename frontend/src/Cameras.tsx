import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  ArrowUpRight,
  Cctv,
  CircleAlert,
  CircleStop,
  LoaderCircle,
  Play,
} from "lucide-react";
import { ApiError } from "./api";
import {
  cameraAlerts,
  cameraCounts,
  clock,
  detectionDelay,
  getCameras,
  segmentLabel,
  startCameras,
  stopCameras,
  type Alert,
  type Camera,
  type CameraStatus,
  type Segment,
} from "./cameraApi";
import { timecode } from "./domain";
import { Shell } from "./Shell";
import { useCameraPreview } from "./useCameraPreview";

const STRIP = 8;

export function CamerasPage() {
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["cameras"],
    queryFn: ({ signal }) => getCameras(signal),
    refetchInterval: 2000,
  });
  const status = query.data;
  const running = status?.running ?? false;
  const preview = useCameraPreview(running);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const offset = useRef(0);
  const [, tick] = useState(0);
  const flashing = useFlashing(status);

  useEffect(() => {
    document.title = "Камеры · Forte Review";
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    if (status) offset.current = status.now * 1000 - Date.now();
  }, [status]);

  async function toggle() {
    setBusy(true);
    setError("");
    try {
      client.setQueryData(
        ["cameras"],
        running ? await stopCameras() : await startCameras(),
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Не удалось.");
    } finally {
      setBusy(false);
    }
  }

  const now = (Date.now() + offset.current) / 1000;
  const alerts = status ? cameraAlerts(status) : [];
  const counts = cameraCounts(status?.segments ?? []);

  return (
    <Shell
      trail={<span className="trail-case">Видеонаблюдение</span>}
      actions={
        running && (
          <button className="btn" onClick={toggle} disabled={busy}>
            <CircleStop aria-hidden="true" /> Остановить
          </button>
        )
      }
    >
      <main className="cams" id="main">
        <header className="cams-head">
          <div>
            <p className="overline">Видеонаблюдение · анализ в реальном времени</p>
            <h1 className="display-m">Камеры кассовой зоны</h1>
          </div>
          {status && <OnAir status={status} counts={counts} />}
        </header>

        {running && status?.jury_waiting && (
          <p className="alert alert-warn" role="status">
            <CircleAlert aria-hidden="true" />
            <span>
              Идёт проверка видео через API. Чтобы её не замедлять, новые
              сегменты с камер сейчас не анализируются.
            </span>
          </p>
        )}

        <div className="cams-grid">
          <section
            className="cams-wall"
            data-count={running ? status?.cameras.length : 0}
            aria-label="Трансляции"
          >
            {running && status ? (
              status.cameras.map((camera) => (
                <CameraTile
                  key={camera.id}
                  camera={camera}
                  segments={status.segments.filter(
                    (s) => s.camera_id === camera.id,
                  )}
                  now={now}
                  bind={preview.bind(camera.id)}
                  signal={preview.hasSignal(camera.id)}
                  flash={flashing.has(camera.id)}
                />
              ))
            ) : (
              <CamerasOff
                status={status}
                loading={query.isPending}
                loadError={
                  query.error instanceof ApiError ? query.error.message : ""
                }
                error={error}
                busy={busy}
                onStart={toggle}
              />
            )}
          </section>
          <AlertFeed alerts={alerts} counts={counts} running={running} />
        </div>

        <p className="fineprint cams-note">
          Демо-режим: камера воспроизводит записи из датасета в реальном
          времени, как поток с настоящей камеры. Реальная камера подключается
          RTSP-адресом в настройках сервера, приём и анализ те же. Каждый
          фрагмент записи уходит модели сразу после окончания, видео вручную не
          загружается.
          {status &&
            ` Трансляция останавливается сама, если её никто не смотрит ${Math.round(status.idle_stop_sec / 60)} мин.`}
        </p>
      </main>
    </Shell>
  );
}

function OnAir({
  status,
  counts,
}: {
  status: CameraStatus;
  counts: ReturnType<typeof cameraCounts>;
}) {
  if (!status.running)
    return (
      <p className="cams-meta">
        {status.stopped_reason === "idle"
          ? "Трансляция остановлена: её никто не смотрел."
          : "Камеры выключены."}
      </p>
    );
  return (
    <p className="cams-meta">
      <span className="onair">
        <i aria-hidden="true" /> В эфире
      </span>
      {status.started_at && (
        <span className="mono">с {clock(status.started_at)}</span>
      )}
      <span>камер: {status.cameras.length}</span>
      <span>проанализировано фрагментов: {counts.analysed}</span>
    </p>
  );
}

function CameraTile({
  camera,
  segments,
  now,
  bind,
  signal,
  flash,
}: {
  camera: Camera;
  segments: Segment[];
  now: number;
  bind: (img: HTMLImageElement | null) => void;
  signal: boolean;
  flash: boolean;
}) {
  const strip = segments.slice(0, STRIP);
  const date = new Date(now * 1000).toLocaleDateString("ru-RU");
  return (
    <figure className={`cam ${flash ? "is-flash" : ""}`}>
      <div className="cam-screen">
        <img ref={bind} alt={`Трансляция: ${camera.name}`} />
        {!signal && (
          <div className="cam-nosignal" role="status">
            {camera.state === "error" ? (
              <>Нет сигнала · переподключение</>
            ) : (
              <>
                <LoaderCircle aria-hidden="true" /> Подключение к камере…
              </>
            )}
          </div>
        )}
        <div className="cam-osd cam-osd-top">
          <span>{camera.name}</span>
          <span className="cam-live">
            <i aria-hidden="true" /> LIVE
          </span>
        </div>
        <div className="cam-osd cam-osd-bottom">
          <span className="mono">
            {date} {clock(now)}
          </span>
          {camera.recording_since && (
            <span className="cam-rec mono">
              REC {timecode(now - camera.recording_since)}
            </span>
          )}
        </div>
      </div>
      <figcaption>
        <span className="sr-only">Последние фрагменты записи</span>
        <ol className="cam-strip">
          {strip.map((s) => {
            const label = segmentLabel(s);
            const body = (
              <>
                <span className="mono">{clock(s.started_at)}</span>
                <span>{label.text}</span>
              </>
            );
            const title = `${clock(s.started_at)} · ${Math.round(s.duration_sec)} с · ${label.text}${s.clip ? ` · запись ${s.clip}` : ""}`;
            return (
              <li key={s.job_id}>
                {s.state === "done" ? (
                  <Link
                    to={`/jobs/${s.job_id}`}
                    className={`cam-seg tone-${label.tone}`}
                    title={title}
                  >
                    {body}
                  </Link>
                ) : (
                  <span className={`cam-seg tone-${label.tone}`} title={title}>
                    {body}
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </figcaption>
    </figure>
  );
}

function CamerasOff({
  status,
  loading,
  loadError,
  error,
  busy,
  onStart,
}: {
  status?: CameraStatus;
  loading: boolean;
  loadError: string;
  error: string;
  busy: boolean;
  onStart: () => void;
}) {
  const disabled = status ? !status.enabled || !status.cameras.length : true;
  return (
    <div className="cams-off">
      <span className="cams-off-icon" aria-hidden="true">
        <Cctv />
      </span>
      <h2>Камеры не подключены</h2>
      <p>
        Сервер принимает поток с камер, режет его на фрагменты и сразу
        отправляет каждый модели. Найденные нарушения появляются справа, с
        кадрами и нормой закона.
      </p>
      {status && status.cameras.length > 0 && (
        <p className="cams-off-list">
          {status.cameras.map((c) => c.name).join(" · ")}
        </p>
      )}
      <button
        className="btn btn-primary"
        onClick={onStart}
        disabled={busy || loading || disabled}
      >
        {busy ? (
          <LoaderCircle aria-hidden="true" className="spin" />
        ) : (
          <Play aria-hidden="true" />
        )}
        Подключить камеры
      </button>
      {status && !status.enabled && (
        <p className="cams-off-error">
          Трансляция камер отключена на этом сервере.
        </p>
      )}
      {(error || loadError) && (
        <p className="cams-off-error" role="alert">
          {error || loadError}
        </p>
      )}
    </div>
  );
}

function AlertFeed({
  alerts,
  counts,
  running,
}: {
  alerts: Alert[];
  counts: ReturnType<typeof cameraCounts>;
  running: boolean;
}) {
  const reduce = useReducedMotion();
  return (
    <aside className="cams-feed" aria-labelledby="feed-title">
      <h2 className="rail-title cams-feed-title" id="feed-title">
        Нарушения <span>{alerts.length}</span>
      </h2>
      {counts.analysed + counts.skipped > 0 && (
        <p className="cams-counts">
          Проанализировано фрагментов: {counts.analysed}
          {counts.skipped > 0 && ` · пропущено: ${counts.skipped}`}
        </p>
      )}
      {alerts.length === 0 ? (
        <p className="cams-empty">
          {running
            ? "Пока нарушений нет. Первый результат появится через несколько секунд после окончания первого фрагмента."
            : "Здесь появятся нарушения, найденные на камерах."}
        </p>
      ) : (
        <ol className="alerts" aria-live="polite">
          <AnimatePresence initial={false}>
            {alerts.map((a) => {
              const delay = detectionDelay(a);
              return (
                <motion.li
                  key={`${a.segment.job_id}-${a.index}`}
                  layout={!reduce}
                  initial={reduce ? false : { opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, ease: [0.2, 0.7, 0.2, 1] }}
                >
                  <Link to={`/jobs/${a.segment.job_id}`} className="alert-card">
                    {a.has_frame ? (
                      <img
                        src={`/app/cameras/evidence/${a.segment.job_id}/${a.index}`}
                        alt=""
                        loading="lazy"
                      />
                    ) : (
                      <span className="alert-noframe" aria-hidden="true" />
                    )}
                    <span className="alert-body">
                      <span className="alert-type">{a.type}</span>
                      <span className="alert-meta">
                        {a.camera} · <span className="mono">{clock(a.at)}</span>
                      </span>
                      {delay !== undefined && (
                        <span className="alert-delay">
                          найдено через {delay} с
                        </span>
                      )}
                    </span>
                    <ArrowUpRight className="alert-go" aria-hidden="true" />
                  </Link>
                </motion.li>
              );
            })}
          </AnimatePresence>
        </ol>
      )}
    </aside>
  );
}

// Cameras whose latest segment just produced a new violation, highlighted for a moment.
function useFlashing(status?: CameraStatus) {
  const seen = useRef<Set<string> | undefined>(undefined);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [flashing, setFlashing] = useState(new Set<string>());
  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => {
    if (!status) return;
    const keys = cameraAlerts(status).map(
      (a) => [`${a.segment.job_id}-${a.index}`, a.segment.camera_id] as const,
    );
    const previous = seen.current;
    seen.current = new Set(keys.map(([k]) => k));
    if (!previous) return;
    const fresh = new Set(
      keys.filter(([k]) => !previous.has(k)).map(([, c]) => c),
    );
    if (!fresh.size) return;
    setFlashing(fresh);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setFlashing(new Set()), 4000);
  }, [status]);
  return flashing;
}
