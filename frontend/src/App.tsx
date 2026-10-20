import { useEffect, useRef, useState, type DragEvent } from "react";
import { Link, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import {
  ArrowLeft,
  Cctv,
  Check,
  CircleAlert,
  FileVideo,
  LoaderCircle,
  Plus,
  RotateCcw,
  Upload,
} from "lucide-react";
import { ApiError, failureMessages, getJob, maxBytes, uploadVideo } from "./api";
import { safeMediaUrl, timecode, validateFile, type Job } from "./domain";
import { CamerasPage } from "./Cameras";
import { Review } from "./Review";
import { useFilmstrip } from "./useFilmstrip";
import { Report } from "./Report";
import { Shell, caseNumber } from "./Shell";


export function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/jobs/:jobId" element={<JobPage />} />
      <Route path="/jobs/:jobId/report" element={<JobPage report />} />
      <Route path="/cameras" element={<CamerasPage />} />
      <Route
        path="*"
        element={
          <Shell>
            <main className="state" id="main">
              <h1 className="display-m">Страница не найдена</h1>
              <Link className="btn btn-primary" to="/">
                К загрузке видео
              </Link>
            </main>
          </Shell>
        }
      />
    </Routes>
  );
}

function UploadPage() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File>();
  const [progress, setProgress] = useState<number>();
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  useEffect(() => {
    document.title = "Forte Review · проверка кассовых операций";
  }, []);

  async function start(next: File) {
    const invalid = validateFile(next, maxBytes);
    setFile(next);
    setError(invalid ?? "");
    if (invalid) return;
    setProgress(0);
    try {
      const { job_id } = await uploadVideo(next, setProgress);
      navigate(`/jobs/${job_id}`);
    } catch (e) {
      setProgress(undefined);
      setError(
        e instanceof ApiError ? e.message : "Не удалось загрузить видео.",
      );
    }
  }
  function drop(e: DragEvent) {
    e.preventDefault();
    setDrag(false);
    const next = e.dataTransfer.files[0];
    if (next && progress === undefined) start(next);
  }
  const busy = progress !== undefined;

  return (
    <Shell
      actions={
        <Link className="btn btn-quiet" to="/cameras">
          <Cctv aria-hidden="true" /> Камеры
        </Link>
      }
    >
      <main className="upload" id="main">
        <motion.section
          className="upload-intro"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.2, 0.7, 0.2, 1] }}
        >
          <p className="overline">Контроль кассовой дисциплины</p>
          <h1 className="display-xl">
            Видео кассы <em>в досье</em> за пару минут
          </h1>
          <p className="lede">
            Загрузите запись с потолочной камеры. Модель отметит моменты,
            похожие на нарушение порядка кассовых операций, и приложит кадры и
            норму закона. Решение остаётся за вами.
          </p>
        </motion.section>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.08, ease: [0.2, 0.7, 0.2, 1] }}
        >
          <label
            className={`drop ${drag ? "is-drag" : ""} ${busy ? "is-busy" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              if (!busy) setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={drop}
          >
            <input
              ref={input}
              className="sr-only"
              type="file"
              accept=".mov,.mp4,video/quicktime,video/mp4"
              aria-label="Видео для проверки"
              aria-describedby="drop-hint"
              disabled={busy}
              onChange={(e) => {
                const next = e.target.files?.[0];
                e.target.value = "";
                if (next) start(next);
              }}
            />
            {busy ? (
              <div className="drop-progress" role="status">
                <FileVideo aria-hidden="true" />
                <div className="drop-file">
                  <strong>{file?.name}</strong>
                  <span>
                    {progress < 100
                      ? `Загружаем на сервер · ${progress}%`
                      : "Файл получен, открываем дело…"}
                  </span>
                  <div
                    className="bar"
                    role="progressbar"
                    aria-label="Загрузка файла"
                    aria-valuenow={progress}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  >
                    <span style={{ width: `${progress}%` }} />
                  </div>
                </div>
              </div>
            ) : (
              <>
                <span className="drop-icon" aria-hidden="true">
                  <Upload />
                </span>
                <span className="drop-title">
                  <span className="hide-sm">
                    Перетащите видео сюда или <u>выберите файл</u>
                  </span>
                  <span className="show-sm">Выберите видео</span>
                </span>
                <span className="drop-hint" id="drop-hint">
                  .mov или .mp4, до {Math.round(maxBytes / 1048576)} МБ. Обычно
                  анализ занимает 1–2 минуты.
                </span>
              </>
            )}
          </label>
          {error && (
            <p className="alert" role="alert">
              <CircleAlert aria-hidden="true" />
              <span>
                {error}
                {file && <small>Файл: {file.name}</small>}
              </span>
            </p>
          )}
        </motion.div>

        <p className="upload-cams">
          Без ручной загрузки:{" "}
          <Link to="/cameras">камеры кассовой зоны в реальном времени</Link>
        </p>

        <ol className="how">
          {[
            [
              "Загрузка",
              "Видео обрабатывается локально, без облачных сервисов, и удаляется через 24 часа.",
            ],
            [
              "Анализ",
              "Модель просматривает запись и выделяет эпизоды с признаками нарушения.",
            ],
            [
              "Досье",
              "Каждый эпизод: таймкод, что видно, какая норма может быть нарушена, кадры.",
            ],
          ].map(([title, text], i) => (
            <li key={title}>
              <span className="how-n">0{i + 1}</span>
              <strong>{title}</strong>
              <p>{text}</p>
            </li>
          ))}
        </ol>
        <p className="fineprint">
          Результат является подсказкой для проверяющего и не является
          юридическим заключением.
        </p>
      </main>
    </Shell>
  );
}

function JobPage({ report = false }: { report?: boolean }) {
  const { jobId = "" } = useParams();
  const query = useQuery({
    queryKey: ["job", jobId],
    queryFn: ({ signal }) => getJob(jobId, signal),
    refetchInterval: (q) =>
      q.state.data?.status === "processing" ? 1500 : false,
    retry: (count, e) =>
      !(e instanceof ApiError && e.status === 404) && count < 1,
    retryDelay: 500,
  });
  const job = query.data;
  const notFound =
    query.error instanceof ApiError && query.error.status === 404;
  if (job?.status === "done")
    return report ? <Report job={job} /> : <Review job={job} />;
  return (
    <Shell trail={<span className="trail-case">Дело {caseNumber(jobId)}</span>}>
      <main className="state" id="main">
        {query.isError ? (
          <Failure
            title="Не удалось открыть проверку"
            text={
              query.error instanceof ApiError
                ? query.error.message
                : "Сервер анализа недоступен."
            }
            retry={notFound ? undefined : () => query.refetch()}
            newCheck={notFound}
          />
        ) : job?.status === "failed" ? (
          <Failure
            title="Видео не удалось проанализировать"
            text={
              (job.error && failureMessages[job.error]) ||
              job.message ||
              failureMessages.processing_failed
            }
          />
        ) : job ? (
          <Processing job={job} />
        ) : (
          <p className="loading" role="status">
            <LoaderCircle aria-hidden="true" /> Открываем дело…
          </p>
        )}
      </main>
    </Shell>
  );
}

const stages = [
  { label: "Видео получено", until: 0 },
  { label: "Ожидание в очереди", until: 0.05 },
  { label: "Анализ записи моделью", until: 0.8 },
  { label: "Подготовка кадров-доказательств", until: 0.95 },
  { label: "Сборка досье", until: 1 },
];

const STRIP = 10;

function Processing({ job }: { job: Job }) {
  const thumbs = useFilmstrip(
    STRIP,
    job.status === "processing"
      ? safeMediaUrl(job.video_url, window.location.origin)
      : undefined,
  );
  const [started] = useState(() => Date.now());
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    document.title = "Анализ видео… · Forte Review";
  }, []);
  const progress = job.progress ?? 0;
  const queued = (job.queue_position ?? 0) > 0;
  const current = queued
    ? 1
    : stages.findIndex((s, i) => i > 0 && progress < s.until);
  const active = current === -1 ? stages.length - 1 : current;
  const elapsed = Math.floor((Date.now() - started) / 1000);
  return (
    <motion.section
      className="processing"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <p className="overline">Идёт анализ</p>
      <h1 className="display-m">{job.filename ?? "Видео"}</h1>
      <p className="processing-meta">
        {job.duration_sec !== undefined &&
          `Длительность ${timecode(job.duration_sec)} · `}
        прошло {timecode(elapsed)}
      </p>
      <div className="scan-strip" aria-hidden="true">
        {Array.from({ length: STRIP }, (_, i) => (
          <span key={i} className={thumbs[i] ? "is-ready" : ""}>
            {thumbs[i] && <img src={thumbs[i]} alt="" />}
          </span>
        ))}
        {!queued && <i className="scan-beam" />}
      </div>
      <div
        className="bar bar-lg"
        role="progressbar"
        aria-label="Ход анализа"
        aria-valuenow={Math.round(progress * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <span style={{ width: `${Math.max(progress * 100, 3)}%` }} />
      </div>
      <ol className="stages" aria-live="polite">
        {stages.map((s, i) => (
          <li
            key={s.label}
            className={i < active ? "done" : i === active ? "now" : ""}
            aria-current={i === active ? "step" : undefined}
          >
            <span className="stage-dot" aria-hidden="true">
              {i < active && <Check />}
            </span>
            <span>
              {s.label}
              {i === 1 && queued && (
                <small> · перед вами {job.queue_position}</small>
              )}
            </span>
          </li>
        ))}
      </ol>
      <p className="fineprint">
        Страницу можно закрыть: ссылка на дело действует 24 часа.
      </p>
    </motion.section>
  );
}

function Failure({
  title,
  text,
  retry,
  newCheck = false,
}: {
  title: string;
  text: string;
  retry?: () => void;
  newCheck?: boolean;
}) {
  useEffect(() => {
    document.title = `${title} · Forte Review`;
  }, [title]);
  return (
    <section className="failure" role="alert">
      <CircleAlert className="failure-icon" aria-hidden="true" />
      <h1 className="display-m">{title}</h1>
      <p>{text}</p>
      <div className="row">
        <Link className="btn btn-primary" to="/">
          {newCheck ? (
            <>
              <Plus aria-hidden="true" /> Новая проверка
            </>
          ) : (
            <>
              <ArrowLeft aria-hidden="true" /> Загрузить видео
            </>
          )}
        </Link>
        {retry && (
          <button className="btn" onClick={retry}>
            <RotateCcw aria-hidden="true" /> Повторить запрос
          </button>
        )}
      </div>
    </section>
  );
}
