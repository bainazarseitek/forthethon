import { useEffect, useMemo } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Printer } from "lucide-react";
import {
  confidenceLevel,
  confidencePercent,
  episodeRange,
  eventKey,
  safeSourceUrl,
  statusLabels,
  story,
  summary,
  timecode,
  type CompletedJob,
} from "./domain";
import { useReviews } from "./useReviews";
import { Evidence } from "./Evidence";
import { Shell, caseNumber } from "./Shell";
import { episodes, verdictText } from "./Review";

export function Report({ job }: { job: CompletedJob }) {
  const events = job.violations;
  const { snapshot } = useReviews(job, false);
  const decisions = snapshot.decisions;
  const counts = summary(events, decisions);
  const created = useMemo(
    () =>
      new Intl.DateTimeFormat("ru-RU", {
        dateStyle: "long",
        timeStyle: "short",
      }).format(new Date()),
    [],
  );
  const caseNo = caseNumber(job.job_id);
  const isViolation = job.verdict === "violation";
  useEffect(() => {
    document.title = `Досье ${caseNo} · ${job.filename ?? "видео"}`;
  }, [caseNo, job.filename]);

  return (
    <Shell
      trail={<span className="trail-case">Досье {caseNo}</span>}
      actions={
        <>
          <Link
            className="btn btn-quiet"
            to={`/jobs/${job.job_id}`}
            aria-label="К видео"
          >
            <ArrowLeft aria-hidden="true" />
            <span className="hide-sm">К видео</span>
          </Link>
          <button className="btn btn-primary" onClick={() => window.print()}>
            <Printer aria-hidden="true" />
            Печать / PDF
          </button>
        </>
      }
    >
      <main className="dossier" id="main">
        <section className="sheet cover">
          <div className="cover-top">
            <span className="brand-word">
              forte <span>review</span>
            </span>
            <span className="mono">ДЕЛО {caseNo}</span>
          </div>
          <p className="overline">Досье проверки видеозаписи кассы</p>
          <h1
            className={`cover-verdict ${isViolation ? "is-alert" : "is-clean"}`}
          >
            {verdictText[job.verdict]}
          </h1>
          <dl className="facts">
            <div>
              <dt>Файл</dt>
              <dd>{job.filename ?? "—"}</dd>
            </div>
            <div>
              <dt>Длительность</dt>
              <dd className="mono">
                {job.duration_sec !== undefined
                  ? timecode(job.duration_sec, true)
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>Эпизоды</dt>
              <dd>{events.length ? episodes(events.length) : "нет"}</dd>
            </div>
            <div>
              <dt>Отчёт сформирован</dt>
              <dd>{created}</dd>
            </div>
            {events.length > 0 && (
              <div>
                <dt>Решения проверяющего</dt>
                <dd>
                  подтверждено {counts.accepted}, отклонено {counts.rejected},
                  нужна проверка {counts.uncertain}, не рассмотрено{" "}
                  {counts.unreviewed}
                </dd>
              </div>
            )}
          </dl>

          {events.length > 0 ? (
            <table className="toc">
              <caption>Оглавление</caption>
              <thead>
                <tr>
                  <th scope="col">№</th>
                  <th scope="col">Эпизод</th>
                  <th scope="col">Время</th>
                  <th scope="col">Уверенность</th>
                  <th scope="col">Норма</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e, i) => (
                  <tr key={eventKey(e, i)}>
                    <td className="mono">{String(i + 1).padStart(2, "0")}</td>
                    <td>
                      <a href={`#finding-${i + 1}`}>{e.type}</a>
                    </td>
                    <td className="mono">{timecode(e.timecode_sec, true)}</td>
                    <td>
                      {e.confidence !== undefined
                        ? confidencePercent(e.confidence)
                        : "—"}
                    </td>
                    <td>{e.rule?.clause ?? (e.rule ? "см. документ" : "—")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="cover-clean">
              Модель просмотрела всю запись и не нашла эпизодов с признаками
              нарушения порядка кассовых операций.
            </p>
          )}

          <div className="cover-foot">
            <p className="disclaimer">
              Результат сформирован моделью машинного зрения и является
              подсказкой для проверяющего. Он не является юридическим
              заключением и не устанавливает вину сотрудника.
            </p>
            <div className="signature">
              <span>Проверяющий</span>
              <span>Подпись</span>
              <span>Дата</span>
            </div>
          </div>
        </section>

        {events.map((e, i) => {
          const s = story(e);
          const r = episodeRange(e, job.duration_sec ?? 0);
          const url = safeSourceUrl(e.rule?.url);
          const d = decisions[eventKey(e, i)]?.status ?? "unreviewed";
          return (
            <section
              className="sheet finding-sheet"
              key={eventKey(e, i)}
              id={`finding-${i + 1}`}
            >
              <header className="sheet-head">
                <span className="sheet-n">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div>
                  <p className="overline">
                    Эпизод {i + 1} из {events.length} ·{" "}
                    <span className="mono">
                      {timecode(e.timecode_sec, true)}
                    </span>{" "}
                    · интервал {r.start.toFixed(1)}–{r.end.toFixed(1)} с
                  </p>
                  <h2>{e.type}</h2>
                </div>
              </header>
              <dl className="facts facts-row">
                {e.confidence !== undefined && (
                  <div>
                    <dt>Уверенность модели</dt>
                    <dd>
                      {confidencePercent(e.confidence)} ·{" "}
                      {confidenceLevel(e.confidence)}
                    </dd>
                  </div>
                )}
                <div>
                  <dt>Решение проверяющего</dt>
                  <dd className={`tag tag-${d}`}>{statusLabels[d]}</dd>
                </div>
              </dl>
              <ol className="story story-print">
                <li>
                  <span className="story-n">1</span>
                  <div>
                    <h3>Что видно на записи</h3>
                    <p>{s.seen}</p>
                  </div>
                </li>
                {s.why && (
                  <li>
                    <span className="story-n">2</span>
                    <div>
                      <h3>Почему это может быть нарушением</h3>
                      <p>{s.why}</p>
                    </div>
                  </li>
                )}
                <li>
                  <span className="story-n">{s.why ? 3 : 2}</span>
                  <div>
                    <h3>Норма</h3>
                    {e.rule ? (
                      <div className="citation">
                        {e.rule.clause && (
                          <p className="citation-clause">{e.rule.clause}</p>
                        )}
                        <p className="citation-doc">{e.rule.document}</p>
                        {url && (
                          <a className="citation-url" href={url}>
                            {url}
                          </a>
                        )}
                      </div>
                    ) : (
                      <p>Норма не указана моделью.</p>
                    )}
                  </div>
                </li>
              </ol>
              {e.evidence_frames.length > 0 && (
                <div className={`sheet-frames n${e.evidence_frames.length}`}>
                  {e.evidence_frames.map((f, n) => (
                    <figure key={n}>
                      <Evidence
                        frame={f}
                        alt={`Кадр ${n + 1}, ${timecode(f.timestamp_sec, true)}`}
                      />
                      <figcaption>
                        Кадр {n + 1} ·{" "}
                        <span className="mono">
                          {timecode(f.timestamp_sec, true)}
                        </span>
                      </figcaption>
                    </figure>
                  ))}
                </div>
              )}
              <p className="sheet-foot">
                Дело {caseNo} · {job.filename} · подсказка модели, не
                юридическое заключение
              </p>
            </section>
          );
        })}
      </main>
    </Shell>
  );
}
