import { useState } from "react";
import { Check, X } from "lucide-react";
import {
  checkAdded,
  timecode,
  type Added,
  type CatalogType,
} from "./domain";

export function typeName(catalog: CatalogType[], typeId: string) {
  return catalog.find((t) => t.type_id === typeId)?.name ?? typeId;
}

// Right-column form. The video stays usable: the reviewer can scrub and take the
// current frame for the start and the end.
export function AddViolationPanel({
  catalog,
  loadError,
  time,
  duration,
  onSeek,
  onCancel,
  onSave,
}: {
  catalog: CatalogType[];
  loadError: string;
  time: number;
  duration: number;
  onSeek: (t: number) => void;
  onCancel: () => void;
  onSave: (item: Omit<Added, "id">) => Promise<void>;
}) {
  const [typeId, setTypeId] = useState("");
  const [start, setStart] = useState(time);
  const [end, setEnd] = useState<number>();
  const [comment, setComment] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  function edit<T>(set: (v: T) => void) {
    return (v: T) => {
      set(v);
      setError("");
    };
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const problem = checkAdded(
      { type_id: typeId, start, end, comment },
      duration,
    );
    if (problem) return setError(problem);
    setError("");
    setSaving(true);
    try {
      await onSave({
        type_id: typeId,
        t_start_sec: Math.round(start * 1000) / 1000,
        t_end_sec: end === undefined ? null : Math.round(end * 1000) / 1000,
        comment: comment.trim(),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось сохранить.");
      setSaving(false);
    }
  }

  return (
    <article className="finding add-violation" aria-labelledby="add-title">
      <header>
        <p className="overline">Решение проверяющего</p>
        <h2 id="add-title" className="finding-title">
          Добавить нарушение
        </h2>
        <p className="muted">
          Модель пропустила нарушение? Остановите видео на кадре, где оно
          начинается, выберите тип и опишите, что видно. Видео можно
          перематывать, пока форма открыта.
        </p>
      </header>
      <form className="add-form" onSubmit={submit} noValidate>
        <label className="field">
          <span className="field-label">Тип нарушения</span>
          <select
            value={typeId}
            onChange={(e) => edit(setTypeId)(e.target.value)}
            disabled={!catalog.length}
            autoFocus
          >
            <option value="" disabled>
              {catalog.length ? "Выберите тип…" : "Каталог не загрузился"}
            </option>
            {catalog.map((t) => (
              <option key={t.type_id} value={t.type_id}>
                {t.name}
              </option>
            ))}
          </select>
        </label>
        <div className="field-row">
          <div className="field">
            <span className="field-label">Начало</span>
            <div className="time-pick">
              <button
                type="button"
                className="time-value mono"
                onClick={() => onSeek(start)}
                title="Показать этот момент в видео"
              >
                {timecode(start, true)}
              </button>
              <button
                type="button"
                className="btn btn-quiet btn-sm"
                onClick={() => edit(setStart)(time)}
                title="Взять время, на котором сейчас стоит видео"
              >
                Текущий кадр · <span className="mono">{timecode(time)}</span>
              </button>
            </div>
          </div>
          <div className="field">
            <span className="field-label">
              Конец <span className="field-hint">необязательно</span>
            </span>
            <div className="time-pick">
              {end === undefined ? (
                <span className="time-value is-empty mono">—</span>
              ) : (
                <>
                  <button
                    type="button"
                    className="time-value mono"
                    onClick={() => onSeek(end)}
                    title="Показать этот момент в видео"
                  >
                    {timecode(end, true)}
                  </button>
                  <button
                    type="button"
                    className="icon-btn icon-btn-sm"
                    onClick={() => setEnd(undefined)}
                    aria-label="Убрать конец"
                  >
                    <X />
                  </button>
                </>
              )}
              <button
                type="button"
                className="btn btn-quiet btn-sm"
                onClick={() => edit(setEnd)(time)}
                title="Взять время, на котором сейчас стоит видео"
              >
                Текущий кадр · <span className="mono">{timecode(time)}</span>
              </button>
            </div>
          </div>
        </div>
        <label className="field">
          <span className="field-label">Что происходит</span>
          <textarea
            rows={4}
            maxLength={2000}
            value={comment}
            onChange={(e) => edit(setComment)(e.target.value)}
            placeholder="Например: кассир вышел из кассы, пачка купюр осталась на столе"
          />
        </label>
        {(error || loadError) && (
          <p className="form-error" role="alert">
            {error || loadError}
          </p>
        )}
        <div className="row">
          <button type="submit" className="btn btn-primary" disabled={saving}>
            <Check aria-hidden="true" />
            {saving ? "Сохраняем…" : "Сохранить нарушение"}
          </button>
          <button type="button" className="btn btn-quiet" onClick={onCancel}>
            Отмена
          </button>
        </div>
        <p className="fineprint">
          Запись хранится на сервере для разбора качества. Модель от неё не
          меняется.
        </p>
      </form>
    </article>
  );
}

// Left-rail list of what the reviewer added, shown even when the model found nothing.
export function AddedList({
  added,
  catalog,
  onSeek,
  onRemove,
}: {
  added: Added[];
  catalog: CatalogType[];
  onSeek: (t: number) => void;
  onRemove: (id: string) => Promise<void>;
}) {
  const [confirming, setConfirming] = useState<string>();
  if (!added.length) return null;
  return (
    <nav className="log log-added" aria-label="Добавлено проверяющим">
      <h2 className="rail-title">
        Добавлено проверяющим <span>{added.length}</span>
      </h2>
      <ol className="index">
        {added.map((a) => {
          const name = typeName(catalog, a.type_id);
          return (
            <li key={a.id} className="added-item">
              <button className="index-item" onClick={() => onSeek(a.t_start_sec)}>
                <span className="index-body">
                  <span className="index-type">{name}</span>
                  <span className="index-meta">
                    <span className="mono">
                      {timecode(a.t_start_sec)}
                      {a.t_end_sec !== null && `–${timecode(a.t_end_sec)}`}
                    </span>
                    <span className="added-badge">проверяющий</span>
                  </span>
                  <span className="added-comment">{a.comment}</span>
                </span>
              </button>
              {confirming === a.id ? (
                <span className="added-confirm">
                  <button
                    className="btn btn-sm btn-danger"
                    onClick={() => onRemove(a.id).finally(() => setConfirming(undefined))}
                  >
                    Удалить
                  </button>
                  <button
                    className="btn btn-sm btn-quiet"
                    onClick={() => setConfirming(undefined)}
                  >
                    Оставить
                  </button>
                </span>
              ) : (
                <button
                  className="icon-btn icon-btn-sm added-remove"
                  aria-label={`Удалить: ${name}, ${timecode(a.t_start_sec)}`}
                  onClick={() => setConfirming(a.id)}
                >
                  <X />
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
