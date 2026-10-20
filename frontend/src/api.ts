import { z } from "zod";
import {
  catalogSchema,
  jobSchema,
  serverReviewSchema,
  type Added,
  type Decision,
} from "./domain";

const BASE = "/app";
export class ApiError extends Error {
  constructor(
    message: string,
    public status = 0,
  ) {
    super(message);
  }
}
export const maxBytes = (Number(import.meta.env.VITE_MAX_UPLOAD_MB) || 2048) * 1048576;
const errorMessages: Record<string, string> = {
  wrong_extension:
    "Этот формат не поддерживается. Выберите файл .mov или .mp4.",
  empty_file: "Файл пуст. Выберите видео с записью.",
  invalid_video:
    "Сервер не смог прочитать видео. Проверьте, что файл не повреждён.",
  file_too_large: `Файл слишком большой. Максимум ${Math.round(maxBytes / 1048576)} МБ.`,
  file_missing: "Файл не был передан. Выберите видео ещё раз.",
  queue_full:
    "Сейчас в очереди уже несколько видео. Повторите загрузку через пару минут.",
  uploads_disabled:
    "Загрузка видео временно отключена администратором. Ранее загруженные проверки доступны по ссылке.",
  job_not_found:
    "Проверка не найдена. Результаты хранятся 24 часа, после этого видео нужно загрузить заново.",
};
export const failureMessages: Record<string, string> = {
  timeout:
    "Модель не успела обработать видео за отведённое время и была остановлена.",
  deadline: "Проверка не уложилась в лимит времени после загрузки.",
  crashed: "Сервер дважды перезапускался во время обработки этого видео.",
  detector_not_ready: "Модель ещё загружалась. Загрузите видео повторно.",
  detector_failed: "Модель не смогла обработать это видео.",
  result_invalid: "Модель вернула результат, который не удалось сохранить.",
  processing_failed: "Во время обработки произошла ошибка.",
};
function statusMessage(status: number, code?: string) {
  if (code && errorMessages[code]) return errorMessages[code];
  if (status === 400) return "Сервер не принял файл. Проверьте формат видео.";
  if (status === 404) return errorMessages.job_not_found;
  if (status === 413) return errorMessages.file_too_large;
  if (status === 422) return errorMessages.file_missing;
  if (status === 429) return errorMessages.queue_full;
  if (status === 503) return errorMessages.uploads_disabled;
  return "Сервер анализа недоступен. Проверьте подключение и повторите попытку.";
}
function errorCode(text: string) {
  try {
    const code = (JSON.parse(text) as { error?: unknown }).error;
    return typeof code === "string" ? code : undefined;
  } catch {
    return undefined;
  }
}
export function parsePayload<T>(raw: unknown, schema: z.ZodType<T>): T {
  const result = schema.safeParse(raw);
  if (!result.success)
    throw new ApiError(
      "Ответ сервера не соответствует контракту. Результат нельзя считать корректным.",
    );
  return result.data;
}
export async function getJob(id: string, signal?: AbortSignal) {
  let response: Response;
  try {
    response = await fetch(`${BASE}/jobs/${encodeURIComponent(id)}`, {
      cache: "no-store",
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(15_000)])
        : AbortSignal.timeout(15_000),
      headers: { Accept: "application/json" },
    });
  } catch {
    throw new ApiError(
      "Не удалось связаться с сервером. Проверьте соединение и повторите запрос.",
    );
  }
  const text = await response.text().catch(() => "");
  if (!response.ok)
    throw new ApiError(
      statusMessage(response.status, errorCode(text)),
      response.status,
    );
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    throw new ApiError("Сервер вернул нечитаемый ответ.");
  }
  const job = parsePayload(body, jobSchema);
  if (job.job_id !== id)
    throw new ApiError("Сервер вернул результат другой проверки.");
  return job;
}
export function uploadVideo(
  file: File,
  onProgress: (progress: number) => void,
) {
  return new Promise<{ job_id: string }>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/analyze`);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.timeout = 300_000;
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable)
        onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onerror = () =>
      reject(
        new ApiError("Загрузка прервана. Соединение с сервером недоступно."),
      );
    xhr.ontimeout = () =>
      reject(
        new ApiError(
          "Сервер не подтвердил загрузку за пять минут. Попробуйте загрузить видео ещё раз.",
        ),
      );
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(
          new ApiError(
            statusMessage(xhr.status, errorCode(xhr.responseText)),
            xhr.status,
          ),
        );
        return;
      }
      try {
        resolve(
          parsePayload(
            JSON.parse(xhr.responseText),
            z.object({ job_id: z.string().min(1) }),
          ),
        );
      } catch {
        reject(
          new ApiError(
            "Сервер вернул некорректный ответ. Видео не получило номер проверки.",
          ),
        );
      }
    };
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

async function json<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
      },
    });
  } catch {
    throw new ApiError(
      "Не удалось связаться с сервером. Проверьте соединение и повторите.",
    );
  }
  const text = await response.text().catch(() => "");
  if (!response.ok) {
    const code = errorCode(text);
    const message =
      code === "review_conflict"
        ? "Кто-то уже изменил эту проверку. Обновили данные, повторите действие."
        : code === "invalid_review"
          ? "Сервер не принял данные: проверьте тип, время и описание."
          : statusMessage(response.status, code);
    throw new ApiError(message, response.status);
  }
  try {
    return parsePayload(JSON.parse(text), schema);
  } catch (e) {
    throw e instanceof ApiError
      ? e
      : new ApiError("Сервер вернул нечитаемый ответ.");
  }
}
export function getCatalog() {
  return json("/catalog", catalogSchema);
}
export function getReview(jobId: string) {
  return json(`/jobs/${encodeURIComponent(jobId)}/review`, serverReviewSchema);
}
export function putReview(
  jobId: string,
  body: { version: number; decisions: Record<string, Decision>; added: Added[] },
) {
  return json(`/jobs/${encodeURIComponent(jobId)}/review`, serverReviewSchema, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}
