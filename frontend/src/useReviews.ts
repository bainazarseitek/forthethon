import { useState } from "react";
import {
  reviewSchema,
  type CompletedJob,
  type Decision,
  type Review,
} from "./domain";

export function useReviews(job: CompletedJob, demo: boolean) {
  const area = demo ? "sessionStorage" : "localStorage";
  const storageKey = `forte-review:${job.job_id}`;
  const [snapshot, setSnapshot] = useState<Review>(() => {
    try {
      const saved = reviewSchema.safeParse(
        JSON.parse(window[area].getItem(storageKey) ?? "null"),
      );
      if (saved.success) return saved.data;
    } catch {
      return { version: 0, decisions: {} };
    }
    return { version: 0, decisions: {} };
  });
  const [notice, setNotice] = useState("");
  function change(id: string, decision: Decision) {
    const next = {
      version: snapshot.version + 1,
      decisions: { ...snapshot.decisions, [id]: decision },
    };
    setSnapshot(next);
    try {
      window[area].setItem(storageKey, JSON.stringify(next));
      setNotice(
        demo
          ? "Учебный черновик сохранён в этой вкладке"
          : "Решение сохранено в этом браузере",
      );
    } catch {
      setNotice("Черновик только в памяти; после обновления будет потерян");
    }
  }
  return { snapshot, change, notice };
}
