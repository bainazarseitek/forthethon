import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getCatalog, getReview, putReview } from "./api";
import {
  sortCatalog,
  type Added,
  type CatalogType,
  type CompletedJob,
  type Decision,
} from "./domain";

// Violations the reviewer adds are stored on the server with the job. The reviewer's
// decisions on model events (kept in the browser by useReviews) are sent along, so the
// server holds the whole review. Nothing here changes the model.
export function useAddedViolations(
  job: CompletedJob,
  decisions: Record<string, Decision>,
) {
  const [added, setAdded] = useState<Added[]>([]);
  const [catalog, setCatalog] = useState<CatalogType[]>([]);
  const [loadError, setLoadError] = useState("");
  const version = useRef(0);
  const current = useRef<Added[]>([]);

  useEffect(() => {
    let live = true;
    getReview(job.job_id)
      .then((r) => {
        if (!live) return;
        version.current = r.version;
        current.current = r.added;
        setAdded(r.added);
      })
      .catch((e: Error) => live && setLoadError(e.message));
    getCatalog()
      .then((c) => live && setCatalog(sortCatalog(c.types)))
      .catch((e: Error) => live && setLoadError(e.message));
    return () => {
      live = false;
    };
  }, [job.job_id]);

  const save = useCallback(
    async (next: (list: Added[]) => Added[]) => {
      const attempt = async () => {
        const r = await putReview(job.job_id, {
          version: version.current,
          decisions,
          added: next(current.current),
        });
        version.current = r.version;
        current.current = r.added;
        setAdded(r.added);
      };
      try {
        await attempt();
      } catch (e) {
        if (!(e instanceof ApiError) || e.status !== 409) throw e;
        // Someone saved in between: re-apply this change on top of their version once.
        const r = await getReview(job.job_id);
        version.current = r.version;
        current.current = r.added;
        await attempt();
      }
    },
    [job.job_id, decisions],
  );

  const add = useCallback(
    (item: Omit<Added, "id">) =>
      save((list) => [
        ...list,
        {
          ...item,
          id: `r${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
        },
      ]),
    [save],
  );
  const remove = useCallback(
    (id: string) => save((list) => list.filter((a) => a.id !== id)),
    [save],
  );
  return { added, catalog, loadError, add, remove };
}
