// Poll a single job until it reaches a terminal state.
//
// Usage:
//   const { job, error } = useJobStatus(jobId, { intervalMs: 2000 });
//   when job?.status === "complete" → workflow done; if it produced a
//   target (e.g. scene_<uuid>), read it from job.target_id.
//
// Stops polling automatically once status ∈ {complete, failed, cancelled}.
// Pass `null` as jobId to disable polling (useful while a parent component
// is still kicking off the job and doesn't have an id yet).

import { useEffect, useState } from "react";

import { ApiError } from "../api/client";
import { getJob } from "../api/jobs";
import type { Job, JobStatus } from "../types";

interface UseJobStatusOptions {
  /** Poll interval in ms (default 2000). */
  intervalMs?: number;
}

interface UseJobStatusResult {
  job: Job | null;
  error: string | null;
  /** True until the first response (success or error) lands. */
  loading: boolean;
}

const TERMINAL: ReadonlySet<JobStatus> = new Set([
  "complete",
  "failed",
  "cancelled",
]);

export function useJobStatus(
  jobId: string | null,
  opts: UseJobStatusOptions = {},
): UseJobStatusResult {
  const intervalMs = opts.intervalMs ?? 2000;
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(jobId !== null);

  useEffect(() => {
    if (jobId === null) {
      setJob(null);
      setError(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    setLoading(true);
    setError(null);

    const tick = async () => {
      let isTerminal = false;
      try {
        const next = await getJob(jobId);
        if (cancelled) return;
        setJob(next);
        setError(null);
        isTerminal = TERMINAL.has(next.status);
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError) {
          setError(err.detail ?? `Error: HTTP ${err.status}`);
        } else {
          setError("Could not reach the server.");
        }
        // Transient errors don't stop polling — next tick may recover.
      } finally {
        if (!cancelled) {
          setLoading(false);
          if (!isTerminal) {
            timer = setTimeout(tick, intervalMs);
          }
        }
      }
    };

    tick();

    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [jobId, intervalMs]);

  return { job, error, loading };
}
