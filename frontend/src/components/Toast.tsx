// Lightweight toast notification system.
//
// One global container (mounted near <App/>) renders a stack of toasts
// in the bottom-right. Components anywhere in the tree fire toasts with
// `useToast().push({...})`.
//
// A toast can carry a `link` (route + label) so it stays useful even
// after the user navigates away from the page that fired it. Used today
// for "Onboarding job started → View in Jobs".
//
// Variant maps to the same colour family as the JobStatus pill — green
// for success, amber for in-flight, red for failure.

import {
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Link } from "react-router-dom";

export type ToastVariant = "info" | "success" | "warning" | "error";

export interface ToastInput {
  title: string;
  message?: string;
  variant?: ToastVariant;
  /** If set, renders a router Link inside the toast. */
  link?: { to: string; label: string };
  /**
   * Auto-dismiss after this many ms. Default 6000. Pass 0 (or undefined
   * after explicit cast) for sticky toasts.
   */
  durationMs?: number;
}

interface ToastEntry extends ToastInput {
  id: string;
}

interface ToastContextValue {
  push: (toast: ToastInput) => string;
  dismiss: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used inside <ToastProvider>");
  }
  return ctx;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastEntry[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (input: ToastInput) => {
      const id = `t_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      setToasts((prev) => [...prev, { ...input, id }]);
      return id;
    },
    [],
  );

  const value = useMemo(() => ({ push, dismiss }), [push, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastViewport toasts={toasts} dismiss={dismiss} />
    </ToastContext.Provider>
  );
}

function ToastViewport({
  toasts,
  dismiss,
}: {
  toasts: ToastEntry[];
  dismiss: (id: string) => void;
}) {
  if (toasts.length === 0) return null;
  return (
    <div className="toast-viewport" role="region" aria-label="Notifications">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={() => dismiss(t.id)} />
      ))}
    </div>
  );
}

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: ToastEntry;
  onDismiss: () => void;
}) {
  const duration = toast.durationMs ?? 6000;

  useEffect(() => {
    if (duration <= 0) return;
    const handle = setTimeout(onDismiss, duration);
    return () => clearTimeout(handle);
  }, [duration, onDismiss]);

  const variant = toast.variant ?? "info";

  return (
    <div className={`toast toast--${variant}`} role="status" aria-live="polite">
      <div className="toast__body">
        <p className="toast__title">{toast.title}</p>
        {toast.message && <p className="toast__message">{toast.message}</p>}
        {toast.link && (
          <Link
            to={toast.link.to}
            className="toast__link"
            onClick={onDismiss}
          >
            {toast.link.label} →
          </Link>
        )}
      </div>
      <button
        type="button"
        className="toast__close"
        onClick={onDismiss}
        aria-label="Dismiss notification"
      >
        ×
      </button>
    </div>
  );
}
