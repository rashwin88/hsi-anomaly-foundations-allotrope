// Minimal class-component error boundary so a render exception in one
// pane (Action card, Output viewer, …) doesn't blank the entire page.
//
// Drop in around panes that pull in unfamiliar / drifting data shapes:
//   <ErrorBoundary fallback={<p>This pane failed to render.</p>}>
//     <ActionDetailPane … />
//   </ErrorBoundary>

import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (err: Error, info: ErrorInfo) => void;
}

interface State {
  err: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { err: null };

  static getDerivedStateFromError(err: Error): State {
    return { err };
  }

  componentDidCatch(err: Error, info: ErrorInfo): void {
    // Surface in the dev console; keep the UI alive.
    // eslint-disable-next-line no-console
    console.error("[ErrorBoundary]", err, info);
    this.props.onError?.(err, info);
  }

  render(): ReactNode {
    if (this.state.err) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="error-boundary">
          <strong>Something broke in this pane.</strong>
          <p>{this.state.err.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}
