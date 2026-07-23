// Login page (Step 3c).
//
// Aesthetic per storyboard-spec § 3 (locked):
//   "branded utility — minimal form (username, password, button) +
//    product mark + version + one restrained domain element"
//
// The "domain element" is the thin spectral strip across the top of the
// card — a faint gradient suggesting hyperspectral wavelengths. No other
// ornamentation; the form is the form.

import { type FormEvent, useState } from "react";
import { type Location, useLocation, useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { useAuth } from "../auth/useAuth";
import { ThemeToggle } from "../components/ThemeToggle";

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  // If the user was redirected here from a deep link (e.g. /scenes), take
  // them back there after sign-in.
  const from =
    (location.state as { from?: Location } | null)?.from?.pathname ?? "/";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(username, password);
      // Auth state flips → Router swaps to authenticated routes. We also
      // explicitly navigate to the intended destination (or "/" by default),
      // replacing /login so back button doesn't bounce back here.
      navigate(from, { replace: true });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError("Invalid username or password.");
      } else {
        setError("Could not reach the server. Try again.");
      }
      setSubmitting(false);
    }
  };

  const canSubmit = !submitting && username.length > 0 && password.length > 0;

  return (
    <div className="login">
      <ThemeToggle floating />
      <div className="login__card">
        <div className="login__strip" aria-hidden="true" />

        <header className="login__brand">
          <span className="brand__name">ALLOTROPE</span>
          <span className="brand__version">v0.0.1</span>
        </header>

        <form onSubmit={onSubmit} className="login__form">
          <label className="login__field">
            <span className="login__label">Username</span>
            <input
              type="text"
              autoComplete="username"
              autoFocus
              spellCheck={false}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </label>

          <label className="login__field">
            <span className="login__label">Password</span>
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>

          {error && (
            <p className="login__error" role="alert">
              {error}
            </p>
          )}

          <button
            type="submit"
            className="login__submit"
            disabled={!canSubmit}
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
