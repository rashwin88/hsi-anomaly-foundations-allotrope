// Admin "Create User" page (Step 3e).
//
// Consumes POST /api/admin/users. The backend gates on is_admin via the
// require_admin dependency; the frontend ALSO gates the page via the nav
// (admin-only link). Belt and suspenders.
//
// Sequence diagram: final design/diagrams/admin-create-user.drawio

import { type FormEvent, useState } from "react";

import * as adminApi from "../api/admin";
import { ApiError } from "../api/client";
import type { User } from "../types";

export function AdminUsersPage() {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<User[]>([]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const created = await adminApi.createUser({
        username,
        email,
        password,
        display_name: displayName || undefined,
        is_admin: isAdmin,
      });
      setRecent((prev) => [created, ...prev]);
      // Clear the form on success
      setUsername("");
      setEmail("");
      setPassword("");
      setDisplayName("");
      setIsAdmin(false);
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 409) {
          setError("Username or email is already taken.");
        } else if (err.status === 422) {
          setError("Invalid input. Password must be at least 8 characters.");
        } else if (err.status === 403) {
          setError("Forbidden. Admin privilege required.");
        } else if (err.status === 401) {
          setError("Not authenticated. Please sign in again.");
        } else {
          setError(err.detail ?? `Error: HTTP ${err.status}`);
        }
      } else {
        setError("Could not reach the server.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  const canSubmit =
    !submitting &&
    username.length > 0 &&
    email.length > 0 &&
    password.length >= 8;

  return (
    <div className="users">
      <section className="panel">
        <h2 className="panel__heading">Create User</h2>
        <form onSubmit={onSubmit} className="form">
          <label className="form__field">
            <span className="form__label">Username</span>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoComplete="off"
              spellCheck={false}
            />
          </label>

          <label className="form__field">
            <span className="form__label">Email</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="off"
            />
          </label>

          <label className="form__field">
            <span className="form__label">
              Password <span className="form__optional">(min 8 chars)</span>
            </span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              autoComplete="new-password"
            />
          </label>

          <label className="form__field">
            <span className="form__label">
              Display name <span className="form__optional">(optional)</span>
            </span>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              autoComplete="off"
            />
          </label>

          <label className="form__checkbox">
            <input
              type="checkbox"
              checked={isAdmin}
              onChange={(e) => setIsAdmin(e.target.checked)}
            />
            <span>Grant admin privileges</span>
          </label>

          {error && (
            <p className="form__error" role="alert">
              {error}
            </p>
          )}

          <button
            type="submit"
            className="form__submit"
            disabled={!canSubmit}
          >
            {submitting ? "Creating…" : "Create user"}
          </button>
        </form>
      </section>

      {recent.length > 0 && (
        <section className="panel">
          <h2 className="panel__heading">Created this session</h2>
          <ul className="users__list">
            {recent.map((u) => (
              <li key={u.id} className="users__row">
                <span className="users__name">
                  {u.username}
                  {u.is_admin && <span className="brand__badge">ADMIN</span>}
                </span>
                <span className="users__email">{u.email}</span>
                <span className="mono users__id">{u.id}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
