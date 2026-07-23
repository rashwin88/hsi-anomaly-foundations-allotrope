// Authenticated home — Identity panel + a placeholder.
// Will grow into the live tiles + landing description per storyboard-spec § 5.1.

import { useAuth } from "../auth/useAuth";

export function HomePage() {
  const { state } = useAuth();
  if (state.status !== "authenticated") return null;
  const { user } = state;

  return (
    <div className="home">
      <section className="panel">
        <h2 className="panel__heading">Identity</h2>
        <dl className="panel__list">
          <div className="panel__row">
            <dt>id</dt>
            <dd className="mono">{user.id}</dd>
          </div>
          <div className="panel__row">
            <dt>username</dt>
            <dd>{user.username}</dd>
          </div>
          <div className="panel__row">
            <dt>email</dt>
            <dd>{user.email}</dd>
          </div>
          {user.display_name && (
            <div className="panel__row">
              <dt>display name</dt>
              <dd>{user.display_name}</dd>
            </div>
          )}
          <div className="panel__row">
            <dt>role</dt>
            <dd>{user.is_admin ? "admin" : "user"}</dd>
          </div>
        </dl>
      </section>

      <p className="main__line main__line--dim">
        scenes, projects, and the analyse workspace come next.
      </p>
    </div>
  );
}
