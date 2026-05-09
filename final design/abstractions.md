---
title: Abstractions — Allotrope (entity model, invariants, lifecycles)
status: in-progress (running discussion)
started: 2026-05-09
companion_to: storyboard-spec.md, final design.md
---

# Abstractions

This is the running discussion on **what each entity is, what it owns, what it relates to, and what its invariants are.** It sits between the storyboard spec and the schema/API.

- The **storyboard spec** says *what the user sees and does.*
- This doc says *what the system thinks in terms of.*
- The **schema and API** (next) follow from this doc.

Working altitude (carried forward from the storyboard): we decide abstractions first — identity, lifecycle, invariants, relationships. Concrete column types, indexes, endpoint paths, and pagination details get coloured in afterwards.

Append-only. Don't reorganize prior entries.

---

## The cast (from storyboard-spec § 6)

These are the locked entities. Each gets a section below as we discuss.

| Entity | One-line definition |
|---|---|
| **User** | Authenticated identity. Owns Projects (and possibly Visualizations / Notes). |
| **Scene** | An onboarded thermal or hyperspectral file or folder. Atomic unit of input. |
| **Annotation** | Optional artifact attached to a Scene. v1 starting type: raster mask. Multiple allowed. |
| **Project** | Workspace bound to exactly one Scene. The "big tent" containing Actions, Visualizations, Notes, and a 1:1 Result. |
| **Action** | A verb taken on a Scene within a Project. Sequential. v1 catalog: anomaly_detection, spectral_detection, cloud_mask. |
| **Action Output** | The artifact produced by every completed Action. **Invariant:** every Action has exactly one Output. |
| **Visualization** | First-class curated project-level item. Source = Scene OR Action Output, paired with a Visualization Template. Saved synchronously via the api. |
| **Visualization Template** | Reusable presentation specification (band assignments, colormap, threshold, overlay style, plot style). |
| **Note** | Project-owned free-form text. Inline-references Actions and the Scene. |
| **Result** | Auto-live, 1:1 with Project. Composes from Actions + Visualizations + Notes. No finalize step. |
| **Job** (queue) | One row per submitted Action while it traverses `queued → running → complete \| failed \| cancelled`. Postgres-backed. |

---

## What each section will lock (per entity)

For each entity we want to land:

1. **Identity** — what makes it unique; how it's referenced from other entities.
2. **Ownership** — who/what creates and destroys it; cascade behavior.
3. **Lifecycle** — states (if any), transitions, terminal conditions.
4. **Invariants** — what must always be true (e.g. *every Action has exactly one Output*).
5. **Relationships** — cardinalities to other entities (already partly in vocabulary; this is where we make them precise).
6. **Storage discipline** — where the *data* (vs metadata) lives: Postgres row vs file in `allotrope_data` / `allotrope_artifacts` / `allotrope_models`.
7. **Mutability** — what fields can change after creation; what's frozen.
8. **Versioning / history** — does the entity track changes over time, or replace-in-place.
9. **Open questions** — anything not yet locked.

---

## Cross-cutting decisions to make once (apply to every entity)

These are decided at the doc level and inherited by every entity unless explicitly overridden:

- **Identifier strategy** — UUID v4, ULID, integer serial, or composite. Affects URL shape, sortability, ingestion ergonomics, and how easy IDs are to read in logs.
- **Timestamps** — `created_at` / `updated_at` on every entity? Soft-delete (`deleted_at`) or hard-delete?
- **Multi-tenancy / ownership scope** — single-user demo for now (per storyboard-spec § 12 deferral), but the model should be tenant-aware from day one if cheap.
- **Audit trail** — do we record *who* did *what when* on each mutation? (Probably overkill for v1; flag.)
- **File-vs-row split** — what lives in Postgres, what lives in volumes. Default: metadata + small structured data in Postgres; large binary artifacts (HSI cubes, masks, rendered images) in volumes, with the row holding a path reference.
- **Naming convention** — table naming, FK naming, JSONB column naming. Boring but easier to settle once.

---

## Where to start

Pick one of three entry points (all valid):

1. **Cross-cutting first** — settle identifier strategy, timestamps, deletion model, file-vs-row split. Then walk the entities top-down with these defaults already set. Fewer rabbit holes per entity.
2. **Entity-by-entity** — start with the most central entity (probably **Project** or **Scene**) and let cross-cutting decisions emerge as we hit them. More natural narrative; risk of revisiting earlier entities once a cross-cutting rule changes.
3. **Relationships first** — sketch the full ER graph in one pass (cardinalities, FK directions, cascade rules) before drilling into any single entity's internals. Gives a complete-but-shallow picture early.

My lean is **(1)** — the cross-cutting decisions pay for themselves immediately and stop us from making the same decision N times. But your call.

**User decision:** **(1) — cross-cutting first.** Proceeding below.

---

## 2026-05-09 — Cross-cutting Session 1: Six decisions to settle once

For each decision: the realistic options, what tilts the choice for *this* product, and my lean. React to each — confirm, modify, or push back.

### CC-1. Identifier strategy

**Realistic options:**
- **`BIGSERIAL` (sequential bigint)** — `/projects/42`. Small, fast, sortable, log-readable.
- **UUID v4 (random)** — `/projects/3f29c…`. Opaque, no coordination needed; bad index-fragmentation behavior in Postgres.
- **UUID v7 / ULID (time-sortable)** — opaque + Postgres-friendly + sortable. The modern compromise.

**What tilts it for this product:**
- Single machine, single Postgres, single tenant. No need for "globally unique without coordination."
- The Action queue gets scanned hot (`SKIP LOCKED`) — sequential bigint is fastest.
- Logs / DB inspection during demo: humans read `42` more easily than `01HF…`.
- No public URLs being shared with anyone outside the demo room — ID opacity isn't a concern.

**My lean: `BIGSERIAL` everywhere.** Cheapest, fastest, most readable. If we ever need an opaque public handle for a Project (e.g. shareable export links), add a `slug` / `public_id` column then.

### CC-2. Timestamps and deletion model

**Two sub-decisions.**

#### Timestamps
- `created_at` on every row — yes, no-brainer.
- `updated_at` — useful only on entities that actually mutate after creation. Adds an `UPDATE` trigger or app-level handling.

**My lean:** `created_at` everywhere. `updated_at` **only** on mutable entities — Project (rename), Note (edit). Skip on Scene, Action, ActionOutput, Visualization (these are effectively write-once after their creation flow completes).

#### Deletion model
- **Hard-delete** with FK cascade rules. Clean rows; `WHERE deleted_at IS NULL` clutter goes away.
- **Soft-delete** (`deleted_at`). Easier to recover; safer if cascade is wrong; more clutter.

The storyboard spec explicitly locks "Project lifecycle: just exist + delete" — no archive state. That argues for hard-delete.

**My lean: hard-delete throughout, with explicit cascade rules per relationship.** Specifically:
- Deleting a **Project** cascades to its Actions, ActionOutputs, Visualizations, Notes, Result, and the queue Jobs of those Actions.
- Deleting a **Scene** is `RESTRICT`-ed if any Project references it. User must delete dependent Projects first. Avoids accidental destruction of investigations.
- Deleting an **Annotation** cascades nowhere (it's a leaf).

### CC-3. Ownership / multi-tenancy scope

**Question:** does v1 model users on entities, even though we'll seed only one user?

**My lean: model `user_id` minimally now.**
- `users` table — already needed for auth.
- `projects.user_id` — Projects are user-owned. The natural ownership root.
- **Scenes are library-shared** (storyboard spec calls the Library "global" and "persistent across users"). Scenes get a `created_by_user_id` for audit, but no access control on read.
- Visualizations, Notes, Actions inherit ownership through Project (no separate `user_id`). Saves columns and matches the mental model.
- Models, Visualization Templates: system-shared in v1 (no user FK). Defer per-user templates.

**Cost:** one column on Projects, one on Scenes, one optional on Templates. Trivial. Free correctness once we add a second user.

### CC-4. Audit trail

**Question:** do we record *who* did *what when*, beyond `created_at`?

Options:
- None.
- `created_by_user_id` per row (where applicable).
- `updated_by_user_id` per row (mutation tracking).
- Dedicated audit log table.

**My lean: `created_by_user_id` only, on Projects and Scenes.** Skip per-update audit. Skip the audit log table. Reasoning: single-tenant demo; the audit value is near-zero, and adding it later is mechanical.

### CC-5. File-vs-row split (storage discipline)

This is mostly inherited from final design.md but worth pinning at the abstraction level.

**Rule:**
- **Postgres holds:** metadata, FKs, structured small data, paths to files, JSONB configuration blobs (when small — say < ~16 KB).
- **Volumes hold:** large binary artifacts.
  - `allotrope_data` → Scene files + Annotation files (raw HE5 / TIF / GeoTIFF folders + raster masks).
  - `allotrope_artifacts` → Action Outputs (anomaly masks, cloud masks, spectral data) + saved Visualization renders + Result exports.
  - `allotrope_models` → model checkpoints.
- **Path convention:** paths stored in DB are relative to the volume's mount point. The volume name (e.g. `allotrope_data`) is *not* in the path string — it's resolved by config at read time. Lets us rename volumes without rewriting rows.

**Edge cases:**
- **Thumbnails** (small images for listings) — stored in `allotrope_artifacts` with a path on the row. Not as `bytea` in the DB. (Files > ~1 KB are almost always better in the filesystem.)
- **Spectral signature output data** (medium JSON) — JSONB column on the ActionOutput row if < ~16 KB, file path otherwise. Defer per-Action-type rule to implementation.

**My lean: lock the rule above as written.** It's a clean default and the edge cases can be decided per Action type.

### CC-6. Naming conventions

Bikeshed territory; settle once and don't revisit.

**My lean (Postgres conventions):**
- Tables: `snake_case` plural — `projects`, `actions`, `action_outputs`, `visualizations`.
- Columns: `snake_case` — `created_at`, `scene_id`, `is_active`.
- Primary key column: just `id`.
- Foreign key column: `<entity>_id` — `project_id`, `scene_id`. For nullable polymorphic refs: `source_scene_id`, `source_action_output_id`.
- JSONB columns named by purpose: `configuration`, `metadata`, `payload`.
- Booleans phrased positively: `is_active`, not `is_inactive`.
- Timestamps end in `_at`: `created_at`, `updated_at`, `started_at`, `completed_at`.
- Lifecycle / state columns: `status` (text or enum).

### Summary of leans (pending your sign-off)

| ID | Decision | Lean |
|---|---|---|
| CC-1 | Identifiers | `BIGSERIAL` everywhere; add `slug` only if public URL-sharing emerges |
| CC-2a | Timestamps | `created_at` everywhere; `updated_at` only on Project & Note |
| CC-2b | Deletion | Hard-delete; cascade through Project; RESTRICT Scene if referenced |
| CC-3 | Ownership | `user_id` on Project; `created_by_user_id` on Scene; rest inherit through Project |
| CC-4 | Audit | `created_by_user_id` on Projects & Scenes only; no audit log table |
| CC-5 | File-vs-row | Metadata + FKs + small JSONB in DB; binary in volumes via relative paths |
| CC-6 | Naming | Postgres-standard `snake_case` plural tables, `id` PK, `_id` FK, `_at` timestamps |

Tell me which to lock as-is, which to push back on, and which to discuss further. Then we walk the entities.

---

## 2026-05-09 — Cross-cutting Session 2: User decisions

### Locked as-is
- **CC-2a** — `created_at` everywhere; `updated_at` only on Project & Note.
- **CC-2b** — Hard-delete; cascade through Project; RESTRICT on Scene-delete when Projects reference it.
- **CC-4** — `created_by_user_id` on Projects & Scenes only; no audit log table.
- **CC-5** — Metadata + FKs + small JSONB in DB; binary artifacts in volumes via relative paths.
- **CC-6** — Postgres-standard naming.

### Modified
- **CC-3 — confirmed and clarified.** Project has `user_id` (owner). Scene has `created_by_user_id` for audit only — **no explicit ownership, no access control**. The Library is shared. Visualizations / Notes / Actions inherit ownership through Project. Models and Templates are system-shared (no user FK in v1).

### CC-1 — replaced with prefixed UUIDs

User: *"I prefer UUID with markers such as `scene#uuid`."*

This is the **Stripe-style prefixed identifier** pattern. The prefix communicates *what kind of thing this ID refers to* without consulting the schema. Catches type mistakes at parse time and reads cleanly in logs and URLs.

#### What it means concretely

- **Storage type:** `uuid` column in Postgres. The "prefix" is **not** stored in the DB — it's a presentation-layer concern at the api boundary.
- **Wire format (api ↔ frontend):** `<entity>_<uuid>`. E.g. `scene_3f29c4a8-…`, `project_72f9b2…`, `action_a82bd4…`.
- **Parsing:** the api accepts either bare UUIDs or prefixed forms (with prefix validated against the expected entity type — passing `action_…` where a `scene_…` is expected returns 400). All outbound serialization is prefixed.
- **Logs / DB inspection:** prefixes are reapplied for human-facing tools (admin views, error messages); raw DB queries see plain UUIDs.

#### Prefix registry (provisional, full-name style — readable beats terse)

| Entity | Prefix |
|---|---|
| User | `user_` |
| Scene | `scene_` |
| Annotation | `annotation_` |
| Project | `project_` |
| Action | `action_` |
| Action Output | `output_` |
| Visualization | `visualization_` |
| Visualization Template | `template_` |
| Note | `note_` |
| Result | `result_` |
| Job (queue) | `job_` |

(Defer if user prefers shorter prefixes Stripe-style — `scn_` / `proj_` / `act_` etc. — that's a one-line bikeshed.)

#### Separator note

User wrote `scene#uuid` with `#`. **`#` is a URL fragment identifier** and would force percent-encoding in path components (`scene%23<uuid>`). I'm reading the `#` as illustrative ("this *kind of* marker") and locking the actual separator as **`_`** (URL-safe, no escape needed, conventional). If you want a literal `#`, say so and we'll deal with the encoding cost.

#### UUID flavor

Recommendation, not yet locked: **UUID v7 (time-ordered)**. UUIDs are otherwise random which fragments Postgres B-tree indexes — a real cost on the hot `jobs` queue scan. UUID v7 keeps newer rows clustered. Pure abstract decision is just "UUID"; the v7 vs v4 choice is an implementation detail to settle when we wire it up.

#### Cost / benefit

- **Cost:** small serializer/deserializer layer in the api; one validator per type.
- **Benefit:** type-safe IDs in logs and URLs; mistakes (passing the wrong-kind ID) caught at parse, not at FK insert; reads as a serious tool — fits the "branded utility / instrument console" aesthetic from the storyboard spec.

### Final cross-cutting summary (locked)

| ID | Decision | Locked |
|---|---|---|
| CC-1 | UUID PKs (storage), prefixed-string serialization at api boundary (`scene_<uuid>`, `project_<uuid>`, etc.) | ✅ |
| CC-2a | `created_at` everywhere; `updated_at` only on Project & Note | ✅ |
| CC-2b | Hard-delete; cascade through Project; RESTRICT Scene if referenced | ✅ |
| CC-3 | `user_id` on Project; `created_by_user_id` on Scene (audit only, no ownership); rest inherit | ✅ |
| CC-4 | `created_by_user_id` on Projects & Scenes only; no audit log table | ✅ |
| CC-5 | Metadata + small JSONB in DB; binary in volumes via relative paths | ✅ |
| CC-6 | Postgres-standard naming (`snake_case` plural, `id` PK, `_id` FK, `_at` timestamps) | ✅ |

### Open micro-question on CC-1
- **Separator preference:** `_` (recommended; URL-safe), or literal `#` (requires percent-encoding in URLs)?

These cross-cutting decisions now apply by default to every entity below. Where an entity overrides one, it'll be called out explicitly.

---

## 2026-05-09 — Entity 1: User

### Purpose
The authenticated identity of someone who logs into the product. The ownership root for Projects. In v1, expected to be a single seeded operator; the model supports many users without changes.

### Identity
- **PK:** `id : uuid` (CC-1).
- **Wire format:** `user_<uuid>`.
- **Natural unique key:** `username` (case-insensitive unique constraint). Used at login.

### Ownership
- N/A — User is itself the ownership root.

### Lifecycle
- **States:** none in v1. The user simply exists.
- **Creation:** seeded by the bundle's bootstrap script at first bring-up (one operator account); admin-created flow deferred.
- **Deletion:** hard-delete, but **`RESTRICT`-ed if any Projects reference the user** (to avoid orphaning investigations). Operator must re-assign or delete the Projects first. Same pattern as Scene-delete from CC-2b.

### Invariants
- `username` is unique and non-empty.
- `password_hash` is non-empty.
- A User cannot be deleted while owning Projects.

### Relationships
- `User 1—N Project` (via `projects.user_id`, ownership).
- `User 1—N Scene` (via `scenes.created_by_user_id`, **audit only** — not ownership; the Library is shared).
- `User 1—N Note` (via Project — Notes inherit through Project; no direct `user_id` on Note).
- `User 1—N Visualization` (via Project — same).
- `User 1—N Action` (via Project — same).
- `User 1—N Session` *(if we go server-side sessions; see Open Question below)*.

### Storage discipline
- All fields in Postgres. No file-side data.

### Mutability
- `username`: **immutable** in v1 (changing usernames is a can of worms — defer).
- `password_hash`: **mutable** (password change flow). This means User legitimately needs `updated_at`, which is a **deviation from CC-2a** (which only listed Project and Note as mutable).
- `display_name` (if added): mutable.
- `last_login_at` (if added): mutable on each login.

### Versioning / history
- No history. Password rotation overwrites in place. If we ever need a "previous-password" lockout policy, we'd add a separate table — defer, not relevant for a single-user demo.

### Provisional fields (the abstraction-level shape)

```
User
  id                   uuid             (PK, prefixed user_<uuid> on the wire)
  username             text             (unique, case-insensitive)
  password_hash        text             (argon2id encoded — algo + params + salt + hash in one string)
  display_name         text?            (optional; falls back to username)
  last_login_at        timestamptz?     (audit; mutable per login)
  created_at           timestamptz      (default now())
  updated_at           timestamptz      (deviation from CC-2a — see Mutability)
```

Deferred to implementation: indexes, exact column types, `LOWER(username)` unique index for case-insensitive lookup, password-hash format details.

### Adjacent decision — session model

Sessions are *not* a property of User but they live or die with this entity. Two realistic options for v1:

- **Server-side sessions table** — `sessions(id, user_id, expires_at, created_at)`. Cookie carries the token. Pros: easy forced logout; rotation is trivial; clean audit. Cons: extra table; one round-trip on every authenticated request.
- **JWT** — token signed with a server secret, contains `user_id` and expiry. Pros: stateless, no DB read per request, simpler in code. Cons: no easy invalidation before expiry.

**My lean: server-side sessions** (small `sessions` table). Reasons: (a) we already have Postgres in the hot path; one extra row read is nothing; (b) clean logout matches the "real auth" framing; (c) JWT secret rotation is a footgun we don't need on a single-machine demo. But this is genuinely defendable either way.

If we go server-side, `Session` becomes a small extra entity (essentially: id, user_id, expires_at, created_at, last_seen_at). Doesn't really need its own walkthrough — would just be appended to this section.

### Open questions for this entity

1. **Confirm the CC-2a deviation:** User gets `updated_at` because password changes are a real mutation. Lock?
2. **Session model:** server-side sessions table, or JWT? My lean is server-side.
3. **Are there fields I'm missing for v1?** I've left out `email`, `role`, `is_active`. Do any of those need to be present from day one, or defer?
4. **Username case-sensitivity:** lock case-insensitive lookup (so `Alice` and `alice` are the same user)? Standard choice; flagging it.

### User decisions (locked)

- **Q3 — `email` added.** *"We need email, that's a minimal addition."* Lock as required, unique, case-insensitive.
- **Q1 — CC-2a deviation locked.** User has `updated_at`.
- **Q2 — server-side sessions** (small adjacent `Session` entity, see below).
- **Q4 — case-insensitive username lookup** locked.
- `role` and `is_active` deferred — not needed for v1.

### Final field set (locked)

```
User
  id              uuid             (PK; wire: user_<uuid>)
  username        text             (unique, case-insensitive, immutable)
  email           text             (unique, case-insensitive)
  password_hash   text             (argon2id encoded)
  display_name    text?            (optional; falls back to username)
  last_login_at   timestamptz?     (updated on each successful login)
  created_at      timestamptz      (default now())
  updated_at      timestamptz      (CC-2a deviation — password / display_name / email changes)
```

---

## 2026-05-09 — Entity 1b: Session (adjacent to User)

> **SUPERSEDED 2026-05-09 — see "Auth update: JWT over server-side sessions" below.** The Session entity is no longer part of the v1 model. Section retained for chronology only.

Added because we locked server-side sessions for v1 (Q2 above).

### Purpose
A short-lived row representing an authenticated browser session. Cookie carries the session id; api validates by looking up the row.

### Identity
- **PK:** `id : uuid` (CC-1).
- **Wire format:** `session_<uuid>` *(though sessions never appear in URLs — only in the auth cookie value, opaque to the user)*.

### Ownership
- Owned by a User (`sessions.user_id`).

### Lifecycle
- **States:** none — a Session simply exists until its `expires_at` passes or it is hard-deleted (logout).
- **Creation:** on successful login.
- **Deletion:** on logout (hard-delete) or on cleanup of expired rows (background sweep). Cascades from User-delete (which itself is RESTRICT-ed by Projects, so practically rare).

### Invariants
- `user_id` references a real User row.
- `expires_at > created_at`.

### Relationships
- `User 1—N Session`. Cascade-delete on User.

### Storage discipline
- All in Postgres. No file side.

### Mutability
- `last_seen_at` updates on each authenticated request *(or on a throttled cadence to avoid one write per request — defer)*.
- `expires_at` may be extended on activity *(rolling sessions)* — defer policy.
- Otherwise immutable.

### Versioning / history
- None. Sessions are ephemeral.

### Field set (locked)

```
Session
  id              uuid             (PK; wire: session_<uuid>)
  user_id         uuid             (FK → users.id, CASCADE on user delete)
  expires_at      timestamptz
  last_seen_at    timestamptz?
  created_at      timestamptz      (default now())
```

No `updated_at` — `last_seen_at` is the audit-of-record for this entity.

### Open follow-ups (deferred to implementation)
- Session lifetime (24h? 7d? rolling on activity?).
- Throttling cadence for `last_seen_at` writes.
- Cleanup strategy for expired rows (cron-like sweep vs. lazy on next read).

---

## 2026-05-09 — Auth update: JWT over server-side sessions

User: *"I prefer JWT actually. I want to learn more about it and I feel statelessness is a good thing. Not too concerned about denylists."*

This **supersedes** the Entity 1b decision above. Locked: **JWT, no `sessions` table, stateless server-side auth.**

### Quick JWT primer

A JWT is just a string with three parts joined by dots:

```
HEADER . PAYLOAD . SIGNATURE
```

Each part is base64url-encoded.

**Header** declares the algorithm and token type:
```json
{"alg":"HS256","typ":"JWT"}
```

**Payload** carries *claims* — small key/value pairs about the authenticated party. Standard claims are 3-letter:
```json
{
  "sub": "<user uuid>",            // subject — who this token is about
  "iat": 1715212800,               // issued at (epoch seconds)
  "exp": 1715299200,               // expires at (epoch seconds)
  "iss": "allotrope",              // issuer — optional, useful if multi-app
  "username": "alice"              // custom claim — anything we want
}
```

**Signature** binds header+payload to the server's secret. With `HS256`:
```
signature = HMAC-SHA256(secret, base64url(header) + "." + base64url(payload))
```

Verification is the reverse: the server recomputes the signature with its secret and checks equality. **If the payload was tampered with, the signature won't match.** No DB lookup needed — the token *is* the credential.

### What "stateless" buys us

- **Zero DB reads per authenticated request.** The api validates the signature locally and trusts the claims.
- **No `sessions` table.** One less thing in the schema, one less cleanup chore.
- **Trivial horizontal scaling** (irrelevant for the demo, but a structural property).

### What "stateless" costs

- **Can't revoke a token before its `exp`.** Logout = delete the cookie client-side. The server never knew about the token in the first place.
- **Compromised token = nuclear option.** Rotate the server secret → all tokens invalid at once. User is fine with this trade ("not concerned about denylists").

### Decisions locked for v1

| | |
|---|---|
| **Algorithm** | `HS256` (HMAC-SHA256). Single secret on the api. Simpler than RS256/ES256, which only pay off when third parties verify tokens. |
| **Storage in browser** | `HttpOnly` + `SameSite=strict` cookie. Not `localStorage` (XSS-readable). Same-origin compose stack means SameSite=strict gives us CSRF protection for free. |
| **Lifetime** | 24 hours (provisional — single setting, easy to change). No refresh-token flow in v1. If the demo ever runs longer than 24h continuous, operator re-logs in. |
| **Server secret** | Generated once at bundle bootstrap and persisted in the api container's environment (or a Docker secret). Long random string (32+ bytes). |
| **Claims set** | `sub` (User uuid), `iat`, `exp`, `username`. Maybe `display_name` if useful for the top-bar render. Keep it minimal — claims travel on every request. |

### Implications across the design

- **`sessions` table removed.** The schema is one entity smaller.
- **User entity unchanged.** Same fields as locked above.
- **`last_login_at` stays useful** — set on token issuance even though the token itself is stateless.
- **Logout endpoint** still exists; it just clears the cookie (`Set-Cookie: …; Max-Age=0`) and returns 204. Server has no row to delete.
- **The `bootstrap` one-shot service** in the bundle now also generates the JWT secret on first run if not present, persisting it for re-runs.

### What to study (since user wants to learn)

In rough order of usefulness:
1. **The IETF spec is shorter than you'd think:** [RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519) — JWT itself.
2. **JOSE family** ([RFC 7515 JWS](https://datatracker.ietf.org/doc/html/rfc7515) for signing, [RFC 7516 JWE](https://datatracker.ietf.org/doc/html/rfc7516) for encryption) — JWT is technically a JWS with JSON claims.
3. **HMAC vs RSA/ECDSA** — when each makes sense. HS256 needs the verifier to share the secret (fine for one server); RS256 lets verifiers hold only a public key (matters for OAuth providers).
4. **Storage debate**: cookies vs localStorage. Read **OWASP** on JWT security cheat sheet.
5. **Why refresh tokens exist** — to keep access tokens short-lived without forcing logins. Not in v1, but worth understanding the pattern.

### Carried forward / deferred
- Refresh-token flow (defer; not in v1).
- Token rotation policy.
- Scoped tokens / claims for finer-grained authorization (only relevant if we add roles).

---

## 2026-05-09 — Auth flow end-to-end

How a JWT actually moves through the system. Five moments matter: bootstrap, login, authenticated request, expiry, logout.

### 0. Bootstrap (once, at first bundle bring-up)

The bundle's `bootstrap` one-shot service:
1. Checks for a JWT secret. If absent: generates 32 random bytes, base64-encodes them.
2. Writes the secret somewhere persistent (Docker secret file, env var injected from a secret-volume, or `allotrope_db` row in a `system_config` table — defer the storage choice to implementation).
3. The api container reads the secret at startup. If the secret is rotated later, **all existing tokens become invalid** (they signed against the old secret). That's our nuclear-option revocation.

### 1. Login — `POST /auth/login`

```
client                                api                          postgres
  │  POST /auth/login                  │                              │
  │  { username, password }            │                              │
  ├───────────────────────────────────▶│                              │
  │                                    │  SELECT … WHERE              │
  │                                    │  LOWER(username) = LOWER($1) │
  │                                    ├─────────────────────────────▶│
  │                                    │◀─────────────────────────────┤
  │                                    │  argon2id_verify(            │
  │                                    │    password, password_hash)  │
  │                                    │                              │
  │                                    │  build claims:               │
  │                                    │   sub  = user.id             │
  │                                    │   iat  = now                 │
  │                                    │   exp  = now + 24h           │
  │                                    │   username = user.username   │
  │                                    │                              │
  │                                    │  token = base64url(header)   │
  │                                    │        + "." + payload       │
  │                                    │        + "." + HMAC-SHA256(  │
  │                                    │            secret, h+"."+p)  │
  │                                    │                              │
  │                                    │  UPDATE users SET            │
  │                                    │    last_login_at = now       │
  │                                    │    WHERE id = user.id        │
  │                                    ├─────────────────────────────▶│
  │  HTTP 200                          │                              │
  │  Set-Cookie: allotrope_jwt=…;      │                              │
  │    HttpOnly; Secure; SameSite=Strict;│                            │
  │    Max-Age=86400; Path=/           │                              │
  │  body: { user: { id, username,     │                              │
  │           display_name } }         │                              │
  │◀───────────────────────────────────┤                              │
```

Failure paths return `401` with a generic error message (don't leak whether the username exists). Constant-time comparison on the password verify (argon2id handles this internally).

**Notes:**
- The token is **never** returned in the response body. It only goes into the cookie. The frontend doesn't see it; JS can't read it (`HttpOnly`).
- `Secure` flag means the cookie only travels over HTTPS — fine in production, but on the demo machine we'll likely run HTTPS via a self-signed cert in the nginx layer to keep the flag on.
- `last_login_at` is the only DB write on login (besides the read).

### 2. Authenticated request — every other endpoint

```
client                                api                          postgres
  │  GET /projects                     │                              │
  │  Cookie: allotrope_jwt=…           │  (cookie attached by browser │
  ├───────────────────────────────────▶│   automatically)             │
  │                                    │                              │
  │                                    │  middleware:                 │
  │                                    │   1. read cookie             │
  │                                    │   2. split into h.p.s        │
  │                                    │   3. recompute signature     │
  │                                    │      with server secret      │
  │                                    │   4. constant-time compare   │
  │                                    │   5. parse payload, check    │
  │                                    │      exp > now               │
  │                                    │   6. attach user_id, claims  │
  │                                    │      to request context      │
  │                                    │                              │
  │                                    │  handler runs                │
  │                                    │   (uses claims.sub for       │
  │                                    │    project ownership filter) │
  │                                    ├─────────────────────────────▶│
  │  HTTP 200                          │◀─────────────────────────────┤
  │◀───────────────────────────────────┤                              │
```

**The api never reads the User row on authenticated requests.** The token's claims are sufficient — `sub` is the user_id used as the FK in ownership filters; `username` and `display_name` (if claimed) populate the top-bar without a join.

This is the payoff of "stateless." The DB hot path is *just* application reads, never auth lookups.

**Caveat:** if a User is deleted while their token is still valid, the token continues working on routes that don't touch user data, and FK-ownership filters silently return empty results on routes that do. For a single-user demo this is fine. For multi-user later: either re-validate user existence on each request (cheap but defeats the stateless win), or keep the 24h window short and rely on expiry.

### 3. Expiry

When `exp <= now`, the middleware returns `401 Unauthorized`. The frontend treats that as "session expired," redirects to login. No cleanup needed — there's no row.

### 4. Logout — `POST /auth/logout`

```
client                                api
  │  POST /auth/logout                 │
  │  (cookie attached)                 │
  ├───────────────────────────────────▶│
  │  HTTP 204                          │
  │  Set-Cookie: allotrope_jwt=;       │
  │    Max-Age=0; Path=/               │
  │◀───────────────────────────────────┤
```

The api validates the cookie (so we know who logged out for the audit log if we add one), then sends a clearing `Set-Cookie`. **No DB write.** The token itself remains technically valid until its `exp`, but the browser no longer holds it. If the user has the same token in another browser tab/window, that tab is unaffected — accept this as the cost of statelessness.

### Failure modes (concrete return codes)

| Situation | Status | Body |
|---|---|---|
| Missing cookie | `401` | `{"error": "unauthenticated"}` |
| Cookie present, malformed token | `401` | same |
| Signature mismatch (tampered or wrong-secret) | `401` | same |
| Token expired | `401` | `{"error": "session_expired"}` *(distinct so frontend can route to login cleanly)* |
| User deleted (FK lookups return 0 rows) | `200` with empty result, OR `404` per route | per route |

### Why CSRF is not a separate problem here

Cookies are auto-attached to same-origin requests, which is normally what makes CSRF possible. We block it without an explicit CSRF token because:
- **`SameSite=Strict`** — browser refuses to send the cookie on cross-origin navigations or fetch.
- **Same-origin** — the api and frontend are served from the same compose-network host (e.g. via nginx fronting both), so legitimate requests are always same-origin.

If we ever serve frontend and api from different origins, we add a CSRF token and switch SameSite to Lax. Defer.

### What a developer needs to remember

- **One secret, one algorithm, one cookie name.** Pinned in api config.
- **All routes except `/auth/login` and `/auth/logout` go through the auth middleware.** No exceptions in v1.
- **Token format is opaque to the frontend.** The frontend only knows: a 200 from `/auth/login` means I'm logged in; a 401 means I'm not.
- **Claims are read-only after issuance.** If a user changes their `display_name`, their existing tokens still carry the old value until next login. Trade-off we accept.

---

## 2026-05-09 — Entity 2: Scene

The atomic unit of input — an onboarded thermal or hyperspectral file (or folder, in EnMAP's case). The first "book" in the Library.

### Purpose
Represents one sensor capture. Created by the onboarding flow. Referenced by Projects (one Scene per Project; many Projects per Scene), Annotations (zero or more per Scene), and Visualizations (when a Scene is the source).

### Identity
- **PK:** `id : uuid` (CC-1).
- **Wire format:** `scene_<uuid>`.
- **Natural uniqueness:** the pair `(sensor_type, sensor_scene_id)` is unique. Prevents re-onboarding the same physical capture twice. The sensor-provided ID (PRISMA scene id, Landsat product id, EnMAP scene name) is globally unique *within* its sensor.

### Ownership
- **No ownership.** The Library is shared (per CC-3 and storyboard-spec).
- `created_by_user_id` for audit only — does not gate read access.

### Lifecycle — option to choose

Onboarding is heavy I/O (multi-GB HSI cubes copy + metadata extract + thumbnail generation). It's not instantaneous. Two ways to model the in-progress state:

- **(A) Scene row exists from the start with `status: onboarding | ready | failed`.** The worker updates the row as onboarding progresses. The Library would surface "in-progress" rows visibly.
  - Pros: single source of truth for "everything that has been started"; transparent to operators.
  - Cons: failed onboarding leaves a `failed` Scene row (a "broken scene"); Library queries need `WHERE status='ready'` everywhere; Scene becomes mutable (deviation from "Scene is write-once").

- **(B) Scene row only exists once onboarding succeeds.** Onboarding tracked as a `Job` row (using the same Postgres queue). Failures live in jobs, not in Scenes.
  - Pros: every Scene in the table is usable; no `WHERE status='ready'` clutter; Scene stays immutable post-creation; cleaner separation between "in-flight work" and "Library content."
  - Cons: the Library tile / Ingest activity tile assembles its display from two sources (jobs for in-progress, scenes for completed) — tiny extra frontend work.

**My lean: (B).** Cleaner, matches the storyboard's mental model that the Library is a shelf of completed "books," and keeps Scene effectively write-once which fits CC-2a (no `updated_at`).

### Invariants
- `(sensor_type, sensor_scene_id)` is unique.
- `raw_path` references a file (or folder root) that exists in the `allotrope_data` volume at row-write time.
- A Scene cannot be deleted while any Project references it (RESTRICT, per CC-2b).

### Relationships
- `Scene 1—N Annotation` — onboarded together; cascade-delete (when a Scene is deleted, its Annotations are deleted with it).
- `Scene 1—N Project` — RESTRICT on Scene delete.
- `Scene 0—N Visualization` (when `source_scene_id` is set on a Visualization). Cascade behavior follows the Visualization's owning Project, not the Scene; killing a Scene that has Visualizations referencing it is RESTRICT-ed transitively through Project.
- `Scene N—1 User` (via `created_by_user_id`, audit-only, **no cascade** — if the User who onboarded a Scene is deleted, the Scene stays; we set the audit field NULL or RESTRICT the User-delete since CC-2b already RESTRICTs User-delete on Project ownership).

### Storage discipline
- **Postgres row holds:** id, sensor + scene id pair, display name, structured metadata (typed columns + JSONB), file paths, audit fields, timestamps.
- **`allotrope_data` volume holds:** the raw scene file(s) — HE5 for PRISMA, TIF for Landsat, folder for EnMAP. Path stored in `raw_path` (relative to the volume mount).
- **`allotrope_artifacts` volume holds:** generated thumbnails. Path stored in `thumbnail_path`.

### Mutability
- All fields **immutable** post-onboarding (option B above).
- No `updated_at`.
- Renaming Scenes is **deferred** — name is set at onboarding, frozen thereafter. (If we add rename later, we add a CC-2a deviation then.)

### Versioning / history
- None. A Scene is a snapshot of one capture.

### What metadata sits as first-class columns vs JSONB

This decision drives the Scenes-page filter surface. Storyboard-spec § 5.2 explicitly calls out "deep advanced filtering" as first-class — so anything we want to filter on cheaply needs to be a real column, not a JSONB key.

**My lean — first-class columns (filter targets):**
- `acquisition_at : timestamptz?` — when the sensor captured the scene.
- `bbox_min_lat`, `bbox_max_lat`, `bbox_min_lon`, `bbox_max_lon : numeric` — geographic bounds. Four numeric columns supports rectangular filtering without PostGIS.
- `projection : text` — EPSG code or string identifier (e.g. `EPSG:32633`).
- `band_count : int`.
- `cloud_cover_pct : numeric?` — derived during onboarding for sensors that expose it.
- `valid_pixel_pct : numeric?` — derived during onboarding.
- `has_annotations : boolean` (denormalized; updated when annotations attach/detach) — enables the "scenes with ground truth" filter without a JOIN.

**Everything else in `metadata : jsonb`** — sensor-specific fields (PRISMA viewing angle, Landsat WRS path/row, EnMAP processing level, etc.), plus anything we discover later. Schema per sensor is documented but not enforced at the column level.

### Open question — PostGIS?

**No PostGIS in v1, my lean.** Simple `bbox_*` columns support rectangular filtering, which covers the demo audience's expected interactions ("scenes in this region"). PostGIS pays off when we need polygon containment, projection transforms, or distance queries. Adding it later is mechanical (a `geometry` column derived from existing bounds). Adding it now bloats the bundle and forces a Postgres-with-PostGIS image (heavier).

### Provisional field set (locked unless pushed back on)

```
Scene
  id                      uuid             (PK; wire: scene_<uuid>)
  sensor_type             text             (enum: prisma | landsat9 | enmap)
  sensor_scene_id         text             (sensor-provided unique id within sensor)
  name                    text             (display name; default = sensor_scene_id)
  acquisition_at          timestamptz?     (capture time)
  bbox_min_lat            numeric
  bbox_max_lat            numeric
  bbox_min_lon            numeric
  bbox_max_lon            numeric
  projection              text             (e.g. EPSG:32633)
  band_count              int
  cloud_cover_pct         numeric?
  valid_pixel_pct         numeric?
  has_annotations         boolean          (default false; maintained by Annotation insert/delete)
  metadata                jsonb            (sensor-specific extras)
  raw_path                text             (relative to allotrope_data volume)
  thumbnail_path          text?            (relative to allotrope_artifacts volume)
  created_at              timestamptz      (= ingested_at since option B)
  created_by_user_id      uuid             (FK → users; audit only; SET NULL on user delete or RESTRICT)
```

UNIQUE constraint: `(sensor_type, sensor_scene_id)`.

### Open questions / decisions for the user

1. **Lifecycle: confirm option (B)** — Scene row only exists once onboarding succeeds; in-progress and failed onboardings live in the `jobs` table. My lean: B.
2. **PostGIS in v1?** My lean: no. Simple bbox columns suffice for the demo's filtering needs; add PostGIS later if real geospatial queries emerge.
3. **First-class column set above** — anything missing or excess for v1?
4. **Should `has_annotations` be a denormalized column, or computed via a JOIN/EXISTS each time?** My lean: denormalized — filter performance matters and updates are infrequent (only when an Annotation is added/removed).
5. **Cascade on `created_by_user_id`** — `SET NULL` (preserve the Scene, lose the audit) or `RESTRICT` (block user delete)? Note that CC-2b already RESTRICTs User-delete via Project ownership, so this is largely academic. My lean: `SET NULL` for simplicity.

### Adjacent — sensor-specific onboarding behavior (deferred)

Each sensor has its own metadata extraction pipeline (HE5 parsing for PRISMA, TIF metadata for Landsat, GeoTIFF folder + XML for EnMAP). The shape of the *Scene* entity is uniform; what varies is *how onboarding fills it in*. The existing pipeline code (`FileHelper`, `DatasetBuilder` etc. per `CONTEXT.md`) handles the sensor-specific work — onboarding wraps that into a Job. Detailed extraction logic is implementation-side; only the column shape needs to be agreed here.

---

## 2026-05-09 — Scene refinement: aligned with existing dataloaders

Inventoried what the existing dataloaders already capture per sensor (cross-referenced against `app/utils/stac/`, `app/models/file_processing/enmap_metadata.py`, `app/utils/files/he5_helper.py`, `app/utils/files/tif_helper.py`, and the per-sensor STAC parsers). Refining the first-class column set to match the data actually flowing through.

### What the existing pipeline already harmonizes (universal across sensors)

The STAC item creation step in `app/utils/stac/stac_utils/file_name_parsers.py` already produces a uniform set of fields for *all three sensors*:

| Field | Source | Type |
|---|---|---|
| `platform` | parser per sensor | text — `Prisma` / `landsat-9` / `EnMAP` |
| `datetime` | parser per sensor | timezone-naive datetime |
| `processing_level` | parser per sensor | text — `L2D` / `L2SP` / `L2A` etc. |
| `product_type` | parser per sensor | text |
| **bbox** `[min_lon, min_lat, max_lon, max_lat]` | per-sensor `get_*_bounding_box` | 4 floats in **EPSG:4326** (already reprojected) |
| **scene id** (parsed from filename) | `file_name_parsers.py:34` | text — `PRS_L2D_…`, `LC09_…`, `ENMAP01-…_L2A-DT…_…` |

This is a gift: STAC harmonization means the sensor-uniform columns are already produced by the pipeline. Onboarding just stuffs them into the Scene row.

### What is sensor-specific (goes to JSONB)

- **PRISMA:** `List_Cw_Swir`, `List_Fwhm_Swir`, `List_Cw_Vnir`, `List_Fwhm_Vnir` (band metadata), `L2Scale*Min/Max` (DN scaling), geolocation arrays. `app/utils/files/he5_helper.py`.
- **Landsat 9:** native CRS/transform, QA_PIXEL flags (cloud/shadow/cirrus/water/snow), B10 adaptive cloud mask presence. `app/utils/files/tif_helper.py`, `landsat_dataset_builder.py`.
- **EnMAP:** `tile_id`, `datatake_id`, full `bounding_polygon` (corner list), `quality_flags` (cloud_cover, haze_cover, cirrus_cover, snow_cover, water_cover, cloud_shadow, dead_pixels_vnir/swir), `detector_boundary`, `spatial_info`. `app/models/file_processing/enmap_metadata.py`.

### Refined first-class column set (locked unless pushed back on)

Changes from the prior proposal:
- **Added** `processing_level`, `product_type`, `native_projection` (these are real, harmonized, and filter-worthy).
- **Renamed** `projection` → `native_projection` to reflect that bbox is *always* in EPSG:4326 (per STAC harmonization) — the native projection is a separate fact about the source file, useful for ops users who care about UTM zones.
- **`cloud_cover_pct`** kept but acknowledged as heterogeneous: EnMAP provides it in metadata; Landsat can derive it from B10 / QA_PIXEL during onboarding; PRISMA may be `NULL`.
- **`valid_pixel_pct`** kept but **soft-deferred** — compute only if cheap during onboarding; otherwise leave NULL. Don't block v1 on producing it where the dataloaders don't already.

```
Scene
  id                      uuid             (PK; wire: scene_<uuid>)
  sensor_type             text             (enum: prisma | landsat9 | enmap;
                                            normalized from `platform`)
  sensor_scene_id         text             (parsed from filename)
  name                    text             (display; default = sensor_scene_id)

  -- harmonized, all-sensor (from STAC)
  acquisition_at          timestamptz?     (from `datetime`, normalized to UTC)
  processing_level        text?            (L2D / L2SP / L2A …)
  product_type            text?
  bbox_min_lon            numeric          (EPSG:4326)
  bbox_min_lat            numeric          (EPSG:4326)
  bbox_max_lon            numeric          (EPSG:4326)
  bbox_max_lat            numeric          (EPSG:4326)
  native_projection       text?            (file's native CRS, e.g. EPSG:32633)
  band_count              int

  -- derived during onboarding (where available)
  cloud_cover_pct         numeric?         (EnMAP: from metadata; Landsat: from B10/QA;
                                            PRISMA: typically NULL)
  valid_pixel_pct         numeric?         (deferred; populate if cheap, else NULL)

  -- denormalized
  has_annotations         boolean          (default false; updated by Annotation insert/delete)

  -- sensor-specific
  metadata                jsonb            (PRISMA band metadata, Landsat CRS/transform,
                                            EnMAP quality_flags + detector_boundary +
                                            full bounding_polygon, etc.)

  -- file references
  raw_path                text             (relative to allotrope_data)
  thumbnail_path          text?            (relative to allotrope_artifacts)

  -- audit / lifecycle
  created_at              timestamptz      (= ingested_at; option B)
  created_by_user_id      uuid?            (FK → users; SET NULL on user delete)
```

UNIQUE: `(sensor_type, sensor_scene_id)`.

### Convention locked
- **All bbox columns are in EPSG:4326.** No per-row CRS for bbox — the convention is the schema. `native_projection` carries the original source CRS for downstream tools that need it.
- **Timestamps stored as `timestamptz` in UTC.** STAC produces tz-naive datetimes; onboarding assumes UTC and normalizes on insert.

### Other Scene leans — locked unless pushed back on

| Open Q | Lean | Status |
|---|---|---|
| Lifecycle option B (Scene = ready by definition; in-flight in `jobs`) | yes | locked |
| PostGIS in v1 | no | locked |
| `has_annotations` denormalized | yes | locked |
| `created_by_user_id` cascade | `SET NULL` | locked |

---

## 2026-05-09 — Two structural additions: Vendable and a Jobs sidebar

User: *"Onboarding in a job. The scene is created only after the job is completed. Let's have a jobs link on the side bar where we can show queued and running jobs. Also I think onboarding a scene means we have to also produce the vendable for the scene and store it."*

Two changes — one to the Scene entity, one to the sidebar.

### Vendable becomes a first-class artifact of every Scene

Per `CONTEXT.md`, the existing pipeline is `raw → FileHelper → DatasetBuilder → VendableDataset → PatchGenerator → webdataset shards`. The **VendableDataset** is the sensor-uniform, ready-to-consume representation of a scene. Promoting it to a stored Scene artifact means every downstream Action consumes the vendable directly — never re-parses the raw HE5/TIF/folder.

**Locked:**
- The Scene row gains `vendable_path : text` (relative to `allotrope_data`).
- **Onboarding's success criterion expands:** raw landed + metadata parsed + **vendable built and persisted** → only then does the Scene row get inserted. Vendable failure = onboarding failure = no Scene row (consistent with lifecycle option B).
- The vendable lives in `allotrope_data`, not `allotrope_artifacts`. Rationale: it's *scene data*, not the output of an Action. Convention: `allotrope_data/scenes/<scene_id>/raw/…` + `allotrope_data/scenes/<scene_id>/vendable/…`.

### Updated Scene field set

Adding one column. Everything else as locked above.

```
Scene
  …                       (all prior fields)
  raw_path                text             (relative to allotrope_data)
  vendable_path           text             (relative to allotrope_data)   ← NEW
  thumbnail_path          text?            (relative to allotrope_artifacts)
  …
```

`vendable_path` is **NOT NULL** — every Scene has a vendable by definition (no Scene exists without one).

### Updated onboarding flow

The worker's processing step (orange phase in the sequence diagram) now includes a "build vendable" beat:

```
worker:
  · move raw files → allotrope_data/scenes/<scene_id>/raw/
  · parse metadata via per-sensor dataloader
  · build VendableDataset → allotrope_data/scenes/<scene_id>/vendable/
  · derive bbox / projection / band_count
  · compute cloud_cover_pct where available
  · generate thumbnail → allotrope_artifacts/thumbnails/
  · validate required fields
```

If vendable construction fails, the job fails, raw files are cleaned up (or left in a quarantine for debugging — defer), and no Scene row is created. Onboarding is atomic from the Library's perspective.

### Sidebar grows to 6 — adds Jobs

User wants a **Jobs** sidebar destination showing **queued + running** jobs (and presumably recent history).

**Updated sidebar (locked):**

```
1. Home
2. Scenes        (Library; hosts Ingest button)
3. Projects
4. Models        (read-only catalog)
5. Jobs          ← NEW (queue visibility)
6. Monitoring    (system + workload metrics)
```

**Why this is the right addition:**
- Onboardings and Actions both go through the queue. Without a Jobs page the operator has no first-class place to *see* a stuck/failed job — it'd be buried in the Project workspace or in Monitoring's metrics.
- It fits the instrument-console aesthetic. "What is the system doing right now" gets a literal answer page.
- It complements Monitoring without duplicating it: **Monitoring = aggregate metrics**, **Jobs = the actual work items, listed**.

### Jobs destination — light spec (structural only)

- A list of jobs, ordered by recency / priority. Filterable by status (`queued`, `running`, `complete`, `failed`, `cancelled`) and type (`scene_onboard`, `action_run`, ...).
- Per-row: id, type, status, started_at, elapsed, owning entity (scene_id for onboarding, action_id+project_id for an action run), failure_reason if failed.
- Click → job detail (configuration payload, full log/error trace, link to the resulting entity if `complete`).
- v1 default view: queued + running. History accessed via filter.

(Detailed UI deferred per the working-altitude rule.)

### Workload tile re-routes

The Workload tile on the Home landing previously navigated to **Monitoring**. With Jobs in the sidebar, a more accurate destination is **Jobs** — the tile shows queue depth and throughput; clicking through, the operator wants to see the actual queued/running items, not aggregate sparklines.

Workload tile content stays: queue depth, throughput, avg duration today (all reads off the same `jobs` table). Navigation target swaps to `/jobs`.

### Storyboard-spec implications (will update separately)

- § 4 sidebar list: 5 → 6 items.
- § 5: add a § 5.6 for Jobs.
- § 5.1 Home tiles: Workload tile navigates to Jobs instead of Monitoring.

### What's unchanged

- Lifecycle option B still holds (Scene only exists once the job completes — the vendable build is part of "completes").
- Cross-cutting decisions all hold.
- Architecture / image set unchanged.

---

## 2026-05-09 — Entity 3: Annotation

Optional artifact attached to a Scene. v1 starting type: a raster mask (e.g. ground truth labels). A Scene may carry zero, one, or many Annotations.

### Purpose
Attach human- or externally-supplied information about a Scene's content. The canonical example is a per-pixel ground-truth mask used to overlay on the scene viewer and (later, deferred) compute evaluation metrics against Action Outputs.

### Identity
- **PK:** `id : uuid` (CC-1).
- **Wire:** `annotation_<uuid>`.
- **Natural uniqueness:** none. Duplicates are allowed (a user might intentionally attach the same labels twice, or two analysts might attach competing labels for the same scene).

### Ownership
- Owned by a **Scene** (`scene_id` FK). Like Scenes themselves, Annotations are Library-scoped, not user-scoped — once attached, they're visible to anyone who can see the Scene.
- `created_by_user_id` for audit only, same pattern as Scene.

### Lifecycle
- Created in two ways:
  - **Bundled with Scene onboarding** — the Ingest flow accepts optional annotation files alongside the raw scene. Both land atomically in the same `scene_onboard` job's terminal transaction (`INSERT scenes; INSERT annotations[*]; COMMIT`).
  - **Attached to an existing Scene later** — a separate `annotation_attach` job type. Same queue, same worker. User picks a Scene from its Detail page, uploads an annotation file, job runs (validate + copy + INSERT + flip `has_annotations`).
- **No states.** An Annotation either exists (it's been validated and persisted) or it doesn't. In-flight attaches live in the `jobs` table — same pattern as Scene lifecycle option B.
- **Hard-delete** is allowed and synchronous. No need to queue an annotation deletion (the file is small enough; row delete + file unlink is a quick api operation).

### Invariants
- `scene_id` references a real Scene.
- `file_path` references a real file in `allotrope_data` at row-write time.
- `type` is one of the supported enum values.
- Inserting an Annotation sets the parent Scene's `has_annotations = true`. Deleting the *last* Annotation flips it back to `false`. Maintained at app level, in the same transaction.

### Relationships
- `Scene 1—N Annotation` — **CASCADE on Scene delete** (annotations belong to their scene; if the scene goes, they go).
- `User 1—N Annotation` — via `created_by_user_id`, audit-only, **SET NULL** on user delete (CC-3 / CC-4 pattern).

### Storage discipline
- Postgres row: id, scene_id, type, name, description, file path, metadata JSONB, audit fields.
- File in `allotrope_data` under the scene's directory, keeping annotations co-located with their scene:
  - **Convention:** `allotrope_data/scenes/<scene_id>/annotations/<annotation_id>/<filename>`

### Mutability
- **Immutable post-creation.** No `updated_at`. Renaming an Annotation is deferred (same call as Scene rename — defer to a future CC-2a deviation if it earns its place).

### Versioning / history
- None. If a labeler produces a new version of the labels, they upload a new Annotation. The old one stays unless explicitly deleted.

### v1 type catalog (locked unless pushed back on)

- **`raster_mask`** — georeferenced raster (TIF / GeoTIFF) aligned to the Scene. Binary or multi-class.

That's it for v1. Storyboard-spec § 6 already deferred other types (vector polygons, point lists, spectral libraries) — they remain deferred. Adding a type later is a one-row enum extension + a per-type handler.

For multi-class rasters, the class label map (`{0: "background", 1: "anomaly", 2: "shadow"}`) lives in `metadata` JSONB along with any provenance info ("hand-labeled by Lab X", "model-generated by RX-Detector v0.3", etc.).

### Provisional field set

```
Annotation
  id                      uuid             (PK; wire: annotation_<uuid>)
  scene_id                uuid             (FK → scenes.id; CASCADE on scene delete)
  type                    text             (enum: raster_mask in v1; extensible)
  name                    text             (display name; user-supplied)
  description             text?            (free-form, optional)
  file_path               text             (relative to allotrope_data;
                                            convention: scenes/<scene_id>/annotations/<id>/<filename>)
  metadata                jsonb            (type-specific extras —
                                            for raster_mask: class label map,
                                            source/provenance, num_classes, etc.)
  created_at              timestamptz
  created_by_user_id      uuid?            (FK → users; SET NULL on user delete)
```

### Open questions / decisions (structure-level)

1. **Type enum: `raster_mask` only in v1?** Confirm; this is what storyboard Session 10 said. (My lean: yes.)
2. **`description` column**: include it? Annotations on scientific datasets often need a longer "what does this represent" beyond the name. (My lean: include — it's free.)
3. **Attach-existing-scene flow: through the queue (`annotation_attach` job) or synchronous via api?** Annotation rasters can be sizable (full-scene resolution). Going through the queue is consistent with onboarding and gives us free async/progress UX. Synchronous keeps the api code surface flat. (My lean: through the queue.)
4. **Delete: synchronous via api?** No queue, immediate row delete + file unlink. (My lean: synchronous — small operation.)
5. **`has_annotations` maintenance — app-level or DB trigger?** App-level is simpler, fits "all writes go through the api/worker," and avoids Postgres trigger surprises during testing. (My lean: app-level.)

### User decisions (locked)

All five questions confirmed with the leans:
- `raster_mask` is the only type in v1; other types added later by extending the enum + adding handlers.
- `description` column included.
- Attach-existing-scene goes through the queue as a `annotation_attach` job.
- Delete is synchronous via the api (row delete + file unlink in one request).
- `has_annotations` is maintained by application code in the same transaction as the annotation insert/delete.

### Locked field set

```
Annotation
  id                      uuid             (PK; wire: annotation_<uuid>)
  scene_id                uuid             (FK → scenes.id; CASCADE on scene delete)
  type                    text             (enum: raster_mask in v1)
  name                    text
  description             text?
  file_path               text             (relative to allotrope_data;
                                            convention: scenes/<scene_id>/annotations/<id>/<filename>)
  metadata                jsonb
  created_at              timestamptz
  created_by_user_id      uuid?            (FK → users; SET NULL on user delete)
```

### New job type unlocked
- `annotation_attach` — added to the `jobs.type` enum (alongside `scene_onboard`, future `action_run`).

---

## 2026-05-09 — Entity 4: Project

The workspace where investigation happens. Bound to exactly one Scene; a Scene can host many Projects. A Project is the **ownership root** for everything inside it — Actions, Visualizations, Notes, and the (1:1) Result. Smallest entity by columns, but the structural pivot of the whole product.

### Purpose
A user-owned container for an investigation on a single Scene. Holds the work products created during that investigation. The most-touched entity in the application — every action the user takes inside a Project workspace traces back to a row here.

### Identity
- **PK:** `id : uuid` (CC-1).
- **Wire:** `project_<uuid>`.
- **Natural uniqueness:** none. A user can reasonably have multiple Projects on the same Scene (different angles, different stages of an investigation).

### Ownership
- **Owned by a User** via `user_id` FK (CC-3 — Project is the ownership root for everything below it).
- **Bound to a Scene** via `scene_id` FK. The relationship is structural, not ownership: a Scene "hosts" a Project but doesn't own it (the Library is shared; ownership lives at Project).
- No separate `created_by_user_id` audit field — `user_id` *is* the creator and remains so (no ownership transfer in v1).

### Lifecycle
- **No states.** Just exists. Storyboard-spec § 5.3 is explicit: lifecycle = exist + delete.
- Created when a user clicks "New Project" and chooses a Scene (canonical dialog from storyboard-spec § 5.3, used both from the Projects landing's New button and the Scene Detail page's Create Project button).
- **Hard-delete** cascades down to all child entities — Actions, Action Outputs, Visualizations, Notes, and the Result row (if Result is materialized — see § Result entity later).

### Invariants
- `user_id` references a real User.
- `scene_id` references a real Scene.
- `name` is non-empty.

### Relationships
- `Project N—1 User` — **RESTRICT** on User delete (we already locked this in the User entity: a User can't be deleted while owning Projects).
- `Project N—1 Scene` — **RESTRICT** on Scene delete (CC-2b: a Scene can't be deleted while any Project references it; user must delete dependent projects first).
- `Project 1—N Action` — CASCADE.
- `Project 1—N Note` — CASCADE.
- `Project 1—N Visualization` — CASCADE.
- `Project 1—1 Result` — CASCADE (decision on whether Result is even a row TBD when we walk Result).

### Storage discipline
- **All in Postgres.** No files belong to the Project entity itself. Files belong to its children (Action Outputs, Visualizations, Result exports).

### Mutability
- `name` — **mutable** (rename).
- `description` (if included) — mutable.
- `user_id`, `scene_id` — **immutable**. No ownership transfer; no scene-rebinding (a Project on a different scene is a different project).
- `updated_at` — per CC-2a, Project is one of the explicit mutable entities, so `updated_at` is present.

### Versioning / history
- None. Renames overwrite in place.

### Provisional field set

```
Project
  id              uuid             (PK; wire: project_<uuid>)
  user_id         uuid             (FK → users.id; RESTRICT on user delete; immutable)
  scene_id        uuid             (FK → scenes.id; RESTRICT on scene delete; immutable)
  name            text             (display name; user-supplied; NOT NULL; mutable)
  description     text?            (free-form, optional; mutable)
  created_at      timestamptz
  updated_at      timestamptz      (CC-2a: mutable entity)
```

Smallest table so far. Most of the *interesting* data lives in entities that FK to this one.

### Open questions / decisions (small)

1. **`description` column — include?** Same pattern as Scene/Annotation. Useful for "what is this investigation about." Free to include. (Lean: include.)
2. **Name uniqueness — any constraint?** Options: globally unique, unique per user, unique per (user, scene), or no constraint. (Lean: no constraint — multiple projects with the same name across different scenes is fine; UI-side disambiguation via scene name + timestamp.)
3. **Should Project carry any transient UI state** (currently-selected Action id, currently-applied Template id on the scene viewer)? (Lean: **no** — that's client-side. Persisting it makes every panel toggle a DB write, and it doesn't survive in any meaningful way across users / sessions.)

That's it for Project at the entity level. The interesting structural questions (workspace layout, action ordering, visualization curation) are storyboard-side and already locked or deferred.

### User decisions (locked)

All three confirmed with leans:
- `description` column included.
- No name uniqueness constraint.
- No transient UI state on Project — strictly client-side.

### Locked field set

```
Project
  id              uuid             (PK; wire: project_<uuid>)
  user_id         uuid             (FK → users.id; RESTRICT on user delete; immutable)
  scene_id        uuid             (FK → scenes.id; RESTRICT on scene delete; immutable)
  name            text             (NOT NULL; mutable)
  description     text?            (mutable)
  created_at      timestamptz
  updated_at      timestamptz
```

Locked relationships:
- `Project N—1 User` — solid, RESTRICT, ownership
- `Project N—1 Scene` — solid, RESTRICT, structural binding
- (1—N relationships to Action / Note / Visualization and 1—1 to Result drawn as those entities lock)

### Project-rooted storage layout (locked)

Project becomes a parent folder in the artifacts volume, mirroring the cascade semantics in the DB. Project deletion is `rm -rf` of the project folder — single operation, no orphans.

```
allotrope_data/
  scenes/
    <scene_id>/
      raw/...
      vendable/...
      annotations/<annotation_id>/...

allotrope_artifacts/
  scenes/
    <scene_id>/
      thumbnail.png                              ← scene-level derived (during onboarding)
  projects/
    <project_id>/
      actions/<action_id>/output/...             ← per-Action artifacts
      visualizations/<visualization_id>/...      ← curated visuals
      exports/<result_id>/<bundle>               ← Result snapshots on export
```

**Conventions:**
- `file_path` columns hold paths relative to the volume mount; the volume name is config, not data.
- Scene-level artifacts (thumbnails, future scene-render caches) live at `allotrope_artifacts/scenes/<scene_id>/`, parallel to `projects/` — they're not owned by any project.
- A Visualization is **always** stored under its owning Project's folder, even when its source is a Scene (the Visualization is project-scoped data; the source Scene is a reference, not a parent).
- Action / Visualization deletion = `rm -rf` of the corresponding subfolder.
- Project deletion = `rm -rf allotrope_artifacts/projects/<project_id>/`.

---

## 2026-05-09 — Entities 5 & 6: Action and Action Output (paired walk)

These two are tightly coupled: every completed Action has exactly one Action Output (1:1 invariant). Walking them together because the Output's identity is meaningless without the Action.

### Note on the Jobs queue (referenced below; formal entity walk later)

These entities interact with the **Jobs** table, which we'll formally walk later. For now, treat it as known: Jobs is the single Postgres-backed queue. Workers pull from `jobs` via `SELECT … WHERE status='queued' FOR UPDATE SKIP LOCKED LIMIT 1`. Each Job row has a `type` (`scene_onboard | annotation_attach | action_run`), a `payload` JSONB, and lifecycle fields. A Job's lifetime is "from submission to terminal state, then it sticks around for history" — visible in the Jobs sidebar.

---

## Entity 5: Action

### Purpose
A **verb taken on a Scene within a Project**. The unit of investigation work. Storyboard-spec § 7 catalog: anomaly_detection, spectral_detection, cloud_mask.

Created when the user clicks "+ New Action" in the Project workspace, configures inputs and parameters, and submits. Runs asynchronously on the worker.

### Identity
- **PK:** `id : uuid` (CC-1).
- **Wire:** `action_<uuid>`.
- **Natural uniqueness:** none.

### Ownership
- **Owned via Project** (`project_id` FK). No `user_id` on Action — ownership inherits through Project (CC-3).
- No `created_by_user_id` either (CC-4 limits audit fields to Projects + Scenes).

### Lifecycle

**Status enum (locked):** `queued → running → complete | failed | cancelled`.

- **Created at submit time** with `status='queued'`. The Action row appears in the workspace immediately, even before the worker picks it up. *(Different from Scene: Scene exists only on success; Action exists from queue-time.)*
- An accompanying **Job row** of type `action_run` is enqueued with `payload={action_id}`.
- Worker pulls the Job; sets `actions.status='running'` AND `jobs.status='running'` in the same transaction.
- On success: worker creates the **ActionOutput row**, sets `actions.status='complete'`, sets `jobs.status='complete'`, all in one transaction.
- On failure: worker sets both to `failed`, records `failure_reason` on both.
- On cancellation:
  - **Cancel-while-queued:** api can mark Action and Job as `cancelled` directly (Action hasn't been picked up yet).
  - **Cancel-while-running:** best-effort. api sets a "cancellation requested" flag; worker checks at boundaries. If acknowledged, both rows go to `cancelled`.

**Status duplication note:** `actions.status` and `jobs.status` are kept in sync by the worker. The Action's status is denormalized so UI queries don't need a JOIN to render the Action list. Workers maintain the invariant inside their transactions.

### Mutability
- **`status`** mutable (state transitions managed by worker; api for cancellation).
- **`failure_reason`** mutable (set on failure transition).
- **`started_at`, `completed_at`** mutable (one-time set by worker).
- All other fields **immutable** post-creation. No `updated_at` (status transitions are the only mutation, and `started_at`/`completed_at` already record them).

### Invariants
- `project_id` references a real Project.
- `type` is one of the v1 catalog types.
- `configuration` JSONB validates against the type's schema (api enforces at submit).
- **`status='complete'` ⇔ exactly one ActionOutput row exists with `action_id=this.id`.** For any non-complete status, zero ActionOutput rows.

### Relationships
- `Action N—1 Project` — CASCADE on Project delete.
- `Action 1—1 ActionOutput` — produced on completion; CASCADE on Action delete.
- `Action 1—1 Job` (the action_run job) — Job's `payload.action_id` points to this Action; deleting the Action cascades the Job (or we may keep Jobs as historical records — TBD when we walk Job).

### Storage discipline
- All Action data in Postgres. The heavy artifacts live on the ActionOutput entity (see § 6 below).

### Inputs

An Action's inputs are heterogeneous: a **Scene** is always required; **prior ActionOutputs** and/or **Annotations** can optionally feed in (e.g. spectral_detection on an anomaly_detection Output, or anomaly_detection with a ground-truth ROI from an Annotation).

**v1 design — inputs in `configuration` JSONB:**

```json
{
  "input_scene_id": "scene_<uuid>",
  "input_action_output_ids": ["output_<uuid>", ...],
  "input_annotation_ids": ["annotation_<uuid>", ...],
  "params": {
    "algorithms": ["RX", "iso_forest"],   // anomaly_detection only
    "threshold": 0.95,                     // type-specific
    ...
  }
}
```

- Validation at submit time: api checks all referenced ids exist and belong to the same Project (no cross-project chaining).
- **No FK enforcement at the DB level** — saves us a polymorphic-FK or join-table headache.
- Referential safety relies on the cascade structure: Actions only delete via Project delete, so all referenceable entities (Scene-via-Project, prior ActionOutputs in the same Project, Annotations on the bound Scene) are guaranteed present for the Action's lifetime.

If/when v2 introduces individual Action delete or cross-project chaining, we revisit and likely add a join table.

### Are Actions individually deletable in v1?

**Lean: no.** Storyboard-spec didn't mention individual Action delete; cleanup happens at Project delete. This:
- Removes the orphan-input problem entirely.
- Keeps the Actions list simple (append-only chronology — fits the science-notebook framing).
- Cancelling is the correct way to "abandon" a started Action; the row stays, status='cancelled'.

If a user really wants to clean up a project, they delete the project and start fresh.

### Provisional field set

```
Action
  id                      uuid             (PK; wire: action_<uuid>)
  project_id              uuid             (FK → projects.id; CASCADE on project delete)
  type                    text             (enum: anomaly_detection | spectral_detection | cloud_mask)
  configuration           jsonb            (input refs + params; type-specific schema)
  status                  text             (enum: queued | running | complete | failed | cancelled)
  failure_reason          text?            (set on status=failed)
  started_at              timestamptz?     (set when worker picks up)
  completed_at            timestamptz?     (set on terminal state)
  cancellation_requested  boolean          (default false; flag for cancel-while-running)
  created_at              timestamptz
```

---

## Entity 6: Action Output

### Purpose
The artifact produced by every **completed** Action. 1:1 with its Action when status='complete'; otherwise no row.

### Identity
- **PK:** `id : uuid` (CC-1).
- **Wire:** `output_<uuid>`.
- **Natural uniqueness:** `action_id` is unique (one Output per Action). Could also be modeled as `action_id` being the PK directly, but a separate UUID lets Visualizations reference the Output by its own opaque id without exposing the Action's id.

### Ownership
- Through Action through Project. No direct user FK.

### Lifecycle
- Created in the same transaction the worker uses to mark the Action complete.
- Immutable thereafter — no edits, no versions.
- Deleted only via Action delete (which only happens via Project delete).

### Invariants
- `action_id` is unique and references an Action with `status='complete'`.
- `artifact_path` references real files on disk at row-write time.
- The Output's *content shape* matches the parent Action's `type` (e.g. `cloud_mask` Action's Output is a binary mask; `anomaly_detection`'s is a heatmap or a bundle of per-algorithm artifacts).

### Relationships
- `ActionOutput 1—1 Action` — PK relationship, CASCADE.
- `ActionOutput 1—N Visualization` — when a Visualization sources from this Output. CASCADE on ActionOutput delete *(Visualization is also under the same Project, so the cascade is internally consistent)*.

### Storage discipline
- **Artifact files** in `allotrope_artifacts/projects/<project_id>/actions/<action_id>/output/...` (per the Project-rooted layout locked above).
- For multi-algorithm anomaly_detection: directory contains per-algorithm subdirs (`output/RX/...`, `output/iso_forest/...`) plus a comparison artifact.
- **Small structured data** (metrics, summary stats, comparison tables for multi-algorithm runs) lives in a `summary` JSONB column on the row, not on disk.

### Mutability
- Fully immutable. No `updated_at`.

### Per-Action-type Output schema (interpretation, not separate tables)

We don't make separate tables per Action type. One unified `action_outputs` row per Action, with the `summary` JSONB and `artifact_path` interpreted by the parent Action's `type`. Reasoning:
- Schema enforcement happens at the api/worker layer (validation against Pydantic models per type).
- Per-type tables would multiply schemas without buying real safety; the JSONB-on-disk pattern handles type-variation cleanly.

### Provisional field set

```
ActionOutput
  id                      uuid             (PK; wire: output_<uuid>)
  action_id               uuid             (FK → actions.id, UNIQUE; CASCADE on action delete)
  artifact_path           text             (relative to allotrope_artifacts; the Output's directory)
  summary                 jsonb            (metrics, comparison tables, small structured data)
  created_at              timestamptz
```

Tiny table. Most Output substance lives on disk (artifact_path) or in JSONB.

---

## Open questions / decisions for the Action / ActionOutput pair

1. **Action not individually deletable in v1.** Cleanup via Project delete. Cancellation is the way to abandon. (Lean: lock.)
2. **Inputs in `configuration` JSONB, no FK enforcement at DB level.** Validation at submit time; referential safety from cascade structure. (Lean: lock for v1; revisit when individual Action delete enters scope.)
3. **Single unified `action_outputs` table; type-shape interpretation by parent Action's `type`.** (Lean: lock.)
4. **`actions.status` denormalized from `jobs.status`, kept in sync by worker.** (Lean: lock — UI queries don't pay JOIN cost.)
5. **Cancellation:** cancel-while-queued is direct; cancel-while-running uses a `cancellation_requested` flag the worker checks at boundaries. Best-effort. (Lean: lock.)

Sign off, push back, or "all leans, lock and move."

---
