# 7. Frontend

React 19 + TypeScript + Vite 6 SPA. Built to a static bundle, served by nginx, which also
proxies `/api/*` to the api container.

## It is deliberately dependency-light

Before you reach for a library, know that these were all consciously declined:

| Not used | What's used instead |
|---|---|
| Redux / Zustand / Jotai | `useState` + three Contexts: `AuthContext`, `ToastProvider`, `HostMetricsProvider` |
| React Query / SWR | hand-rolled `fetch` in `src/api/client.ts` |
| MUI / shadcn / any UI kit | hand-rolled components; inline SVG icons in `Icon.tsx` |
| Tailwind / CSS modules | one global `src/index.css` (~5,900 lines), CSS custom properties |
| WebSockets / SSE | polling (jobs 3 s, metrics 1–2 s, action detail 3 s) |

Match this. A PR that adds a state library is a design change, not a convenience.

There is **no ESLint, no Prettier, and no tests**. `package.json` has three scripts: `dev`,
`build`, `preview`.

## Layout

```
src/
  api/          one module per backend resource; all call through client.ts
  auth/         AuthContext + useAuth
  components/   all UI — chrome, viewers, panels, charts
  data/         hand-authored React Flow graphs (model + action diagrams)
  hooks/
  pages/        one file per route
  types.ts      wire types mirroring the backend schemas
  index.css     all styling
```

## Two files dominate

- **`components/ActionDetailPane.tsx`** (~3,900 lines) — the Action card plus every
  per-action-type output viewer: the linked three-panel RGB/reconstruction/score viewer,
  the interactive threshold explorer, ROC charts, the spectral-match modal.
- **`components/SceneVisualizations.tsx`** (~1,000 lines) — the scene viewer: colour/NIR/
  SWIR/NDVI/band-mosaic modes, on-demand band rendering, click-a-pixel spectrum probe,
  annotation overlays.

If you are adding a viewer for a new action type, it goes in the first one, next to its
siblings.

## Talking to the API

`const API_BASE = "/api"` is **hard-coded**. There are no environment variables anywhere —
grep for `import.meta.env` returns nothing. The same-origin nginx proxy is what makes this
work.

`fetchJson<T>(path, init)` sets `credentials: "include"` (the auth cookie) and throws
`ApiError(status, detail)` on non-2xx. Multipart uploads bypass it and use XHR directly so
they can report progress.

## Notable libraries that earn their place

- `@xyflow/react` + `elkjs` — architecture and action flowcharts
- `uplot` — charts (lazy-imported; its built-in legend breaks past 5 series, hence the
  custom one)
- `panzoom` — linked pan/zoom keeping several image panels in transform lockstep
- `hyparquet` — reads `matches.parquet` **client-side in the browser**, so material-match
  hover probes need no server round trip

`vite.config.ts` manually splits chunks because the single bundle otherwise exceeds Vite's
import-analysis parser buffer. Leave that alone unless you know why it's there.

## The product spec lives elsewhere

Frontend source files cite [`final design/storyboard-spec.md`](../final%20design/storyboard-spec.md)
by section number, and `final design/diagrams/*.drawio` by filename. If you are changing UX
behaviour, that spec is the authority, not this file.

---

**Next:** [8. Deploy](08-deploy.md) · [9. Known issues](09-known-issues.md)
