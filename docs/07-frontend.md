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
  the interactive threshold explorer, ROC charts. The spectral-match viewer lives in
  `components/spectralMatch/` (data loaders, `MatchMap`, the selection and sensitivity logic,
  the anomaly-points drawing, the side panels, the spectrum chart, the modal).
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

**One exception: the Inventory pages** (`pages/Inventory*.tsx`, `api/inventory.ts`) talk to
a different service, the raw-dataset inventory, at `INVENTORY_BASE = "/inventory-api"`. nginx
(`frontend/nginx.conf`, a template rendered at container start) only forwards **GET**s, only
after `/api/auth/me` accepts the session cookie, and adds the inventory's api key itself — the
browser never holds it. The upstream is resolved per request, so when the opt-in `inventory`
service isn't running nginx still starts and the pages show "not reachable or not
configured" (502/503/504/401 all map there). Lists can be huge — a collection holds ~500k
files — so the dataset page pages its files through `/files` and filters live in the URL.

## Notable libraries that earn their place

- `@xyflow/react` + `elkjs` — architecture and action flowcharts
- `uplot` — charts (lazy-imported; its built-in legend breaks past 5 series, hence the
  custom one)
- `panzoom` — linked pan/zoom keeping several image panels in transform lockstep
- `hyparquet` — reads `matches.parquet`, `family_matches.parquet` and the namer's parquet
  files **client-side in the browser**, so hover and selection need no server round trip.
  The spectral match viewer shows two linked maps (classified objects, anomaly points) whose
  layers all sit inside each map's panzoom'd element; both overlays are painted client-side,
  and the class selection and man-made sensitivity are recomputed in the browser from the
  namer parquet (`docs/lld/spectral-namer-in-product.md` §2 "Viewer")

`vite.config.ts` manually splits chunks because the single bundle otherwise exceeds Vite's
import-analysis parser buffer. Leave that alone unless you know why it's there.

## The product spec lives elsewhere

Frontend source files cite [`final design/storyboard-spec.md`](../final%20design/storyboard-spec.md)
by section number, and `final design/diagrams/*.drawio` by filename. If you are changing UX
behaviour, that spec is the authority, not this file.

## Visual style

The UI follows Edward Tufte: paper and ink, structure from whitespace and hairlines, colour
only where it means something. It lives in **`src/tufte.css`**, imported once in `main.tsx`
right after `index.css`. `index.css` still owns layout; `tufte.css` restyles it, mostly by
re-pointing the legacy tokens (`--bg`, `--fg`, `--fg-dim`, `--accent`, ...) at the ones below
and otherwise by repeating a selector so it wins by order. New screens should use the tokens
directly.

```
--paper #fffff8   --ink #111   --ink-2 #555   --ink-3 #999   --rule #ddd
--accent-manmade #c0392b   --accent-natural #2e7d6b   --accent-select #d4a017
--font-text / --font-ui  "ET Book", Georgia, serif
```

Dark mode is the same scheme inverted: `:root[data-theme="dark"]` in `tufte.css` redefines
those tokens and everything follows. Charts read the tokens at build time (`readInk()` in
`DiagnosticCharts.tsx`), so they follow the theme too.

**Font.** ET Book (MIT licence, github.com/edwardtufte/et-book) is self-hosted from
`src/assets/fonts/` (roman and bold with lining figures, italic, semi-bold), so Vite hashes it
into the bundle and it works on the air-gapped VM. No Google Fonts, no CDN. Its U+2039/203A
glyphs draw "ffi", so the `@font-face` rules exclude them by `unicode-range`.

**Rules for new screens.**
- No card boxes, shadows, gradients or tinted header bars. Separate with whitespace and a
  `1px solid var(--rule)` hairline.
- Labels are small caps in `--ink-2` (`font-variant-caps: all-small-caps`), not uppercase
  in a coloured block. Hierarchy comes from size, weight and spacing.
- Buttons are quiet: the primary action is an ink outline with weight 600 that fills with
  ink on hover; secondary is a `--rule` outline; links are underlined ink. Every focusable
  element gets the `--accent-select` focus ring.
- Status is a coloured word, not a pill: complete `--accent-natural`, failed
  `--accent-manmade`, running / needs-threshold a dark gold, queued and cancelled `--ink-3`.
  Selection is a 2 px `--accent-select` bar on the left.
- Tables: no stripes, a `--ink-3` rule under the header, `--rule` between rows, numbers
  right-aligned (`td.num`) in tabular figures.
- Charts: no grid, no background, thin `--ink-2` ticks, a range frame (axis lines span only
  the data), and labels at line ends instead of a legend (`SpectrumChart directLabels`;
  the ROC chart writes its labels inside the plot).
- At about 400 px the sidebar becomes a wrapped row of links and wide tables scroll inside
  their own box; nothing should widen the page.

---

**Next:** [8. Deploy](08-deploy.md) · [9. Known issues](09-known-issues.md)
