# Allotrope

Hyperspectral and thermal satellite **anomaly detection** — a web application that ingests
raw satellite imagery, finds pixels that don't belong, and exports a georeferenced
shortlist of candidates with the material each one probably is.

Part of the GAIC HSI anomaly-detection project.

```
Scene ──► Project ──► Action ──► Action ──► … ──► Export
```

Upload a scene, chain processing Actions over it, pick a threshold by eye, export a
GeoTIFF + Shapefile + CSV bundle.

## Quick start

```bash
cp docker/.env.example docker/.env       # set POSTGRES_PASSWORD, JWT_SECRET, ADMIN_PASSWORD
docker compose -f docker/docker-compose.yml up -d
```

Then open <http://localhost:3010>.

> First build takes 30–45 minutes, and on a clean clone the frontend image **fails to
> build**. Read [docs/02-setup.md](docs/02-setup.md) before you start.

## Documentation

Start at **[docs/01-orientation.md](docs/01-orientation.md)** and read in order — the nine
files are meant to be read start to finish in about an hour.

| | |
|---|---|
| [1. Orientation](docs/01-orientation.md) | what this is, the core idea, codebase map |
| [2. Setup](docs/02-setup.md) | run it locally, the dev loop, first scene |
| [3. Data pipeline](docs/03-data-pipeline.md) | raw file → comparable cube |
| [4. Models](docs/04-models.md) | the seven foundation models |
| [5. Detectors](docs/05-detectors.md) | classical RX, scoring, material matching |
| [6. Backend](docs/06-backend.md) | API, worker, adding an Action |
| [7. Frontend](docs/07-frontend.md) | React SPA conventions |
| [8. Deploy](docs/08-deploy.md) | Docker, shipping to a GPU box |
| [9. Known issues](docs/09-known-issues.md) | what's currently broken |

Also in the repo:

- **`final design/`** — the product and UX spec the frontend implements. Frontend source
  files cite it by section; it is the authority on UX behaviour.
- **`spectal_match_sample/WALKTHROUGH.md`** — the algorithmic spec for spectral library
  matching, cited from `_spectral_library_match_run.py`.
- **`CLAUDE.md`** — orientation for AI coding agents.
- **`.claude/skills/`** — two skills that load automatically for coding agents:
  `allotrope-orientation` (domain + codebase briefing) and `iterative-nano-chunking`
  (design first, then ≤20-line chunks executed one at a time with review between each).

## Layout

```
app/        portable science — no database, no FastAPI
backend/    FastAPI api + queue worker
frontend/   React 19 + Vite SPA
scripts/    patch generation, training, deploy
research/   experiments, notebooks, per-model walkthroughs
docs/       the nine docs above
```

## Status

The core Action chain is built end to end. The **frontend build is currently broken** — a
one-file fix, and the first entry in [docs/09-known-issues.md](docs/09-known-issues.md).

There is no CI. Tests cover `app/` only.

## Licence

See [LICENSE](LICENSE).
