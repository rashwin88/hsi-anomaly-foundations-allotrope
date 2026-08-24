# Part 3 - Where you are

> **The one thing this part teaches:** Allotrope is a web application with four nouns, and
> every piece of analysis you will read about is an **Action** inside that structure.

## Why this comes before any code

You are about to read about detectors and neural networks. Without knowing where they run,
you will build the wrong mental model - most people assume a script that takes a file and
prints results, and then cannot work out why the code is shaped as it is.

It is a product. A user logs in, uploads a file, clicks things, and downloads a result.

## The four nouns

```
Scene ────► Project ────► Action ────► Action ────► ... ────► Export
  |                          |
uploaded                 each Action's output feeds the next
raw file
```

**Scene** - one raw satellite file, uploaded and processed into a standard form. A scene is
a place and a moment: this patch of ground, this date.

**Project** - a piece of work on exactly one Scene. Analysing the same scene two different
ways means two Projects.

**Action** - one processing step. Filter the bands. Mask the clouds. Score for anomalies.
Actions chain: later ones consume earlier ones' outputs, so a Project accumulates a small
graph of work.

**Export** - the deliverable. A zip containing a GeoTIFF, a Shapefile and a CSV naming every
candidate anomaly with its coordinates.

## The shipped chain

```
band_filter_apply          clean up the bands, make them comparable
      |
scene_segmentation  or  cloud_mask       decide which pixels to ignore
      |
anomaly_scoring            produce a score for every pixel
      |
anomaly_detection_prep     combine scores, wait for a human threshold
      |
spectral_library_match     name the material          [hyperspectral only]
      |
export                     the bundle
```

Parts 7 through 14 walk this chain. **Keep coming back to this diagram** - each part tells
you where it sits.

## Two processes, one queue

Two programs run:

- **api** - the web server. Handles requests, reads and writes the database, and hands out
  work. It never does heavy computation.
- **worker** - does the actual analysis. No web interface at all.

They communicate through a **jobs table in Postgres**. There is no Redis, no Celery, no
message broker. The api inserts a row; the worker polls for rows marked `queued`, claims
one, and works on it.

This surprises people who expect a task queue. It is deliberate: one fewer service to run,
one fewer thing to fail, and the job history is queryable with SQL.

The consequence you will actually feel: **the api can be perfectly healthy while the worker
is dead.** Its health check talks to the database, not to the worker. This has happened
twice in this repo's history, and jobs simply sat in `queued` forever. When something does
not run, check the worker before anything else.

## The step that waits for a human

`anomaly_detection_prep` is unusual. It does its computation, writes its output, and then
**stops in a state called `needs_threshold`** rather than completing.

It is waiting for a person to look at the score map and choose a cutoff.

The reason is in part 12, and it is a genuine constraint rather than a design preference: no
fixed threshold works across scenes. A calm lake and a fire-affected hillside produce scores
an order of magnitude apart, so one number flags everything in the first and nothing in the
second.

## Where the code lives

```
app/          the science. numpy, torch, rasterio. No database, no web framework.
backend/      the orchestration. api routes, worker loop, action registry.
frontend/     the React interface.
```

The rule that keeps this clean: **`app/` never imports the database or the web framework.**
It is a library of portable analysis code. `backend/` knows about both and calls into `app/`.

Practically, when you want to find how something works, the algorithm is in `app/` and the
thing that *runs* it is a pair of files in `backend/allotrope/action_types/`.

## Common confusions

**"Can one Project cover several Scenes?"**
No. A Project binds to exactly one Scene. Comparing two scenes means two Projects.

**"Actions sound like a workflow engine."**
Lighter than that. An Action names its inputs by referencing earlier Actions' output ids.
There is no scheduler and no automatic re-run; a user creates each Action.

**"Why is scoring separate from thresholding?"**
Because scoring is deterministic and thresholding is a judgement call. Splitting them means
you can try five thresholds against one expensive scoring run.

**"api and worker sound like the same codebase."**
They share one package - `backend/allotrope/` - but only the worker gets the heavy machine
learning dependencies. There is a strict rule about lazy imports that keeps torch out of the
api process. Part 13 shows why it matters.

## Check yourself

<details>
<summary>1. Name the four nouns and the relationship between each pair.</summary>

Scene is an uploaded file. A Project is work on exactly one Scene. An Action is one step
inside a Project, and Actions chain by consuming earlier outputs. An Export is the
deliverable a Project produces.
</details>

<details>
<summary>2. A user wants to analyse the same scene with two different band-filter settings. How many Scenes and how many Projects?</summary>

One Scene, two Projects. The Scene is the uploaded file; each analysis path is its own
Project, because a Project binds to one Scene but a Scene can back many Projects.
</details>

<details>
<summary>3. Jobs are stuck in "queued" and the api reports healthy. What is your first hypothesis?</summary>

The worker is not running - crashed or crash-looping. The api's health check queries the
database, not the worker, so it stays green regardless. Check the worker's container state
and logs first.
</details>

<details>
<summary>4. Why does anomaly_detection_prep stop at needs_threshold instead of completing?</summary>

It is waiting for a human to pick a threshold from the score map. No fixed cutoff works
across scenes, because typical scores differ by an order of magnitude between a calm scene
and an extreme one.
</details>

<details>
<summary>5. You need to change how cloud masking works. Which directory holds the algorithm, and which holds the thing that runs it?</summary>

The algorithm is in `app/` - specifically `app/statistical_models/`. The Action that runs it
is in `backend/allotrope/action_types/`. `app/` stays free of database and web-framework
imports.
</details>

---

Next: [part 4](04-the-sensors.md) - the five sensors and what each hands you.
