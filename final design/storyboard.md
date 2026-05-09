---
title: Demo Storyboard — Allotrope
status: ideating
started: 2026-05-08
parent: final design.md
---

# Demo Storyboard — Allotrope

Running notes from the storyboard ideation. The storyboard answers: *"What does a viewer see, click, and walk away thinking?"* Everything downstream (API surface, screens, data on disk) should serve the storyboard, not the other way around.

---

## 2026-05-08 — Session 1: Framing the storyboard

Before we sketch scenes, we need to lock the framing. The same product looks completely different as a 5-minute pitch vs. a 30-minute technical deep-dive.

### Framing questions to answer
1. **Who is the audience?**
   - Technical/scientific (geospatial researchers, ML engineers) — they want to see anomaly maps, model architecture, residuals.
   - Operational (defense, intel, environmental analysts) — they want to see "did it find the thing? how fast?"
   - Commercial / funders — they want to see polish, a clear ROI story, and a smooth click-path.
   - Mixed.

2. **How long is the demo slot?**
   - 5 min: one scripted golden path, no detours.
   - 15 min: golden path + one "what if I drop in new data?" moment.
   - 30+ min: golden path, fresh-data ingest, model internals, monitoring.

3. **What is the single "wow moment"?**
   - The visual reveal of an anomaly on a hyperspectral scene? (Most likely candidate — HSI false-color is striking.)
   - The speed (live inference on a fresh scene)?
   - The "drop in unseen data and watch it work" moment?
   - The system panel showing real GPU utilization?

4. **What do we want the viewer to walk away saying?**
   - "I want to use this." (Commercial.)
   - "This is technically credible." (Scientific.)
   - "This solves my problem." (Operational.)
   - This drives whether the demo emphasizes UX polish, scientific rigor, or end-to-end speed.

5. **Constraints / non-goals:**
   - Offline (already locked in).
   - Single operator clicking through (no multi-user theatrics).
   - What we **don't** want to show? (E.g. don't open a terminal, don't show training, don't show raw HE5 file paths.)

### Strawman storyboard (to react to, not to commit to)
Just to give us something concrete to push back against:

1. **Open** — landing screen shows a list/gallery of pre-loaded scenes (PRISMA, Landsat, EnMAP), each with a thumbnail.
2. **Pick a scene** — click a tile → scene viewer opens with RGB false-color, band selector, zoom/pan.
3. **Run anomaly detection** — click "Detect anomalies" → progress indicator → anomaly heatmap overlay appears on the scene. System panel (top-right) shows the spike in GPU utilization.
4. **Inspect** — click on a flagged region → side panel with spectral signature plot, anomaly score, model confidence.
5. **Drop in fresh data (optional act 2)** — operator drags a new dataset folder onto the app → ingestion progress → new scene appears in the gallery → repeat steps 2–4 on it.
6. **Close** — export an anomaly map (PNG/GeoTIFF) to show "this is real output, not just a viewer."

### Open ideation questions
- Should there be a "compare two scenes" view (e.g. before/after, or two sensors over the same area)?
- Do we want a model-selection dropdown, or is there exactly one model in the demo?
- How is the anomaly visualized — colored heatmap overlay, contour boxes, both?
- Does the side-panel spectral plot come from real per-pixel spectra, or an aggregate? (Real is more credible but requires holding the cube in memory.)
- What's the failure story? If inference takes 45 seconds, what does the screen show during that time so the audience doesn't get bored?

---

## 2026-05-08 — Session 2: Framing answers locked

User has answered the framing questions. Recording verbatim-ish so we don't lose intent.

### Audience: **operational + commercial** (mixed)
- Operational viewers care about "did it find the thing, how fast, how reliably."
- Commercial viewers care about "is this real, is it polished, is the team credible."
- Implication: we need both technical substance AND visible polish. Cannot lean fully on either side.

### Time: **30 minutes**
- Long enough for the full arc: setup → scenes → detection → inspect → fresh data → export.
- Long enough that pacing matters — we should think about beats, not just screens.

### "Wow moment": **no single moment — visual density, features, attention to detail**
- This is a reframing of the question, not a sidestep. The product wins by **looking and feeling like a serious tool**, not by having one cinematic reveal.
- Implication for design: every screen needs to be earning its place. Empty gradients, oversized hero cards, "lorem ipsum" sidebars — out. Real data on every panel, real numbers, real spectra, real overlays.
- Visual density = lots of meaningful information per screen, but legibly arranged. Not crammed. Think Bloomberg terminal or a flight ops console more than a SaaS landing page.

### Walk-away line: **"I'd fund more compute for you to build more sophisticated models"**
- This is the north star. Every design decision should support it.
- It implies the audience leaves believing: (a) the team can ship, (b) the existing model is real, (c) the bottleneck is compute/sophistication, not capability or seriousness.
- So the demo should hint at the model's current ceiling and gesture at what bigger compute would unlock — without ever claiming things the model can't do today.

### Anti-goal: **must not look "AI-generated single-prompt"**
- User explicitly does not want the UI to feel like the output of a one-shot LLM prompt. No generic gradient hero. No dashboard template. No emoji-decorated cards.
- Build from primitives upward. Earn every component. The aesthetic should feel hand-crafted and specific to the domain (HSI, geospatial, scientific instruments), not "modern startup template."

### How we'll drive this
- **User leads the UX vision.** I (Claude) am a sounding board, not the designer. I'll capture, push back where useful, and ask sharpening questions — but the experience description comes from the user.
- We start from basics. No proposing finished screens. Build up the experience piece by piece in the user's words first, then translate to layout.

### What's pending
The user is going to describe the experience they want, in their own words. Stop proposing storyboards until that lands.

---

## 2026-05-08 — Session 3: Beat 1 — Login

User-led description (verbatim intent): **"First we need a login page. Simple — a username and a password."**

### What's locked
- The product opens on a **login page**, not directly into the app.
- Two fields: **username**, **password**. No SSO, no magic link, no third field, nothing else specified yet.
- "Simple" is the operative word — login is not the place to flex visual density. It's the doorway.

### What's still open (questions for the user, not assumptions)
1. **Real auth or demo stub?**
   - Real: Postgres `users` table, hashed passwords (bcrypt/argon2), session token (JWT or cookie), proper logout. Adds backend surface but signals seriousness.
   - Stub: one hardcoded credential checked in-memory, returns a session flag. Simpler, but a savvy demo viewer might notice if pressed.
   - Recommendation to react to: real auth, even if there's only one seeded account. Cheaper than it sounds, and consistent with the "operational + commercial" framing — they'll expect it.
2. **One user or many?** Single operator account for the demo, or do we seed several roles (e.g. analyst, admin)? Affects whether we need a users table with roles or just a single login.
3. **What happens after a successful login?** (Don't sketch it — just need to know if the next screen is something specific you already have in mind, so we know where the login hands off.)
4. **What happens on a failed login?** Inline error under the field, a banner, or just a shake. Small detail but it's part of the "attention to detail" promise.
5. **Aesthetic anchor — does the login screen set the tone?** First impression for the audience. Two directions, pick one to react to:
   - **Pure utility:** centered card, two fields, one button, dark background. Zero ornamentation. Sets a "this is a tool, not a product brochure" tone immediately.
   - **Branded utility:** same minimal form, but framed with a small product mark, version number, and maybe a single contextual visual element (e.g. a faint hyperspectral spectrum strip across the top). Still restrained — but signals identity.

### Captured (do not over-interpret)
The user said *simple*. Whatever we propose for the login should be defensible as "simple." If we find ourselves adding a third element, we should justify it or drop it.

### Locked from this session
- **Auth: real.** Postgres-backed `users` table, hashed passwords (argon2 or bcrypt), proper session token. Even if only one seeded account at demo time, the implementation is real.
- **Aesthetic: branded utility.** Minimal form (username, password, button) framed with a small product mark, version number, and one restrained domain element (e.g. a faint hyperspectral spectrum strip). No ornamentation beyond that. The login is identity-aware but not decorative — it sets the tone for the whole app.
- **Tone-setting note:** every later screen should feel consistent with what the login establishes. If we find ourselves designing a later screen that clashes with "branded utility," it's the later screen that's wrong.

### Still open (carry forward)
- ~~What happens immediately after a successful login?~~ **Answered in Session 4 below.**
- What happens on failed login? (Smaller detail, can be revisited.)
- Single seeded user or multiple roles? (Defer until we know whether the rest of the app exposes role-gated features.)

---

## 2026-05-08 — Session 4: Beat 2 — Landing page (post-login)

User-led description (verbatim intent): **"On logging in the user sees a landing page — it has a sidebar which contains different options that the user has, the top bar has the usual user icon and some settings etc. The landing page has clear description of the platform — what it can do and the different capabilities."**

### What's locked
- After successful login → **landing page** (not directly into a workflow screen).
- Persistent **sidebar** with navigation options. (Sidebar persists beyond the landing page — it's the app's spine.)
- Persistent **top bar** with user icon and settings.
- Landing page **content** = clear description of the platform + its capabilities.

### What this implies (capturing for awareness, not deciding)
- The sidebar items are effectively the **table of contents of the entire product**. Whatever ends up there is what the demo will walk through. Defining this list is the highest-leverage decision in the whole storyboard — every other screen lives behind one of these entries.
- The landing page is **operational + commercial framing in one screen**: it has to read as "this is what we do" without becoming a marketing brochure. Tension to manage explicitly.
- Top bar + sidebar = standard app shell. The shell needs to be designed once and inherited by every screen — that's where "branded utility" gets stress-tested.

### Tension to flag
A "landing page with a clear description of platform capabilities" can very easily drift into the exact thing the user said no to in Session 2 — a SaaS-template hero section with gradient cards and feature tiles. We need to be careful that this page reads like a **mission console's startup screen** (think: scientific instrument boot screen, ground station home, ops center default view), not a product website.

One way to think about it: the audience is *already inside* the app — they don't need to be sold on it. The landing page is informing, not pitching. It can be denser and more matter-of-fact than a public website would be.

### Questions for the user (highest leverage first)
1. **What are the sidebar items?** This is the most important question on the table. Strawman list to react to (drawn from what the platform already does, per CONTEXT.md and the existing pipeline) — tear it apart:
   - **Home** (the landing page)
   - **Scenes** (browse loaded HSI scenes — PRISMA, Landsat, EnMAP)
   - **Detection** (run anomaly detection on a selected scene)
   - **Results** (browse past detection runs and their outputs)
   - **Ingest** (drop in a fresh dataset)
   - **Models** (which models are available, their checkpoints, performance metrics)
   - **Monitoring** (live system + GPU/CPU metrics)
   - **Settings** (probably top-bar accessed, not sidebar — but flagging)

   Are any of these wrong, missing, or named badly?

2. **The "capabilities" description on the landing page — is it textual, visual, or both?**
   - Textual: short paragraphs, lists, plain language. Reads more like a mission brief.
   - Visual: small thumbnails / tiny live previews of each capability (a strip of HSI false-color, an anomaly heatmap, a live system metric). Reads more like a console.
   - Both: one block of orienting text + a row of "live tiles" that each link into a sidebar destination.

3. **Top bar — beyond user icon and settings, what else?** A clock? Project/dataset name? A status indicator (e.g. "GPU online")? Or strictly minimal — just user + settings?

### Captured (do not over-interpret)
- "Clear description of the platform" is informational, not promotional. Adjust register accordingly.
- "Different capabilities" — we should resist inventing capabilities to fill space. Whatever the sidebar lists is what we describe; nothing more.

### Locked from this session

**Sidebar** (in order, top to bottom):
1. **Home** — the landing page.
2. **Ingest** — drop in / register a fresh dataset.
3. **Scenes** — list of ingested scenes; can be browsed, possibly viewed.
4. **Analyse** — pick a scene and start an "analysis project of sorts." A single scene can be analysed multiple ways with different settings. **This is the heart of the product** (user explicitly flagged it).
5. **Results** — one result package per analysis.
6. **Models** — available models, checkpoints, metrics.
7. **Monitoring** — live system metrics, deeper than the top-bar sparklines.

**Settings** lives in the **top bar**, not the sidebar.

**Landing page capabilities block:** a small row of **live tiles**. Each tile shows real, current information from a capability area — not static marketing cards. Tiles need real data behind them, so the landing page is a live screen, not a static brief.

**Top bar:**
- User icon + settings (right side).
- **Active project** name (if one is loaded — empty/dimmed otherwise).
- **GPU status** indicator.
- **Small sparklines** of CPU util, GPU util, RAM. Persistent on every screen — gives the audience continuous visual proof that the system is real and working.

### Implications worth flagging

**1. The data model is becoming visible.** From the user's description:
- A **Scene** is the input unit (an ingested dataset).
- An **Analysis** is a configured run on a Scene — same scene can have many analyses with different settings.
- A **Result** is a 1:1 package produced by an Analysis.

So roughly: `Scene 1—N Analysis 1—1 Result`. The user's phrase *"analysis project of sorts"* hints that an Analysis might also be a longer-lived container (notes, multiple sub-runs, history) — needs clarification before we lock the schema.

**2. The top-bar sparklines are dense info, but they fit the aesthetic.** They're the visual signature of an instrument console — exactly the Bloomberg/flight-ops register we're aiming for. They earn their place; they're not decoration.

**3. Live tiles tie the landing to backend liveness.** They imply a `/summary` endpoint or a small set of polled endpoints feeding the home screen. Worth noting now so we don't forget when we get to API design.

**4. "Analysis project" needs concretizing.** Two readings:
   - (a) An Analysis = a single run with a single settings configuration. "Analyse a scene different ways" = create multiple Analyses on the same scene. Result is per-analysis.
   - (b) An Analysis Project = a longer-lived workspace that holds multiple runs against a scene, with notes and history. The "Result" is the latest/canonical run's output, but earlier runs are inspectable inside the project.

   These look similar but the schema and UI diverge meaningfully. Reading (b) is richer and matches the word "project"; (a) is simpler and faster to demo.

### Questions for the user (highest leverage first)

1. **Concretize "Analysis."** Is it (a) a single configured run, or (b) a longer-lived project that contains multiple runs and notes? This decides the Postgres schema and the entire Analyse workflow.

2. **What goes on each live tile?** A tile per sidebar item (Ingest, Scenes, Analyse, Results, Models, Monitoring = six tiles), or a curated subset? And what's the *content* of each tile — a count, a recent item, a tiny chart, a status?
   Strawman to react to (one tile per relevant capability):
   - **Ingest** — last ingest event ("EnMAP folder, 12 min ago") or empty state.
   - **Scenes** — count of scenes by sensor (e.g. "PRISMA: 4  ·  Landsat: 7  ·  EnMAP: 2").
   - **Analyse** — most recent or in-progress analysis.
   - **Results** — count of results, or thumbnail of the latest anomaly map.
   - **Models** — active model name + checkpoint version.
   - **Monitoring** — current GPU util as a number + tiny sparkline.

3. **Models page — read-only or interactive?** Just a browseable inventory, or can the user select/swap which model an Analyse run uses from there?

### Carry-forwards
- Top-bar metric polling cadence — 1s feels right for instrument feel; we can revisit when we wire it up.
- Empty states — what does each sidebar destination look like before any data exists? Important for the very-first-launch demo opening.
- Whether Monitoring should also surface per-process / per-model metrics, beyond what the top bar shows.

---

## 2026-05-08 — Session 5: The Analysis Project, defined

User-led clarification (verbatim intent): **"An analysis project corresponds to a single scene, can involve multiple models and involves the user taking notes etc. A project can have single runs of multiple models on a scene. So in that sense multiple runs."**

User explicitly **contested the framing of "run as the natural unit."** The **Project** is the unit; runs happen inside it.

### Vocabulary (locked — use these terms consistently from here on)

- **Scene** — an ingested dataset (PRISMA HE5, Landsat TIF, EnMAP folder, or a fresh-on-the-spot dataset).
- **Project** (a.k.a. *Analysis Project*) — the investigation workspace. Belongs to a single Scene. Holds multiple Runs and the user's Notes. **This is the unit of work in the product.**
- **Run** — one execution of one Model with one settings configuration, inside a Project. A Project typically has several Runs.
- **Note** — user-authored text inside a Project (possibly attached to a Run or a region — TBD).
- **Result** — the **project-level** deliverable. Bundles all Runs + Notes + comparisons into the export package. There is **one Result per Project**, not per Run.

### Why this matters

The deliverable is the **investigation**, not any single inference output. That's much more credible to an operational + commercial audience: they buy investigations, not predictions. It also gives the user a real reason to use multiple models — because comparing them inside a project *is* the work product.

### Refined data model (provisional, will inform Postgres schema)

```
User 1—N Project
Scene 1—N Project        (a Scene can host multiple Projects; not yet confirmed — see Q1 below)
Project 1—1 Scene
Project 1—N Run
Project 1—N Note
Project 1—1 Result       (the Result is the rolled-up project export)
Run    1—1 Model
Run    1—1 Settings      (the settings config used for that run)
```

### Implications

- **Scope is bigger than option (a) from Session 4.** We're now building a multi-run workspace, not a "click → result" flow. That's the right call for the audience but worth naming explicitly so we don't underestimate the build.
- **The Analyse area is now a multi-pane workspace**, not a single screen. It needs at minimum: scene context, runs list, current run config, run output viewer, notes panel.
- **A "compare runs" view becomes natural** — multiple models on the same scene begs for it, and it's exactly the kind of dense, instrument-console feature that fits the aesthetic.
- **Results page changes character.** It's no longer "list of model outputs" — it's "list of completed investigations." Each Result row is a closed Project.

### Questions for the user

1. **Scene → Project cardinality.** Can a single Scene have **multiple Projects** (different investigators, different angles, different times)? Or is it strictly one Project per Scene? My lean is *multiple* — investigations on the same scene by different people or with different framings are a natural use case, and it costs us nothing in the schema. But your call.

2. **Notes — what do they attach to?**
   - Project-level only (one notes pane per project, free text).
   - Project-level + per-Run (each run can have its own notes).
   - Project-level + per-Run + per-Region (notes pinned to a coordinate or polygon on the scene). This is the most powerful and most demo-worthy, but also the most build.

3. **Is the Result generated automatically, or does the user explicitly "finalize/publish" a project?**
   - Auto / live: the Result page always shows the project's current state. No "done" concept.
   - Explicit finalize: the user clicks "Generate Result" and it freezes a snapshot — adds a sense of *closing an investigation*, which feels right for the operational framing. Also matches "Result package" language.

4. **Compare-runs view inside a project — yes/no?** Almost a given once we have multiple runs, but want to confirm before designing for it. If yes, we should sketch it as a first-class feature, not an afterthought.

### Still owed (carried forward, don't lose these)
- **Live tiles** — content of each one. Strawman in Session 4 still on the table.
- **Models page** — read-only inventory, or interactive (select / swap models from there)?

---

## 2026-05-08 — Session 6: Library, Project, Action, Output

User-led reframing (verbatim intent):
> "A scene is exactly what it sounds like — a thermal or a hyperspectral file. Users can onboard a scene and come back to it later. A scene is like a book in a virtual library. The library grows as users onboard scenes. **Critical to note that a scene is onboarded to the scene library.**
>
> A project is a workspace. Each project operates on only one scene. But a scene can be associated with multiple projects.
>
> **Actions** are taken on a scene within a project. Running multiple anomaly detection algorithms on a scene is an action. Running spectral detection on detected anomalies is an action. A scene + some action configuration produces an **action output**. This way we can have many actions on a scene in a project workspace and their associated outputs."

This is a substantial refinement of the vocabulary. Updating accordingly.

### Vocabulary (revised — these are now the canonical terms)

- **Scene** — a single thermal or hyperspectral file (or folder, in EnMAP's case). The atomic unit of input. **Books** in the library metaphor.
- **Scene Library** — the global, persistent collection of all onboarded scenes. The library *grows* over time as users onboard scenes. The "Scenes" sidebar item is the view into the library.
- **Onboarding** — the act of bringing a scene into the library. The "Ingest" sidebar item is where this happens. Details of what onboarding entails are deferred to a later session (user explicitly said so).
- **Project** — a workspace. **Strictly one Scene per Project.** A Scene can be associated with **multiple Projects**.
- **Action** — a verb taken on a Scene **within** a Project. Examples the user gave:
  - Running multiple anomaly detection algorithms on a scene.
  - Running spectral detection on detected anomalies.
- **Action Configuration** — the parameters / settings that specify how an Action is performed.
- **Action Output** — the artifact produced when an Action is executed against a Scene with a given Configuration. **One Output per Action.**
- **Note** — user-authored text inside a Project (still TBD what it attaches to).
- **Result** — the **Project-level** deliverable. Bundles all Action Outputs + Notes into the export package. **One Result per Project.**

**Retired terms:** "Run" is out. The user explicitly contested it earlier and "Action" is the replacement. Don't reintroduce "Run" anywhere.

### Critical observation — Actions can chain

The user's second example — *"running spectral detection on detected anomalies"* — is decisive. The spectral-detection Action takes the **output of a prior anomaly-detection Action** as its input. This means:

- Actions form a **DAG** within a Project (directed acyclic graph), not a flat list.
- An Action's input can be either: the raw Scene, or an existing Action Output (or both).
- The Analyse workspace is therefore not a "click → result" form. It's closer to a **pipeline composer** — sequence Actions, route outputs into the next Action's input, build up an investigation.
- This is a major UX win for the "fund more compute for sophisticated models" line: the audience visibly sees the user *composing* an investigation, not just clicking a button. The product looks like a tool for thinking, not a model demo.

### Refined data model

```
Library 1—N Scene                    (the persistent global library)
Scene   1—N Project
Project 1—1 Scene
Project 1—N Action                   (Actions in the project DAG)
Project 1—N Note
Action  1—1 ActionConfiguration
Action  1—1 ActionOutput
Action  0—N input_action_outputs     (an Action's inputs can include other Actions' outputs — the DAG edge)
Project 1—1 Result                   (rolled up from Outputs + Notes)
```

### Implications for screens (capturing, not designing)

- **Scenes sidebar = Scene Library** — the page should feel like a library, not a generic list. Sortable, searchable, with sensor type, ingest date, size, thumbnail. The "books" metaphor is a useful aesthetic anchor — **shelves of scenes**, each with a spine showing key metadata at a glance.
- **Analyse sidebar = Projects** — but specifically a Project workspace. The list of projects is one screen; opening a project is the workspace screen, which is multi-pane and DAG-aware.
- **Results sidebar** — list of completed Project Results. Each row = one closed investigation, one downloadable package.
- **Ingest** is the onboarding flow into the Library. Specifics deferred.

### Implications for the Postgres schema (preview, will design later)

- `scenes` table — one row per onboarded scene (sensor type, file paths in `allotrope_data` volume, ingest metadata, thumbnail path).
- `projects` table — one row per project, FK to `scenes`.
- `actions` table — one row per action, FK to `projects`, `action_type`, `configuration` JSONB, status, FK to `models` (if applicable).
- `action_inputs` table — many-to-many between `actions` and prior `actions` (the DAG edges).
- `action_outputs` table — one row per output, FK to `actions`, paths to artifact files in `allotrope_artifacts` volume.
- `notes` table — FK to `projects`, optionally to `actions`, and possibly to a region (TBD).
- `results` table — one row per project, FK to `projects`, snapshot path.

### Questions for the user

1. **Confirm action chaining is real.** My read of "spectral detection on detected anomalies" is that the spectral-detection Action consumes the anomaly Action's Output as input — i.e. Actions chain into a DAG. **If yes**, the Analyse workspace becomes pipeline-composer-shaped (huge UX implication, but a strong fit for the audience and the north star). **If no**, Actions stay independent and the workspace is a flat list. This is the highest-leverage question on the table.

2. **"Running multiple anomaly detection algorithms" — one Action or many?** You phrased it as a single Action. Two ways to read that:
   - (a) One Action with a config like `{algorithms: [A, B, C]}` that fans out internally and produces one bundled Output.
   - (b) Three separate Actions, each running one algorithm, each with its own Output. The "multiple algorithms" framing is just the user's *intent*, expressed as several Actions.
   Either is defensible. (a) gives a cleaner Output ("here's the comparison"); (b) gives more granular history and lets the user kill/rerun individual algorithms. Which matches your mental model?

3. **Initial action catalog.** Beyond *anomaly detection* and *spectral detection on anomalies*, what other Actions should we plan for in v1? Some candidates from the existing pipeline / domain:
   - Band math / spectral indices (NDVI, NDWI, custom).
   - Cloud masking (we already have an adaptive B10 cloud masker — this is an obvious Action).
   - Validity / nodata masking.
   - Visualization / false-color rendering (or is this just a viewer concern, not an Action?).
   - Classification / segmentation.
   - Change detection (would require two scenes — breaks the "one scene per project" rule, so probably not in v1).
   Which of these are real, which are noise?

### Still owed (these are getting old — let's clear them next)
- **Live tiles** — content of each one (Session 4 strawman still on the table).
- **Models page** — read-only inventory, or interactive selection?
- **Notes** — what they attach to (Project / per-Action / per-Region).
- **Result generation** — auto-live vs explicit finalize.

---

## 2026-05-08 — Session 7: Action chaining, multi-algorithm, catalog

### Locked

**1. Actions chain by reference, not by composition. No pipeline-composer UX.**

User explicitly rejected the "compose a DAG then execute" framing. The flow is sequential and discrete:
1. Create an Action (pick type, pick inputs from what's available, configure, run).
2. Wait for it to complete.
3. *Then*, separately, create another Action whose input may be an earlier Action's Output.

Important consequences:
- The data model still supports an Action taking another Action's Output as input (the DAG edges are real in storage).
- But the **UI is not a graph editor**. It's a **sequential list of Actions** within a Project, each with status, configuration, and output. New Actions are created via a "+ New Action" affordance that lets you pick inputs from anything already produced in this Project.
- This is much simpler to build and to demo. It also reads more like a **science notebook** — discrete cells of work, each producing an output the next can refer to. That fits the "tool for thinking" aesthetic better than a Node-Red-style pipeline canvas.

**2. Multi-algorithm anomaly detection = one Action.**

Confirmed reading (a) from Session 6: a single Action can fan out across multiple algorithms internally, producing one bundled Output. The Output itself can hold multiple per-algorithm artifacts plus a comparison view. This means Actions are higher-level than "one model run" — they're more like *steps of an investigation*.

**3. Initial action catalog (v1).** User confirmed all four:
- **Anomaly detection** — runs one or multiple algorithms on a Scene; bundled comparison Output.
- **Spectral detection** — operates on a Scene + an anomaly Action's Output (or a region of interest).
- **Cloud masking** — leverages the existing B10 adaptive cloud masker. Probably mostly used as input/preprocessing for other Actions.
- **Rendering** — false-color / band-combination visualizations of a Scene. Producing presentation-grade imagery for inclusion in the Result.

The catalog is a fixed v1 set; expansion comes later.

### Implications for the Analyse workspace

Now that Actions, their inputs, and their sequential creation are clear, the Analyse workspace has a definite shape. **Not designing it yet** — just naming what it must contain so we can come back and lay it out:

- A **scene context panel** (which Scene this Project is on, key metadata, thumbnail).
- A **list of Actions in this Project**, in the order they were created — each shows type, status, configuration summary, timestamp.
- A way to **inspect any Action's Output** (clicking a completed Action opens its output viewer).
- A **+ New Action** affordance — opens a small dialog: pick action type (from the catalog), pick input(s), configure, run.
- A **notes panel** somewhere (TBD what it attaches to).
- The **scene viewer** as the dominant visual area — overlaid with the currently-selected Action's Output (anomaly heatmap, cloud mask, rendered band combo, etc.).

The mental model: a scientist's investigation log. Linear list of steps, each producing something you can see, each available as input for the next step.

### What "Action" gives us as an abstraction

Worth naming because it changes how we'll talk about everything downstream:
- **The Action catalog is the product surface area.** Adding a capability = adding an Action type. That's a cleaner extensibility story than "adding a new model" or "adding a new feature page."
- **An Action is a contract:** typed inputs (Scene, optional ActionOutput refs, configuration), typed output. Backend implementation can plug in behind a uniform interface.
- **Demos scale with the catalog.** Want to show the audience something new? Add an Action type. The UI doesn't change shape; the catalog grows.

### Carry-forward backlog (let's clear next session)

These are accumulating; pick them off before we go deeper:
- Live tile contents (Session 4 strawman).
- Models page — read-only or interactive.
- Notes — Project / per-Action / per-Region.
- Result generation — auto-live vs explicit finalize.

Plus a few new ones from this session:
- **Action status states** — what does the lifecycle look like? `queued → running → complete | failed | cancelled`?
- **What happens during a long-running Action?** (Anomaly detection on a full HSI scene could take seconds to minutes.) Progress bar, log stream, or just a spinner?
- **Action Output viewer is per-Action-type.** Anomaly detection's viewer is a heatmap; cloud masking's is a mask overlay; rendering's is an image; spectral detection's is a spectrum + flagged regions. Each type has its own viewer component.

---

## 2026-05-08 — Session 8: Backlog clearing + live tile proposal

### Locked

**Models page = read-only Model Catalog.**
Surfaces the available models with **architectures and explanations** — what the model is, what it does, key references / metrics. Not interactive (no swap-from-here, no edit). Selecting a model for an Action happens inside the Action's config dialog, not from this page. The Models page is essentially a reference / educational view of what the platform can run.

Implications:
- The catalog needs a way to show architecture diagrams. Could be pre-rendered SVG/PNG per model (simple), or a small generated diagram from a layer manifest (over-engineered for v1). Default: pre-rendered images shipped with the model entry.
- Each model card has: name, version, architecture diagram, plain-language description, training data summary, performance metrics, intended uses & limitations (the "model card" pattern from ML practice — fits the audience perfectly).

**Notes — Project-owned, referencing.**
Notes belong to the **Project**. Their *content* can reference Actions and the Scene. So:
- Storage: `notes` table FK to `project_id`. Content is text/markdown. References to Actions (and the Scene) are inline links inside the note content.
- A note isn't pinned to a region or a specific Action — it's a free-form project-level scratchpad that can mention things via inline references.
- Reference syntax: TBD; could be `@action/<short-id>` or `[[Action: name]]` or a UI affordance (slash-command, picker). Defer the syntax decision; the data model is what matters now.
- The notes panel in the Analyse workspace shows the project's notes; clicking a referenced Action from inside a note jumps to that Action's Output viewer.

**Result generation — auto-live.**
The Result always reflects the project's **current state**. No "finalize" / "publish" button. Implications:
- Simpler: no `result_status` column, no snapshot freeze.
- The "Results" sidebar = list of all Projects (each Project's current state *is* its Result). Probably named "Results" but that's just the export-oriented view of the same projects.
- **Export** is a separate action from "having a Result" — there's a button that bundles the current Project state into a downloadable package. The package itself is a snapshot at export-time, but the on-screen Result is always live.

**Action status — lifecycle + polling.**
- Lifecycle states (provisional): `queued → running → complete | failed | cancelled`.
- Frontend **polls** for status. No WebSockets in v1. Simpler stack, fits Postgres-backed status well.
- Polling cadence: ~1s while any Action in the visible Project is `queued` or `running`; back off to ~5s otherwise. Polling stops when the user navigates away. Can revisit if 1s feels heavy.
- Long-running Action UX: progress bar where we can compute progress (e.g. tile-based inference: tiles done / total), spinner + elapsed-time clock + most-recent-log-line where we can't. Avoid pure spinners with no information — they erode "attention to detail."

### Live tiles on the Home landing — concrete proposal

User asked for a suggestion. Below is a six-tile layout, one tile per non-Home sidebar destination, each clickable to navigate. Tiles are designed as **instrument-console panels**, not SaaS feature cards: dense, real-data, monospaced numbers where it fits, no gradients, no marketing copy.

The aesthetic test for every tile: would this fit on the boot screen of a ground station? If yes, keep. If it would fit on a SaaS landing page, redesign.

**Tile 1 — Library** *(→ Scenes)*
- Top: total scene count, big monospaced number.
- Below: per-sensor breakdown — `PRISMA · 4   LANDSAT · 7   ENMAP · 2`
- Right edge: a vertical stack of 3–4 tiny thumbnail crops of the most recently onboarded scenes, with sensor-color tag.
- Empty state: "No scenes onboarded. Open Ingest to begin."

**Tile 2 — Ingest** *(→ Ingest)*
- Top: most recent onboarding event — sensor type, scene name, relative timestamp (`EnMAP · S2A_…_2024-05-12 · 12 min ago`).
- If an onboarding is in progress: replace with the active progress (`Ingesting EnMAP folder · 64% · 2:14 elapsed`).
- Below: small running counter — "Onboarded today: 3 · this week: 11."

**Tile 3 — Projects** *(→ Analyse)*
- Top: count of active projects.
- Middle: the most recent project — project name, scene, "last activity 6 min ago."
- If an Action is currently running on any project: a one-line callout — `▶ Anomaly Detection · PRISMA_2024_05 · 0:42 elapsed` (this is the "the system is doing something right now" signal).

**Tile 4 — Results** *(→ Results)*
- Top: count of projects with completed Action Outputs (i.e. anything exportable).
- Middle: a horizontal strip of 3 small thumbnail crops of the most recent result imagery (false-color, anomaly heatmap, rendered band combo) — these are *real product output*, not stock images. Tells the audience instantly that the platform produces visual artifacts.

**Tile 5 — Model Catalog** *(→ Models)*
- Top: number of models in the catalog.
- Middle: model names list (top 4 by recency-of-use), each with a one-token tag (`SegFormer-MAE`, `Iso-Forest`, `RX-Detector`, etc.).
- Subtle, no logos, monospace.

**Tile 6 — Workload** *(→ Monitoring)*
- Different layer than the top-bar sparklines (which are *host* CPU/GPU/RAM). This tile shows **workload metrics**: inference throughput (e.g. patches/sec or scenes/hr), Action queue depth, average Action duration today.
- Tiny inline sparkline next to throughput.
- Distinct from the top bar so we're not duplicating info — top bar = "is the machine alive," this tile = "is the platform doing work."

Why this set works for the audience:
- Every tile shows **real numbers tied to real work** — no decorative content.
- The active-Action callout in Tile 3 + the workload sparkline in Tile 6 = visible proof of liveness on every visit to Home.
- Together they communicate: scenes onboarded, work in progress, work completed, models available, system alive — the whole product surface area, on one screen, without explaining anything.

To react to: the six-tile count, the Workload tile being separate from the top bar, the per-sensor breakdown style, and whether thumbnails on Library / Results are crops of actual scene imagery vs. simple icons.

### Carry-forward (smaller now)
- Reference syntax for Notes (Action / Scene mentions).
- Action-Output viewer components per Action type.
- Empty states across the app, especially for first launch.
- Whether the Workload tile metrics need a backend instrumentation pass to expose.

### User decision (2026-05-08)
Tiles proposal accepted as-is. Treated as **a not-one-way-door decision** — we can revisit the tile content / count later if something feels off when we see it on screen. Moving on.

---

## 2026-05-08 — Session 9: Walking into the Scenes space

User direction: **walk into the Scenes space first** (before Analyse). Good sequencing — Scenes is upstream of Analyse (you need a scene before you can start a project on one), and it's where the library/book metaphor from Session 6 lives.

### Framing carried in from earlier sessions

- The **Scenes** sidebar is the view into the **Scene Library**.
- The Library is a global, persistent collection of all onboarded scenes — *"like books in a virtual library; the library grows as users onboard scenes."*
- Three sensor types in v1: **PRISMA** (hyperspectral HE5), **Landsat 9** (thermal TIF), **EnMAP** (hyperspectral GeoTIFF folder + XML metadata).
- The aesthetic anchor is **branded utility** + **instrument console**, not SaaS list view.
- Selecting a scene from this space is what enables creating a Project on it — so this space is the launchpad into investigation.

### What I'm not doing
Not proposing a finished layout. Letting the user describe what they see and feel when entering this space.

### Framing questions to drive the description
Pick whichever you want to start with:

1. **What's the first thing the user sees when they click "Scenes" in the sidebar?** A literal bookshelf-style view (spines, cards in rows)? A dense catalog table with metadata columns? A map of scene footprints? Or something else entirely?

2. **What's the primary thing a user does on this page?**
   - (a) **Launchpad mode:** browse and click → open a scene to start or open a Project on it. The Library is mostly a way to get into Analyse.
   - (b) **Viewer mode:** browse and click → enter a full scene viewer (zoom/pan, band selection, false-color rendering) without needing a Project. The Library is a destination in itself.
   - (c) **Both:** open a scene-detail view that has metadata, a viewer pane, and project history (existing Projects on this scene + a button to start a new one).

3. **What metadata do you want on the listing itself?** The defaults are cheap (sensor, name, ingest date, file size). The richer ones cost more but read very serious to the audience: cloud cover %, valid-pixel %, geographic footprint, acquisition date, band count, projection. Which of these matter? Which are noise?

4. **How literal is the book metaphor in the visual?** Three points on a spectrum:
   - Lightly metaphoric — informs the layout language (rows, spines, metadata "on the side") without being skeuomorphic.
   - Strongly metaphoric — visible spines, cover-thumbnail, "shelf" sections by sensor.
   - Not metaphoric at all — the book framing was conceptual; the page is a clean catalog.

5. **Filtering / search.** Sensor-type filter is the bare minimum (3 sensors, three buttons). What else does the user want to slice by? Date range, geographic area, cloud cover, "scenes I've started a project on"?

(Step in wherever feels right. The most consequential one for screen shape is probably (1) and (2).)

### Locked from this session

**Scenes landing = a table.**
- Rows = onboarded scenes.
- Columns include at minimum: name, sensor/product, some metadata, a small thumbnail.
- *Specific* column choice is **deferred** — onboarding (a future session) will produce a rich scene-metadata object, and we'll pick listing columns from it then. Don't decide on columns yet.
- Not a bookshelf. Not a card grid. Not a map. A table. The book metaphor stays **conceptual** — it informs how we *think* about the Library (a growing collection that has identity), not how it looks.

**Clicking a row opens a Scene Detail page.** The detail page must contain:
1. **Visuals of the scene** — content TBD (deferred: *"fill in the blanks here later"*). This is the natural home for a scene viewer (zoom/pan, band selection, false-color rendering), but we're not designing it now.
2. **A list of Projects associated with this Scene** — every Project that has this Scene bound to it. This is how the user re-enters prior investigations on the same scene.
3. **A button to create a new Project** with this Scene as the basis. This is the primary on-ramp from the Library into Analyse.

Layout of the detail page is also deferred until visuals are designed.

**Deep advanced filtering is a first-class feature of the Scenes page.**
User emphasized this explicitly. Implications:
- The filter surface needs to be powerful — not just "filter by sensor." Likely faceted across many dimensions (sensor type, date ranges, geographic bounds, cloud cover, valid-pixel %, band count, projection, plus eventually anything else the metadata object exposes).
- Filter UI shape (persistent left rail vs. expandable drawer vs. top-row chip bar) is **deferred** — but we should size it as a real UX surface, not a half-measure.
- Saved filter sets / quick-filters could matter to operational users (e.g. "all scenes with <20% cloud over the last 30 days"). Worth flagging as a possible v1 affordance; defer.

**Implication for backend (worth pinning early):**
- The scenes table will need many indexed columns to support filtering performance.
- The metadata object captured at onboarding **drives the entire filter surface**. So onboarding's metadata-extraction step is upstream-load-bearing: skimping there caps how powerful filtering can be.

### What we are NOT deciding right now (deliberately deferred)
- Listing column selection.
- Visual layout of the Scene Detail page.
- The scene viewer (the "visuals") — content, interactions, controls.
- The full filter UI shape.
- Saved filter sets / quick-filters.

These all funnel into the **onboarding session** — once we know what metadata onboarding produces, listing columns, filter facets, and detail visuals all follow naturally.

### Questions for the user

1. **Create-Project button — what happens on click?**
   - (a) Jump straight into a new Project workspace, scene auto-bound, project gets a default name (`<scene_name> · <timestamp>`).
   - (b) Small dialog first: name + optional description, then enter the workspace.
   - (c) Stay on the scene detail page, just append the new project to the projects list, user navigates to it manually.
   My lean is (b) — naming the project up front pays off later when the projects list on the detail page has 5+ rows.

2. **Filtering UI shape — first instinct?** Persistent left rail (always visible, lots of facets), expandable drawer (hidden until invoked), top-row chip bar (lightweight)? You can defer this if you want, but a first instinct is useful.

3. **Anything else on the Scenes landing besides the table?** A header counter ("`Library · 13 scenes · 2 added today`"), a global search bar across all metadata, an ingest shortcut button — or strictly just the table?

### User answers (this session)
- **Q1: (b) — dialog first.** Locked.
- **Q2: defer.** Filter UI shape postponed.
- **Q3: nothing else on the landing for now.** Just the table.

### New requirement — Annotations (optional, scene-associated)

User added: **scenes can have optional associated annotations onboarded alongside the core scene** — ground truth being the explicit example.

This is a significant addition. Capturing carefully because it shows up everywhere downstream.

**Vocabulary update:**
- **Annotation** — an optional artifact associated with a Scene that carries human-supplied or externally-supplied information about the scene's content. Ground truth (per-pixel labels, per-region labels, etc.) is the canonical example. A Scene may have **zero, one, or many** Annotations.

**Data model update:**
```
Scene 1—N Annotation        (zero or more)
```
- Annotations live in the same data volume as the scene, alongside the scene's files.
- Annotations table: FK `scene_id`, type, file path(s), ingest metadata, optional source/labeler info.
- Specific annotation **types** (raster mask, vector polygon, point list, spectral library reference, etc.) — defer to the onboarding session. The data model just needs to be type-aware.

**Why this is load-bearing for the demo:**
- Ground truth turns Action Outputs from "predictions" into **evaluatable predictions**. An Action Output compared against an Annotation = precision, recall, IoU, false-alarm rate, ROC. That's the language operational + commercial audiences trust.
- It's a massive credibility multiplier without changing the v1 Action catalog. Existing Actions stay the same; the *output viewer* gains an "overlay ground truth" toggle and a metrics panel.
- It also gives the "fund more compute for sophisticated models" line concrete teeth: the audience watches the user run two models on the same scene with the same ground truth, sees the difference numerically, and connects "more compute = better numbers." Hard to argue against that.

**Implications across the storyboard (capturing, will revisit):**
- **Onboarding (Ingest):** must accept optional annotation files in addition to the core scene file(s). Probably as a second optional step in the onboarding flow ("Attach annotations? [Skip / Upload]").
- **Scene Detail page:** the visuals area needs a way to overlay annotations on the scene. The metadata block needs an "Annotations: 2" indicator. List of attached annotations with type/source.
- **Scenes table (Library):** worth adding an annotation indicator column — even just `✓` or `✗` if you'd rather keep it minimal. Strong signal for filtering ("show only scenes with ground truth").
- **Analyse / Actions:** Action Output viewers should be able to overlay ground truth where present, and (optionally) compute evaluation metrics against it. This may justify a future Action type ("Evaluate") in v2 — but for v1, ground-truth comparison can live inside existing Output viewers as a built-in feature, not a separate Action.
- **Filter facets:** "has annotations" / "has ground truth" becomes a key filter facet on the Library.

### Questions for the user (about annotations)

1. **What kinds of annotations should v1 support?** Some candidates:
   - Raster mask aligned to the scene (binary or multi-class).
   - Vector polygons (shapefile / GeoJSON).
   - Point list (CSV / GeoJSON).
   - Spectral library / reference signatures.
   - Free-form metadata document (JSON).
   Pick the must-haves; we can defer the rest. My lean: raster mask + GeoJSON polygons cover ~95% of HSI ground-truth use cases.

2. **How load-bearing should annotations be in v1?**
   - (a) **Display only** — overlaid on the scene viewer for context. No metric computation.
   - (b) **Display + evaluation** — Output viewers can overlay them and compute precision/recall/IoU against an Action Output. The big credibility win.
   - (c) **Display + evaluation + Action input** — annotations can be consumed by Actions as input (e.g. "use these spectral signatures as the reference for spectral detection"). The most powerful, also the most build.
   My lean: (b) for v1. (c) is on the roadmap but probably v2.

3. **Multiple annotations per scene — supported?** Single ground truth file or many? (E.g. mask + polygon set, or two different labelers' opinions.) Cheap to allow at the schema level even if v1 UI only shows one at a time.

### User answers (annotations)
- **Annotation types:** raster mask / ground truth file is the v1 starting point. Specific format details deferred.
- **Use:** annotations are **combined with the core scene to produce visuals on the Scene Detail page**. So at minimum: visualization overlay. Whether they also drive evaluation metrics (precision/recall/IoU panels) is **not yet locked** — captured as a follow-up.
- **Multiple per scene:** yes, supported.

---

## 2026-05-08 — Session 10: Ingest moves into the Scenes page

User-led decision: **"Ingest is a button in the Scenes page, not a separate sidebar item."**

This changes the sidebar shape we locked in Session 4. It's the right call — Ingest is *the verb that produces a row in the Library*, so the entry point belongs where the Library lives.

### Sidebar — revised

```
1. Home
2. Scenes        (now also hosts the Ingest button)
3. Analyse
4. Results
5. Models
6. Monitoring
```

(Six items including Home, down from seven. Cleaner, and the sidebar is now a list of *destinations* rather than mixing verbs and nouns.)

### Implications

**Scenes landing now has at least one non-table element.**
Earlier in Session 9 the user said "we are good for now" on additional landing elements — that referred to things like header counters and search. The Ingest button is a different category (it's not decoration, it's a primary action). It should sit in a header / toolbar area above or beside the table.

**Live tile #2 ("Ingest") on the Home landing — fate?**
The tile still earns its place: it surfaces recent onboarding activity ("EnMAP folder · 12 min ago") and shows in-progress ingestions. But it should now **navigate to Scenes** (where the Ingest button lives), not to a now-nonexistent Ingest page. The tile may also want a small inline "Onboard new" affordance so it doubles as a shortcut.

Three options for the Home tile:
- (a) Keep as-is, navigation routes to Scenes. Simplest.
- (b) Keep but rename — e.g. "Recent ingest" — to make clear it's an activity tile, not a destination.
- (c) Drop it, since there's no destination it can stand for. Surface recent ingest events as part of the Library tile instead.
   My lean is (a) — the tile's content is meaningful and still tile-shaped, and the audience won't notice that it now navigates to Scenes rather than to its own page.

**Onboarding flow entry point.**
Clicking the Ingest button on the Scenes page opens the onboarding flow. The shape of that flow (modal / wizard / side panel / dedicated page) is the **subject of a future onboarding-flow session.** For now we just know the entry point.

**Annotations come along for the ride.**
Onboarding handles core scene + optional annotations together. We can confirm UX details when we design the onboarding flow.

### Open follow-ups (carry these forward — these are now explicitly DEFERRED to implementation)

1. **Ingest button placement** on the Scenes page.
2. **Onboarding flow shape** — modal / wizard / page.
3. **Evaluation metrics with annotations** (precision/recall/IoU panels). Visualization overlay is locked; metric computation is not.
4. **Home Ingest tile** — keep / rename / drop.

### Working-altitude correction (locked for the rest of the storyboard)
User: *"We will color in details as we implement. Right now we are defining broad structure and abstractions, so let us board at that level."*

From here on, storyboard sessions stop drilling into UX micro-details (button placement, tile variants, polling cadence, modal-vs-wizard) and stay at the abstraction level: sidebar destinations, the major abstractions each screen contains, vocabulary, data model. Detail decisions get noted as deferred and revisited during implementation.

---

## 2026-05-08 — Session 11: Sidebar collapse — Analyse + Results → Projects

User-led restructuring: **"There is only Projects in the sidebar. Results come from actions, and Analysis is basically subsumed in the concept of actions."**

This is a clean simplification. Capturing the consequences.

### Sidebar — revised (final at this stage)

```
1. Home
2. Scenes        (Library; hosts Ingest button)
3. Projects      (replaces Analyse + Results — investigations live here)
4. Models
5. Monitoring
```

Five destinations including Home. Settings stays on the top bar.

### Why this is structurally cleaner

- **Result is a property of a Project** (auto-live, 1:1 from Session 8). Giving it its own destination duplicated information already inside the Project.
- **Action is the unit of work inside a Project** (Session 7). "Analyse" was just the name of the workspace where Actions happen — but the workspace *is* the Project. Two names for one thing was redundant.
- The sidebar now reads as a list of pure destinations:
  - Scenes = the Library (data in)
  - Projects = the workspaces (work being done)
  - Models = the catalog (capabilities)
  - Monitoring = system state
  - Home = orientation across all of the above

### What this changes (structural)

- **The "Project workspace"** (formerly the Analyse workspace) is what you see when you open a Project from the Projects landing. It contains *everything* investigation-related: scene context, list of Actions, Action Output viewers, Notes, and the live Result view.
- **The Project Result is surfaced inside the Project workspace** — as a panel / tab / view (specifics deferred to implementation). It's not a separate destination.
- **Export** (the bundled, downloadable Result snapshot) is a button inside the Project, not a separate page.

### What this changes for the Home live tiles

The tile set from Session 8 needs trimming. New shape (still at abstraction level — content details deferred):

```
1. Library tile       (→ Scenes)
2. Ingest activity tile  (still earns its place; navigates to Scenes)
3. Projects tile      (→ Projects; subsumes the old Projects + Results tiles —
                         shows active projects, in-progress Actions, recent Outputs)
4. Model Catalog tile (→ Models)
5. Workload tile      (→ Monitoring)
```

Five tiles. The Projects tile now does the work of two: surfaces both *what's being investigated* and *what has been produced*. This is a better fit anyway — the old "Results" tile was a slightly forced split.

### Vocabulary status

- **"Analyse"** retired as a sidebar / destination word. The *activity* of analysing is preserved — it just happens inside a Project, via Actions.
- **"Result"** retained as a concept (the project-level rollup, auto-live). Just doesn't have its own destination.
- **"Project"** is now the most prominent word in the product. It's where the user spends most of their time, and it's where everything happens.

### Implications worth pinning early

- **Postgres schema unchanged** by this restructure — Project, Action, ActionOutput, Result, Note tables stay exactly as designed in Session 6. We were only ever collapsing the *navigation*, not the data model. (Worth noting because it would have been a red flag if a UX simplification rippled into schema changes.)
- **API surface largely unchanged** — there's no separate `/results` namespace; results are nested under projects (`/projects/{id}/result`).
- **Compose / images / bundle** all unchanged. Backend doesn't care that the sidebar shrank.

### What still needs walking at structure level
- **Projects destination** (the most consequential remaining walk — landing list + project workspace shape).
- **Models destination** (lighter — read-only catalog).
- **Monitoring destination** (lighter — system + workload metrics).

---

## 2026-05-08 — Session 12: Walking into Projects (structure level)

The Projects destination is the heart of the product. It's where the demo spends most of its 30 minutes, and it's where every abstraction we've defined comes together.

### Two views inside this destination

This destination has two distinct screens. Naming them up front so we know what we're walking through:

1. **Projects landing** — the list of all Projects across the Library. Reached by clicking *Projects* in the sidebar.
2. **Project workspace** — what you see when you open a single Project. The investigation environment.

### Abstractions that must live somewhere in this destination

Pulling from earlier sessions — these all need to surface inside Projects. We're not deciding *where* on the screen yet, just acknowledging the cast:

- **Scene context** — which Scene this Project is bound to, key metadata, base imagery for visuals.
- **Annotations** — optional, overlay-able on the scene visuals.
- **Action list** — the sequential list of Actions created in this Project, each with status (queued / running / complete / failed / cancelled).
- **+ New Action** affordance — open a small flow that picks an Action type, picks input(s), configures, runs.
- **Action Output viewers** — per-Action-type viewers (anomaly heatmap, cloud mask, spectral signature plot, rendered image, etc.).
- **Notes** — project-owned, can reference Actions and the Scene.
- **Result** — auto-live, project-level rollup. Surfaced inside the workspace (panel / tab / view — TBD).
- **Export** — button that bundles the current Result state into a downloadable package.

### What we already know about Project structure

- A Project is bound to exactly one Scene.
- A Scene can host multiple Projects.
- Action ordering is sequential / notebook-style (Session 7 — *not* a DAG editor).
- Result is auto-live, exists 1:1 with the Project, exported on demand.
- Status comes from polling the api (Session 8); workers process Actions off the Postgres queue (Session 4 of final design.md).

### What's open at structure level (not detail level)

The user drives these. Floor's open — pick whichever order:

1. **Projects landing — basic shape.** A table (like Scenes), a card grid, a list grouped by scene, or something else? What does a row/card represent (project name, scene, status of latest Action, last activity)?

2. **Project workspace — the major regions.** At the abstraction level: how many distinct regions does the workspace have, and which abstractions live in which region? E.g. "scene + visuals dominate one region; Actions list down one side; Notes in another region; Result is a separate tab/view." We're not designing pixels — we're agreeing on the structural skeleton.

3. **Comparison across Actions inside a Project.** Multiple Actions on the same scene begs for comparison (Session 5 implication). Is comparison a first-class structural feature (its own region / mode), or is it implicit (you just open two Action Outputs side-by-side)?

4. **Where does the Result surface inside the workspace?** A persistent panel, a tab, a separate view you switch to? It's auto-live, so it's always available — but it doesn't need to dominate the screen most of the time.

5. **Project lifecycle states (if any).** Projects might just exist (no states), or they might have an "active / archived" notion. Affects the landing.

(Pick whichever you want to describe first — or just walk through the experience in your own words.)

### Locked from this session

**Projects landing — structurally a table.**
- Columns at minimum: project name, associated Scene.
- A **New Project** button opens a dialog that lets the user choose a Scene; on confirm, the user lands in the new Project's workspace.
- (This is consistent with the Scene Detail page's "Create Project" button from Session 9 — both flows funnel through the same dialog. Two entry points, one canonical creation flow.)

**Project workspace — two-region structure (at minimum).**
- **One pane: Action list + new-Action controls.** The list of Actions performed in this Project, plus the affordance to create a new Action.
- **Another part of the screen: Action detail.** Clicking an Action shows its details (configuration + Output viewer) **in place** — no new window, no navigation. Single workspace, switching happens within it.
- The two regions plus the persistent Result panel (below) are the workspace's structural skeleton. Other abstractions (Scene context, Annotations overlay, Notes) need homes — those are still open.

**Result — persistent panel inside the workspace.** Locked.

**No comparison feature.** Multiple Actions on the same scene do not get a first-class comparison surface. If users want to compare, they switch between Actions in the action list.

**Project lifecycle — just exist, plus deletion.** No active/archived/finalized states.

### New abstraction — Visualization Template

User-introduced: **"a visualization template which can operate on a scene or on an action result."**

This is a meaningful new concept. Capturing carefully.

**Definition:** A **Visualization Template** is a reusable, configurable rendering specification. It defines *how* something is presented (band combination, colormap, threshold, overlay style, plot style, etc.) and can be applied to either:
- A **Scene** (e.g. SWIR-NIR-Red false-color of a hyperspectral cube), or
- An **Action Output** (e.g. anomaly heatmap with a specific colormap + threshold + opacity).

**Why this is structurally important:**
- Decouples *what data is* from *how it's presented*. Same Scene can be viewed many ways; same Output can be rendered many ways. No re-running of Actions to change presentation.
- Standardises looks across Projects — consistent visual language for Results, exports, briefings.
- Becomes a curatable surface: the user (or the platform) can build up a library of templates over time.
- For the audience: signals that the platform thinks in terms of *workflows* and *reuse*, not one-off visualizations. Operational/commercial reads as "this team has built infrastructure."

**Data model addition (provisional):**
```
VisualizationTemplate
  - id, name, description
  - input_type   (scene | action_output)
  - applicable_to  (sensor types for scenes; Action types for outputs)
  - configuration JSONB  (band assignments, colormap, threshold, overlay opts, etc.)
```
- Cardinality: a template is applied to many things; a thing can be viewed under many templates. No FK relationship; the template is a presentation function, not owned by any single Scene or Action.

**Conceptual relationship to the "Rendering" Action (clarification worth pinning):**
- Both produce visuals from a Scene. Easy to confuse; they're complementary.
- **Visualization Template** = applied at *view time*. Ephemeral. No stored artifact. Cheap. The user toggles between templates in the Scene Detail page or Action Output viewer.
- **Rendering Action** = produces a stored Action Output (a rendered image artifact, possibly using a template). Persistent. Goes into the Result. Used when you want to *capture* a specific rendering as part of the investigation record.
- One way to think of it: a Rendering Action is a Template "frozen" into an output for inclusion in the Result.

**Where templates show up in the UI (structural):**
- **Scene Detail page** (Session 9, deferred visuals) — the template-picker is what populates the visuals area. Same Scene, swap templates, see different views.
- **Action Output viewer** in the Project workspace — same idea. The viewer for an anomaly heatmap may have multiple templates available (default, threshold-tight, threshold-loose, custom).
- **Template management surface** — open question (see below).

### Open questions about Visualization Templates (structure-level only)

1. **Scope: global, project-scoped, or both?** A global library (shared across all projects) signals reuse and standardization. Project-scoped lets a single investigation curate its own looks. Both is most flexible. My lean: **global library** for v1, project-scoped if the demand emerges.

2. **Where does template management live?** The sidebar is locked at five destinations and we're not adding a sixth. Plausible homes:
   - Inside **Settings** (top bar) — fits if templates are a quiet curation surface that the user touches occasionally.
   - Implicit / in-context — templates are created and edited via a dialog wherever they're applied (Scene Detail page, Output viewer). No dedicated management page.
   - A sub-area of **Models** (since both are "capability catalogs"). Risky — Models is read-only and Templates are user-editable; conflating them muddies the abstraction.

3. **Are templates typed by Action type?** A heatmap template wouldn't apply to a cloud mask. Probably yes — the data model carries `input_type` + `applicable_to` so the picker only offers compatible templates for what's currently being viewed.

### Sidebar status (unchanged)
```
Home · Scenes · Projects · Models · Monitoring
```
Templates do not get a sidebar destination. They live inside the surfaces where data is viewed (and possibly Settings for management).

### Implications for the storyboard so far
- **Scene Detail page** now has a clearer skeleton: the deferred "visuals" block = a template-driven render of the scene + an annotation overlay (where annotations exist) + a template picker.
- **Action Output viewer** also gains a template picker. The "per-Action-type viewer" framing from Session 7 still holds, but templates make those viewers parameterized rather than hardcoded.
- **Sidebar still 5 items.** Templates do not add a destination.

### Carry-forwards
- Template scope (global / project / both).
- Template management surface (Settings / in-context / elsewhere).
- The exact set of v1 template types (band-combo for hyperspectral, thermal stretch for Landsat, anomaly heatmap, mask overlay, etc. — defer to implementation).

---

## 2026-05-08 — Session 13: Correction — Rendering Action consumes input + Template

User correction (verbatim): *"I think a rendering action takes action output and applies a template to it. am I wrong?"*

User is right. My Session 12 framing of Rendering as a "Template frozen into an output" was loose. The clean structural relationship:

### Rendering Action — corrected definition

- **Input:** a data source (a **Scene** or an **Action Output**) + a **Visualization Template**.
- **Output:** a persistent rendered image artifact, stored as a regular Action Output.
- **Effect:** captures a specific Template-rendered view for inclusion in the Result, exports, briefings.

The "Scene OR Action Output" input duality matches the Template's own duality (Templates already operate on either). It also avoids an awkward edge case: capturing a false-color render of the raw scene has no path if Rendering only accepts Action Outputs as input — you'd have to wrap it in a no-op Action first.

### Visualization Templates serve two roles, same definition

1. **View-time presentation** — applied in the Scene Detail page or Output viewer. Lightweight, ephemeral, no stored artifact. The user swaps Templates to explore.
2. **Rendering Action input** — the same Template fed into a Rendering Action produces a persistent, stored render.

This is the elegant part: one definition of a Template, two contexts of use. Once a user finds a view-time Template they like, capturing it persistently is just "create a Rendering Action with this Template."

### Action input model — generalized

This refines what an Action's "input" is. From Session 7 we said inputs could be a Scene or a prior Action Output. The Rendering Action makes clear that **Action configuration** is a structured spec, not just a single input pointer:

```
Action
  - type           (anomaly_detection | spectral_detection | cloud_mask | rendering)
  - input_data     (scene_id | action_output_id | both, depending on type)
  - configuration  (type-specific: e.g. algorithms list for anomaly_detection,
                                    template_id for rendering, threshold for cloud_mask)
```

The Template, when used by a Rendering Action, is just one field within `configuration` — typed and validated by Action type. This keeps the data model clean: Templates aren't a special FK on Actions; they're referenced from inside the configuration JSON for Action types that use them.

### What this changes (or doesn't)

- **No vocabulary additions.** Same words: Action, Action Output, Visualization Template.
- **No sidebar changes.** Still five.
- **Schema impact: minimal.** Templates table is as in Session 12. Actions reference Templates only inside their configuration JSON, no FK gymnastics.
- **Storyboard impact: small.** Session 12's claim "Rendering Action = Template 'frozen' into an output" is replaced by the cleaner "Rendering Action = input + Template → stored render."

### Pending confirmation
- Rendering Action input can be **Scene** or **Action Output** (not only Action Output). Asked the user; awaiting confirmation.

---

## 2026-05-08 — Session 14: Untangling Result, Action Output, Visualization

User correction: **"We need to [not] confuse the concept of Result and Action Output. Yes a Rendering Action produces a visualization result, but it is up to us whether we want to keep it as part of the project or let it go. Projects are thus a big tent for actions and the visualizations which may or may not be related to specific actions."**

### Three concepts, distinct

- **Action Output** — the artifact an Action produces. Every Action has one. Owned by the Action, stored, addressable. (anomaly mask, cloud mask, rendered image, spectral plot data, etc.)
- **Visualization** — a first-class project-level item. **Curated** by the user. May *come from* a Rendering Action's Output (with an explicit keep step), or possibly other paths (open question). Visualizations are *not* automatically created from every Output.
- **Result** — the project-level rollup. Auto-live, 1:1 with the Project (Session 8 lock unchanged). Composed from Actions + Visualizations + Notes.

### Project as a "big tent"

Per user phrasing, a Project contains **two parallel collections**:

```
Project
├── Actions          (the work performed)
│     └── each Action has an Action Output
├── Visualizations   (curated visuals, may or may not be tied to specific Actions)
└── Notes            (project-owned, can reference both)
```

Crucially: **Visualizations are not a derivative view of Actions.** They are their own collection. Some Visualizations are produced by Rendering Actions and then "kept"; some may have other origins (open question).

### Project workspace — refined surfaces

The workspace must surface:
- **Actions list** + + New Action affordance (Session 12 lock).
- **Action detail pane** — when an Action is selected, its config + Output viewer appear in place (Session 12 lock).
- **Visualizations collection** — the curated visuals. New surface added by this session.
- **Notes** — somewhere TBD.
- **Result panel** — persistent, live (Session 12 lock).

The workspace skeleton is now **at minimum three regions** (Action list, Action detail, Result panel) plus the Visualizations collection and Notes living somewhere among them. Exact layout deferred to implementation per the working-altitude rule (Session 10).

### Data model addition (provisional)

```
Visualization
  - id, project_id (FK), name
  - source_action_output_id  (FK, NULLABLE — where this came from, if a Rendering Action)
  - source_template_id       (FK, NULLABLE — the Template used)
  - source_scene_id          (FK, NULLABLE — for view-time saves directly off a Scene, if supported)
  - artifact_path            (the rendered image file)
  - created_at
```

The nullable FKs let a Visualization carry provenance without requiring exactly one source — fits the "may or may not be related to specific actions" framing.

### Result rollup composition

The Result now composes from three sources, not one:
- All Actions in the Project (and their Outputs, accessible by reference).
- All curated Visualizations in the Project.
- All Notes in the Project.

This is consistent with the auto-live framing — the Result reflects the project's current state across all three collections.

### Open questions (structure-level)

1. **What does "let it go" mean operationally?**
   - (a) The Rendering Action's Output is discarded (row + file deleted). Clean, no orphans.
   - (b) The Action Output always stays (invariant — every Action has one). "Letting go" means *not promoting* it to the Visualizations collection. The Output stays accessible via the Action's detail pane.
   - **My lean: (b).** Action Outputs are invariant; Visualizations are a curated layer on top. Matches the "big tent" framing better and lets the user change their mind.

2. **Visualization provenance — only from Rendering Actions, or also view-time saves?** A user applies a Template to a Scene in the viewer, likes the result, saves it as a Visualization — without running anything on the queue. Plausible and lightweight (no GPU work needed for simple Template renders). Or do all persistent visuals have to flow through a Rendering Action for consistency?

### What's unchanged
- Sidebar still 5 items.
- Vocabulary stays clean: Action, Action Output, Visualization, Result, Note, Template, Scene, Project — each with one definition.
- Schema additions are minimal: a `visualizations` table with nullable provenance FKs.

### User answers (locked)

**Q1 — "let it go" semantics: (b).**
Action Outputs are invariant. Every Action that completes has its Output, period. "Letting it go" means *not promoting* a Rendering Action's Output into the Visualizations collection. The Output remains accessible via the Action's detail pane; it just isn't featured in the curated Visualizations.

**Q2 — Visualization provenance: from a Rendering Action OR from a Scene.**

Two valid origin paths:
- **Heavy path: via a Rendering Action.** Rendering Action input is `Scene | ActionOutput` plus a Template; output is an Action Output that the user can promote to a Visualization. *(This also implicitly confirms Session 13's pending question — Rendering Action input can be Scene or Action Output.)*
- **Light path: directly from a Scene at view time.** User applies a Template to a Scene in the viewer, likes the result, saves it as a Visualization. No queue job, no Action — the rendered view is persisted directly into the Visualizations collection.

**Notably absent: there is no light path from an Action Output.** If a user wants to capture an anomaly heatmap (or any non-Rendering Action's Output) under a specific Template as a Visualization, they go through a Rendering Action. Rationale: heavy-data Outputs deserve a queue job for proper rendering; only Scene-direct view-time saves earn the lightweight shortcut.

### Implication for Rendering Action input — confirmed

Rendering Action input is `Scene | ActionOutput`, plus a Template. The Session 13 pending confirmation is now resolved.

### Status of the Projects destination

Structurally complete at this altitude. The workspace cast is locked:
- Actions list + new-Action affordance.
- Action detail pane (in-place).
- Visualizations collection (curated).
- Notes.
- Result panel (persistent, live).

Layout, navigation between regions, save-as-Visualization affordances, etc. — all deferred to implementation.

---

## 2026-05-08 — Session 15: Rendering removed from the Action catalog

User challenge: *"If an action is completed and produces a persistent output, why does rendering need to go through the queue? Or is rendering itself heavy enough to need compute time?"*

User is right. Working through it:

- Rendering an Action Output (heatmap, mask): colormap on a 2D array → milliseconds.
- Rendering a Scene (HSI → false-color RGB): read 3 bands, stretch, composite → seconds at worst, I/O bound, no GPU.
- Heavy work lives in the *Actions whose Output is being rendered*, not in rendering itself.

So a "Rendering Action" never genuinely needs the queue or the worker's GPU. Treating it as an Action conflates a lightweight presentation step with the queue-routed compute pattern. Cleaner to simply remove it from the Action catalog.

### Locked

**v1 Action catalog (revised — three items, not four):**
1. Anomaly detection
2. Spectral detection
3. Cloud masking

**Visualization creation — uniform path:**
- The user views *anything* (Scene or Action Output) under a Template in a viewer.
- "Save as Visualization" persists the (source, Template) pair + a cached rendered artifact.
- Source = Scene OR Action Output. **Same path either way.**
- Handled synchronously by the **api**. No queue, no worker. Returns in milliseconds.

**Asymmetry from Session 14 dissolves.** Earlier rule was "Visualizations from Action Outputs go through a Rendering Action; from Scenes go direct." That rule is gone. Both go direct. Uniform.

### What's unchanged

- **Worker image** still exists, still gets `--gpus all`, still handles the three remaining heavy Actions. Compose stack from final design.md Session 4 is unaffected.
- **Postgres-backed queue** is still load-bearing for the three Actions.
- **Visualizations data model** (Session 14) is unchanged — nullable source FKs already accommodate both source types. The provenance chain just shortens (no intermediate Rendering Action row).
- **Sidebar** still 5 items.
- **Workspace skeleton** still: Actions list, Action detail pane, Visualizations collection, Notes, Result panel.
- **Templates** still serve their two roles: view-time presentation and (when saved) configuration captured in the Visualization's source pair. The "Rendering Action input role" is now folded into the synchronous save path.

### If heavy rendering ever becomes a real need

(E.g. tiled WMS for gigapixel cubes, complex multi-overlay composites, video timelapses.) We add a `render` Action type at that point. The hooks are clean — Action types are extensible by design (Session 7's "Action catalog is the product surface area" principle). Don't pay for it now.

### Net effect on the storyboard

- Action catalog: 4 → 3. Cleaner; rendering wasn't a real compute Action.
- Visualization origin: 2 paths → 1 path. Cleaner; "from Scene" and "from Action Output" are now the same path with different source types.
- Architecture: unchanged.

---
