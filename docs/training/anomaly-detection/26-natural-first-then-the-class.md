# Part 26 - Naming materials: natural first, then the class

> **The one thing this part teaches:** the shipped family names were assigned by regular expressions on
> library file names, so they sort materials by use rather than by what they are made of and some are
> barely a degree apart; the redesign agreed on 2026-10-31 first asks "could nature explain this
> pixel?" and only then "which man-made class?", multiplies the two answers' probabilities, and
> matches pixels against library spectra instead of classifying them, so a new library does not need
> a retrain.

## Where we are

Part 15 built the family matcher: subtract a ring background, score whole families with ACE, calibrate
against the scene's own pixels, and name a man-made family only above `p*`. Part 25 closed the detection
experiments. This part goes back to naming, because on 2026-10-31 the user and Claude took the naming
step apart and agreed a different shape for it.

**Status (updated 2026-10-31, evening): designed, built, training.** This part is the *why*: the
families, the two levels, the matching idea and the classical baseline. The design was written up the
same day as `docs/proposals/spectral-namer.md` (agreed by the user: "Yes. Please on all three"), and
the model was **built** the same day in `app/spectral_match/namer/`. Its first run, N001, trains on an
H100 capacity block from 2026-10-31 11:30 UTC; **no results exist yet**. Part 27 teaches the model as
it was built, which departs from this part's design in a few places - the most important being how
level 1 is trained (see the note at the end of Step 3). Nothing here runs in the product: the namer
is not wired into `spectral_library_match`, by the proposal's own rule ("Product wiring ... only after
it passes"). The experiment README's heading "Design so far" is older than the proposal and is
superseded by it.

| file | what it holds |
|---|---|
| `scripts/curate_family_library.py` | how every library spectrum got its family: regular expressions on file names |
| `data/family_library/index.json` | the shipped library: 411 spectra, `per_chapter_counts` per family |
| `app/spectral_match/family_library.py` | `FAMILY_CAP = 40`, `balanced_subset`, `tag_library` |
| `app/spectral_match/family_match.py` | the shipped matcher (part 15) |
| `research/experiments/spectral-namer/separability.py` | the separability check: ACE, leave-one-spectrum-out, four fills |
| `research/experiments/spectral-namer/README.md` | the separability check: method, all results and the conclusions for the class list |
| `docs/proposals/spectral-namer.md` | the design written up, plus an "As built" list of where the code departs (part 27) |
| `app/spectral_match/namer/` | the namer itself (part 27) |
| `research/experiments/spectral-namer/separability_{prisma,enmap,aviris_ng}.json` | its results per sensor, with confusion counts |
| `research/math/spectral-matching/spectral_matching_step_by_step.tex` | the slow mathematics, including the family angles and Part VII on this design |
| `docs/proposals/spectral-matching-fresh-start.md` | the parked namer plan (2026-10-28) this redesign replaces in part |

---

## Step 1 - Where the ten man-made families came from

Open `scripts/curate_family_library.py`. It reads two published libraries straight from their zip files
(USGS splib07 and ECOSTRESS) and gives each spectrum a **group** (natural or man-made) and a **family**
(roofing, metal, ...). A **regular expression** (regex) is a text pattern: `r"asphalt|concrete"` matches
any name that contains "asphalt" or "concrete". The USGS man-made rules, in order:

```python
# USGS chapter A, first match wins; excludes first so "Iron_oxide" never reaches "metal".
CHAPTER_A_RULES = [
    ...                                                       # three exclusion rules
    (r"^oil|oiled", "manmade", "oil"),
    (r"sheet_?metal|aluminum|rusted_tin|steel", "manmade", "metal"),
    (r"asphalt_shingle|asphalt_tar|tar_paper|roofing_felt|roofng|roofing|clear_roof|cedar_shake",
     "manmade", "roofing"),
    (r"asphalt|concrete_gds375|brick.*pav", "manmade", "pavement"),
    (r"brick|cinder|concrete", "manmade", "building"),
    (r"plastic|polystyrene|tyvek|visqueen|fiberglass", "manmade", "plastic"),
    (r"fabric|carpet|webbing|pile|terrycloth|burlap", "manmade", "textile"),
    (r"wood|plywood|board|paper", "manmade", "wood_paper"),
]
```

"First match wins" means the list is read top to bottom and the first pattern that matches decides.
Trace three file names by hand:

```
"asphalt_shingle_..."   roofing rule matches first        -> roofing
"asphalt_road_..."      roofing rule fails ("asphalt_" then "road"), pavement rule matches -> pavement
"concrete_gds375"       pavement rule matches             -> pavement
"concrete_wall_..."     pavement rule fails, building rule matches                         -> building
```

The ECOSTRESS rules do the same on the second and third tokens of the file name
(`^concrete\.pavingconcrete` -> pavement, any other `^concrete` -> building). Three problems follow.

**Problem 1 - families by use, not by material.** Asphalt is in two families (roofing shingles and
road pavement), and concrete is split between pavement and building. A satellite sees what a surface is
made of, not what it is for. A concrete roof and a concrete road have nearly the same spectrum; the regex
puts them in different families because of a word in a lab technician's file name.

**Problem 2 - family sizes reflect the libraries, not what we need.** From
`data/family_library/index.json`, `per_chapter_counts`:

| man-made family | spectra | | natural family | spectra |
|---|---|---|---|---|
| building | 11 | | mineral | 40 |
| glass | **1** | | npv (dry plant matter) | 40 |
| metal | 13 | | rock | 40 |
| oil | 40 | | snow_ice | 20 |
| paint | 8 | | soil | 40 |
| pavement | 15 | | vegetation | 40 |
| plastic | 39 | | water | 15 |
| roofing | 27 | | | |
| textile | 14 | | | |
| wood_paper | 8 | | | |

Add them: man-made `11 + 1 + 13 + 40 + 8 + 15 + 39 + 27 + 14 + 8 = 176`; natural
`40 + 40 + 40 + 20 + 40 + 40 + 15 = 235`; total `176 + 235 = 411`, the `n_entries` in the same file.
The 40s are `FAMILY_CAP = 40` (`app/spectral_match/family_library.py`, line 27): families that had more
spectra were cut down to 40. Glass has one spectrum because only the ECOSTRESS rules look for glass
(`\.glas`) and a single glass spectrum passed them and the 95% grid-coverage test. Oil has 40 and glass 1, which says nothing about how often either appears from orbit.

**Problem 3 - some families are not separable.** The step-by-step book (chapter "Families as subspaces",
table `tab:angles`) computes the smallest angle between each pair of families, treating each family as
the 3-dimensional space its spectra span:

| pair | smallest angle |
|---|---|
| pavement vs building | 0.51 degrees |
| metal vs plastic | 2.04 degrees |
| metal vs roofing | 3.70 degrees |
| plastic vs textile | 4.53 degrees |
| paint vs plastic | 5.92 degrees |

What does 0.51 degrees mean? Its cosine is `cos(0.51 deg) = 0.99996`, and the part of a unit target's
energy that points outside the other family is `sin^2(0.51 deg) = 0.0089^2 = 0.00008`. **Pavement and
building share 99.99% of their energy: they are one material with two names.**

## Step 2 - What can be named from orbit at all

Before choosing classes, ask what fills enough of a pixel to leave a signal. The **fill** is the share of
a pixel's ground area a target covers (part 15's "30 % fill"). Take a car, 4.5 m by 1.8 m:

```
car area           4.5 x 1.8 = 8.1 m^2
30 m pixel         30 x 30   = 900 m^2       fill = 8.1 / 900 = 0.009   (about 1%)
5 m pixel          5 x 5     = 25 m^2        fill = 8.1 / 25  = 0.324   (about a third)
```

At PRISMA's and EnMAP's 30 m, **a vehicle is about 1% of a pixel**: only large objects (roofs, tank farms,
ships, tarpaulin-covered dumps, fields of solar panels, airstrips) can be named. AVIRIS-NG, flown at low
altitude, reaches about 5 m (the figure used in the discussion; it depends on flight height) and reaches
vehicle scale.

The discussion replaced the ten use-based families with a class list aimed at what a defence analyst
asks:

| class | why it is on the list | library spectra? (this course's reading) |
|---|---|---|
| camouflage | the first thing an analyst wants found | **none - needs field spectra** |
| military coatings | as above | **none - needs field spectra** |
| painted metal | vehicles, containers, roofs | yes |
| bare metal | aluminium, galvanised steel | yes |
| vessels on water | ships; part 21 kept water in scope for this | no single spectrum - paint or metal mixed with water |
| synthetic fabric / tarps | covered stores | yes |
| fuel / oil | spills, storage | yes |
| concrete / asphalt | one class, ending problem 1 | yes |
| disturbed earth | digging, tracks | from soils |
| cut vegetation | cleared or harvested plants | from vegetation and npv |
| glass / PV | solar farms, glazing | thin |
| none / natural | everything else | the bulk of every scene |

The two classes analysts care about most have no spectra in either library. That fact shapes Step 3.

## Step 3 - Two levels: natural first, then the class

The user's design splits naming into two questions:

```
             pixel (already flagged as anomalous)
                          |
         level 1:  could nature explain this?
              /                         \
          natural                    man-made
          (stop)                         |
                          level 2:  which man-made class?
                             /                    \
                 a class is confident        none is confident
                 "man-made: painted metal"   "man-made, class unknown"
```

Three possible answers: **natural**, **man-made, class unknown**, or **man-made plus a class**. The
probabilities multiply:

```
P(class) = P(man-made) x P(class | man-made)
```

`P(class | man-made)` reads "the probability of the class, given that the pixel is man-made". It is the
answer level 2 gives when level 1 is taken as settled. Do one on paper. Level 1 says
`P(man-made) = 0.9`; level 2 says painted metal 0.6, bare metal 0.25, tarp 0.15:

```
P(painted metal) = 0.9 x 0.6  = 0.54
P(bare metal)    = 0.9 x 0.25 = 0.225
P(tarp)          = 0.9 x 0.15 = 0.135
P(natural)       = 1 - 0.9    = 0.10
check: 0.54 + 0.225 + 0.135 + 0.10 = 1.00
```

Now a pixel with `P(man-made) = 0.3` and the same level-2 answer: painted metal is
`0.3 x 0.6 = 0.18`. A level-2 answer of 0.6 is worth 0.54 behind a confident level 1 and 0.18 behind a
doubtful one. **The product rule stops a confident-sounding class name from hiding a doubtful "is it even
man-made?".**

**Why level 1 is trained as "not explained by nature".** The obvious way to build level 1 is a two-class
classifier, natural against man-made, trained on library spectra of both. It fails for the most important
case: the man-made side would only know library materials, and camouflage is not in the library. A
camouflage net that resembles vegetation would be called natural.

The natural side is different. Every scene is mostly natural pixels, and the project holds billions of
them (SpectralEarth and the native EnMAP, PRISMA and AVIRIS-NG shards). So level 1 is trained to describe
**nature** as tightly as possible, and anything nature cannot explain is called man-made, whether or not
it is in any library. That is called **open-set** recognition: the model is allowed to meet things it has
never seen and say so.

> Analogy: a doorman who knows every resident by face does not need a list of every possible burglar. He
> stops anyone he does not recognise. A doorman with a list of known burglars waves through a burglar who
> is not on it. Level 1 is the first doorman; a natural-vs-library classifier is the second.

Here is the difference in two dimensions (the step-by-step book, Part VII, works it in full). Natural
pixels lie along the line `x + y = 1`: (0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1). The
library has one man-made spectrum, at (0.9, 0.9). A camouflage net sits at (0.4, 0.4).

```
classifier view: nearest average
  natural average      = (0.5, 0.5)
  distance to it       = sqrt(0.1^2 + 0.1^2) = sqrt(0.02) = 0.141
  distance to library  = sqrt(0.5^2 + 0.5^2) = sqrt(0.50) = 0.707      -> "natural"

level-1 view: can nature explain it?   residual = |x + y - 1|
  natural pixels       |0.1 + 0.9 - 1| = 0, and the same 0 for all five
  camouflage           |0.4 + 0.4 - 1| = 0.2                            -> "not natural"
```

The camouflage is closer to the natural *average* than the natural pixels at the ends of the line are
(they sit 0.566 from it), so no distance-to-average rule can catch it. But it is off the line nature
lives on, and only a model of nature notices.

> **As built, level 1 is not trained this way - the code wins.** The namer's level-1 head is trained
> with binary cross-entropy on planted samples: library *man-made* spectra mixed into real pixels are
> the positives, natural library spectra and plain pixels the negatives (`app/spectral_match/namer/train.py`,
> `loss`; `generator.py`, the three sample kinds). So it does see man-made examples, and it is a
> two-class classifier on the generator's world rather than a one-class model of nature. Whether it
> still flags a material it never saw is what run N002 measures: it holds the whole paint/coatings
> class out and asks how often level 1 still says man-made (pass: at least 70%). Part 27 covers it.

## Step 4 - How ACE matching works, on three bands

Level 2 needs a way to say how well a pixel matches a library spectrum. The separability check (Step 6)
uses ACE, the same score as part 15, in this form:

```
d   = pixel - ring mean              what the pixel adds to its surroundings
s_j = t_j - ring mean                what library spectrum j would add if it filled the pixel
ACE = (s_j . d)^2 / ((s_j . s_j) (d . d))      after whitening both
```

**Whitening** (part 15) divides each direction by how much it usually varies, so a change in a quiet band
counts for more than the same change in a noisy one. With independent bands it divides each band by its
standard deviation. Take part 15's grass ring, roof and soil, and band standard deviations
(0.02, 0.04, 0.01):

```
ring mean m = grass = (0.05, 0.40, 0.25)

roof:  r - m      = (0.30 - 0.05, 0.32 - 0.40, 0.35 - 0.25) = (0.25, -0.08, 0.10)
       whitened   = (0.25/0.02, -0.08/0.04, 0.10/0.01)      = (12.5, -2, 10)
soil:  s - m      = (0.20 - 0.05, 0.25 - 0.40, 0.30 - 0.25) = (0.15, -0.15, 0.05)
       whitened   = (0.15/0.02, -0.15/0.04, 0.05/0.01)      = (7.5, -3.75, 5)
```

A pixel that is 20% roof: `d = 0.2 x (r - m)`, whitened `0.2 x (12.5, -2, 10) = (2.5, -0.4, 2)`.

```
roof:  s.d = 12.5*2.5 + (-2)(-0.4) + 10*2 = 31.25 + 0.8 + 20 = 52.05
       s.s = 156.25 + 4 + 100 = 260.25
       d.d = 6.25 + 0.16 + 4  = 10.41
       ACE = 52.05^2 / (260.25 x 10.41) = 2709.2025 / 2709.2025 = 1.00

soil:  s.d = 7.5*2.5 + (-3.75)(-0.4) + 5*2 = 18.75 + 1.5 + 10 = 30.25
       s.s = 56.25 + 14.0625 + 25 = 95.3125
       ACE = 30.25^2 / (95.3125 x 10.41) = 915.0625 / 992.2031 = 0.9223
```

Roof 1.00, soil 0.92: roof wins.

**Why the fill drops out.** Now make it 50% roof: whitened `d = 0.5 x (12.5, -2, 10) = (6.25, -1, 5)`.

```
soil:  s.d = 7.5*6.25 + 3.75*1 + 5*5 = 46.875 + 3.75 + 25 = 75.625
       d.d = 39.0625 + 1 + 25 = 65.0625
       ACE = 75.625^2 / (95.3125 x 65.0625) = 5719.14 / 6201.27 = 0.9223
```

Exactly the same 0.9223. Multiplying `d` by the fill multiplies the top by `fill^2` (it is squared) and the
bottom by `fill^2` (from `d.d`); they cancel. **Fill changes the length of `d`, not its direction, and ACE
looks only at direction.** That is why one match score works for a 20% roof and a whole one.

**What does not drop out: noise.** Real pixels carry noise of a fixed size whatever the fill. Add one
standard deviation of noise to every band, whitened `(1, 1, -1)`:

```
fill 0.2:  d = (2.5, -0.4, 2) + (1, 1, -1) = (3.5, 0.6, 1)
           roof: s.d = 43.75 - 1.2 + 10 = 52.55;  d.d = 12.25 + 0.36 + 1 = 13.61
                 ACE = 52.55^2 / (260.25 x 13.61) = 2761.50 / 3542.00 = 0.780
fill 0.5:  d = (6.25, -1, 5) + (1, 1, -1) = (7.25, 0, 4)
           roof: s.d = 90.625 + 0 + 40 = 130.625;  d.d = 52.5625 + 0 + 16 = 68.5625
                 ACE = 130.625^2 / (260.25 x 68.5625) = 17062.89 / 17843.39 = 0.956
```

The same noise costs the 20% roof 0.22 of its match and the 50% roof 0.04. Small fills are hard because the
signal shrinks while the noise does not. (Checked by `research/math/spectral-matching/compute_two_level.py`.)

**Leave-one-spectrum-out.** To measure how well ACE names a family, the check plants each library
spectrum into a real pixel and matches it against the **other** 410 spectra; it sets the spectrum's own
score to -1 (`ace[e] = -1` in `separability.py`). Matching a spectrum against itself would always win and
would measure nothing. The question becomes: is the nearest *other* spectrum from the same family?

## Step 5 - A classifier, or a matching model?

Two ways to build level 2:

**A closed classifier.** A network ends in one output per class, and is trained until the right output
is largest. "Closed" means the set of classes is fixed when you train it. Add a class and the network has
no output for it: you add one and retrain.

**A matching model** (a **dual encoder**). Two networks: one turns the pixel (with its ring) into a short
vector, an **embedding**; the other turns a library spectrum into an embedding of the same length. The
match score is the cosine between the two. To name a pixel, compare its embedding with every library
spectrum's and take the best.

The user asked: "do we have to retrain if we have another library?". For the matching model, no: run the
spectrum encoder on the new spectra and they are candidates at once. A tiny case in two dimensions: the
pixel embeds to (0.8, 0.6), both of length 1.

```
painted metal (1, 0)        cos = 0.8*1 + 0.6*0     = 0.80
tarp          (0, 1)        cos = 0.8*0 + 0.6*1     = 0.60
glass (new)   (0.6, 0.8)    cos = 0.8*0.6 + 0.6*0.8 = 0.48 + 0.48 = 0.96   <- best, no retraining
```

A classifier with two outputs, one per known class, can only answer "painted metal" or "tarp". ACE itself
is a matching model with no learning at all: both "encoders" are "subtract the ring and whiten". **Level 1
does not depend on any library**, because it describes nature, not man-made materials: a new library
changes level 2's candidates and nothing else.

The trade-off is real. A matching model can only generalise to a new spectrum if the spectrum encoder has
learned to place spectra sensibly, which needs many varied training spectra; a classifier trained on a
fixed class list can be sharper on exactly those classes. The book's Part VII covers how a dual
encoder is trained (a contrastive loss); part 27 shows the one that was built, which scores whole
classes rather than single spectra.

## Step 6 - The separability check: how far physics and the library let us go

Before training anything, `research/experiments/spectral-namer/separability.py` asks how well a strong
classical matcher does. Its docstring: "No training - this bounds what the physics and the library allow
before we design classes." The recipe (the experiment's `README.md`, "Method"):

1. For each scene, resample the 411-spectrum library to the scene's bands. PRISMA: the four bench scenes.
   EnMAP and AVIRIS-NG: three implant-bench host scenes each.
2. Plant each library spectrum `t` into a real background pixel `b` (at least 8 pixels from any labelled
   object or implant) at fill `a`, with a smooth atmospheric gain error `g` (within 10%) and the scene's own
   per-band noise `n`: `x = g * (a t + (1 - a) b) + n`. Twelve background pixels per spectrum.
3. Match `d = x - g * ring mean` with ACE against every other library spectrum's direction, whitened with
   the scene's covariance (30 principal components plus a ridge).
4. The best match's family is the level-2 answer; its group (natural or man-made) is level 1.

Plantings per fill: PRISMA `411 x 12 x 4 = 19,728`; EnMAP and AVIRIS-NG `411 x 12 x 3 = 14,796` each. All
three runs finished on 2026-10-31 (`separability_prisma.json`, `separability_enmap.json`,
`separability_aviris_ng.json`).

**Level 1**, man-made called man-made / natural called natural:

| fill | PRISMA | EnMAP | AVIRIS-NG |
|---|---|---|---|
| 0.10 | 0.61 / 0.78 | 0.48 / 0.72 | 0.59 / 0.77 |
| 0.25 | 0.77 / 0.83 | 0.66 / 0.79 | 0.76 / 0.81 |
| 0.50 | 0.85 / 0.86 | 0.80 / 0.84 | 0.84 / 0.85 |
| 1.00 | 0.87 / 0.86 | 0.87 / 0.87 | 0.87 / 0.85 |

**Level 2**, share of man-made plantings named with the right family (and, on PRISMA, the share given that
level 1 was right):

| fill | PRISMA | PRISMA, given level 1 right | EnMAP | AVIRIS-NG |
|---|---|---|---|---|
| 0.10 | 0.38 | 0.63 | 0.23 | 0.36 |
| 0.25 | 0.55 | 0.72 | 0.44 | 0.53 |
| 0.50 | 0.61 | 0.72 | 0.58 | 0.59 |
| 1.00 | 0.64 | 0.73 | 0.64 | 0.62 |

Check the product rule of Step 3 on these measured counts. A class can only be right if level 1 was right,
so at PRISMA fill 0.10:

```
P(class right) = P(level 1 right) x P(class right | level 1 right)
               = 0.6096 x 0.6309 = 0.3846           (the 0.38 in the table)
```

**Per family**, PRISMA at fill 0.25 (from the JSON's confusion rows; EnMAP and AVIRIS-NG are similar):

| family | right | most often called instead |
|---|---|---|
| oil | 1537 / 1920 = 80% | vegetation, mineral |
| plastic | 1319 / 1872 = 70% | roofing, mineral |
| textile | 464 / 672 = 69% | roofing, npv |
| wood_paper | 239 / 384 = 62% | textile, npv |
| paint | 174 / 384 = 45% | plastic, roofing |
| metal | 255 / 624 = 41% | soil, mineral |
| building | 205 / 528 = 39% | pavement (82, 16%), soil |
| roofing | 398 / 1296 = 31% | npv, plastic |
| pavement | 75 / 720 = 10% | soil (111, 15%), rock (94, 13%), building (87, 12%) |
| glass | 0 / 48 | untestable: one spectrum |

**A small disagreement, and the data win.** The experiment README gives pavement as "~12%". The JSON's
pavement row says 75 of 720 right, 10.4%; the 12% is pavement called *building* (87 of 720).

Four readings:

- **Level 1 is the harder half at small fills.** At 10% fill on PRISMA, 39% of man-made plantings are
  called natural; on EnMAP, 52%. Most targets at 30 m are 10% or less. At full fill all three sensors reach
  0.87.
- **Materials with their own absorption features separate well**: oil, plastic, textile, wood and paper
  (hydrocarbon, polymer C-H and cellulose features, the README says).
- **The use-based families fail as Step 1 predicted.** Roofing (shingle, wood shake, metal, clear plastic)
  is scattered; building and pavement confuse each other and natural rock and soil. **Glass is right 0
  times**: with one spectrum, leave-one-out leaves no other glass to match.
- These are a **baseline, not a forecast**: the planted spectra are library spectra, so the domain gap
  between laboratory and satellite (part 15) is absent, and a real namer will do worse on real objects.

The README's conclusion turns this into level 2's class list: **oil / fuel, plastic, textile / fabric,
wood / paper, paint / coatings, metal, concrete / asphalt** (building and pavement merged; roofing
dissolved into its materials), glass / PV once spectra exist, and camouflage and military coatings once
field spectra arrive. "These classical numbers are the baseline the learned namer must beat, mainly at
10-25% fill." Notice that this is a list of *materials* for level 2, narrower than Step 2's list of
*things an analyst wants found*: vessels, disturbed earth and cut vegetation are combinations or contexts,
not single spectra.

**Values that will drift:** every number in this step; a new class list, library or set of host scenes
changes all of them.

---

## Common confusions

**"Two-level" - is this the two-level covariance from the book?** No. The step-by-step book also has a
"two-level covariance" (a few loud directions, many quiet ones) in its matched-filter chapter. That is a
model of noise. Here "two levels" means two questions asked in turn. Same words, unrelated ideas.

**"The families are wrong, so the matcher is broken."** The matcher (part 15) does what it says. The
families are a labelling choice made by regex for convenience; the matcher inherits it. Fixing the labels
is a curation change, not a matcher change.

**"ACE ignores fill, so fill does not matter."** ACE's *formula* ignores fill; its *accuracy* does not,
because noise stays the same size while the signal shrinks with fill. The Step 6 level-1 table falls from 0.87
to 0.61 on PRISMA (0.48 on EnMAP) for exactly that reason.

**"Level 1 is a man-made detector."** In the design, it is a nature model that calls "man-made" whatever it
cannot explain - which is what would let it flag camouflage, and also why haze or an odd natural surface it
was not trained on could fool it. As built (part 27) it is trained on planted man-made library spectra
against natural ones, so it *is* partly a man-made detector; N002's held-out class tests how open it
still is.

**"A matching model never needs retraining."** It needs none to add spectra *like* the ones its encoder
learned from. A spectrum unlike anything in training (camouflage, before any field spectra) is still a
problem; that is why level 1 exists.

**"P(class) = 0.54 means a 54% chance it is painted metal."** Only if both levels' probabilities are
calibrated (part 15's percentile calibration is one way). An uncalibrated network's 0.9 is a score, and the
product of two scores is a score.

## Check yourself

1. Using the regex list in Step 1, which family does a file named `brick_paver_gds...` get, and which does
   `brick_wall_...` get? Why is that a problem from orbit?
2. A container 12 m by 2.4 m. What fill does it have in a 30 m pixel and in a 5 m pixel? Show the steps.
3. Level 1 says `P(man-made) = 0.7`; level 2 says bare metal 0.5, painted metal 0.3, tarp 0.2. Compute every
   class's probability and `P(natural)`, and check they add to 1.
4. In Step 4's example, make the pixel 100% roof with no noise. Compute the whitened `d` and ACE against
   soil. Then explain in one sentence why the answer could have been written down without arithmetic.
5. From the Step 6 table at fill 0.25, compute `P(level 1 right) x P(class right | level 1 right)` and
   compare it with the table.
6. Why can the separability check never get glass right, whatever the fill?

<details>
<summary>Answers</summary>

1. `brick_paver`: the roofing rule fails, the pavement rule `brick.*pav` matches -> **pavement**.
   `brick_wall`: pavement fails (no "pav" after "brick"), the building rule `brick` matches -> **building**.
   The same fired clay lands in two families depending on a word in its file name; from orbit a brick
   path and a brick wall top look alike.
2. Area `12 x 2.4 = 28.8 m^2`. 30 m pixel: `28.8 / 900 = 0.032`, about 3%. 5 m pixel: `28.8 / 25 = 1.152`
   - more than 1, so the container fills a whole pixel and spills into the next ones.
3. Bare metal `0.7 x 0.5 = 0.35`; painted metal `0.7 x 0.3 = 0.21`; tarp `0.7 x 0.2 = 0.14`; natural
   `1 - 0.7 = 0.30`. Sum `0.35 + 0.21 + 0.14 + 0.30 = 1.00`.
4. `d = 1.0 x (12.5, -2, 10) = (12.5, -2, 10)`. Soil: `s.d = 93.75 + 7.5 + 50 = 151.25`; `s.s = 95.3125`;
   `d.d = 260.25`; `ACE = 151.25^2 / (95.3125 x 260.25) = 22876.56 / 24804.58 = 0.9223`. Without
   arithmetic: fill only scales `d`, and ACE does not change when `d` is scaled.
5. `0.7724 x 0.7151 = 0.5523`, the table's 0.55 (level 2 right). It must match, because a class can only be
   right when level 1 was right.
6. The library holds one glass spectrum. Leave-one-spectrum-out removes it from the candidates, so no glass
   remains to be matched; the best a glass planting can get is a wrong family.

</details>

---

Next: part 27, the namer as it was built on 2026-10-31 - the dual encoder on V2's stem, the planted-sample
generator, the class-level loss, and the test it must pass.
