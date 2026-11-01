# Part 34 - The verdict on HyperSL, and what would have to change

> **The one thing this part teaches:** HyperSL is a sound model of what spectra look like inside one scene, and on every
> one of this product's jobs a design choice that serves that purpose (per-scene scaling, band masking within a pixel, a
> tokenizer over band index, smoothing heads, cloud-free pretraining) works against the job; making it fit would mean
> retraining it with our choices, which is what Indradhanu already is.

## Where we are

Parts 29-33 measured HyperSL job by job. This part puts them side by side. It adds no new measurement; every number has its
source in the part that measured it. The book version is `research/textbook/chapters/49-hslverdict.tex`.

> **Analogy.** A superb road bicycle offered to a team that delivers parcels across a ploughed field. Thin tyres, no rack,
> a racing position: each a deliberate choice for a road, each a problem in a field. Fit fat tyres, a rack and a new frame
> and you have built a different bicycle - and the team already owns one.

## Step 1 - the decision table

| requirement | HyperSL | this product | part |
|---|---|---|---|
| same spectrum, same input in every scene | no: per-band min-max over the scene | `WavelengthStats`, one fixed curve | 30 |
| a library spectrum has an input | none; the scene frame clipped 41% of ECOSTRESS | the same curve, the same stem | 30 |
| anomalies matched to man-made entries | below SAM on 3 of 4 scenes; one leaf took 59.6% of set2 | Simplex ACE (adopted), the namer (training) | 30 |
| band lists with removed windows | convolution over index; a 5-band window spans 217 nm | 40 nm wavelength bins | 30 |
| a pixel must not help rebuild itself | rebuilds a pixel's bands from its own bands | hides whole pixels | 31 |
| detection, 909-object bench | own residual 130 objects, pAUC 0.018; with our RM 350, 0.083 (< PCA-RX 0.128, no model) | FTH3 ep 5: 0.189, 449 objects | 31 |
| sub-pixel targets kept | 0.25 m target: object AUC 0.68 | ACE 0.97 | 32 |
| blind unmixing, exact truth | worse than VCA + FCLS on every set | VCA + FCLS | 32 |
| thin cloud found | 28.9% of EnMAP's cirrus | 67.9% from raw bands; Indradhanu's head | 33 |

**Where HyperSL did better** - a fair verdict lists these too:

- library grouping, 5-neighbour same-class share 0.965 vs 0.949 (part 30);
- thick-cloud recall 87.1% vs 80.4%, and fewer high-plateau false calls 4.34% vs 6.62% (part 33);
- as a false-alarm filter with a hindsight cut, 40% vs 23% false alarms removed (part 31);
- agreement with FCLS references on Samson, Jasper, Urban (part 32);
- its paper's clustering and fine-tuning claims reproduce (part 29).

None is one of our jobs as we do them, and the filter result did not survive an honest threshold.

## Step 2 - what it costs

| cost | record |
|---|---|
| compute | 1,773 s of laptop GPU for the 4 bench scenes (438 us per pixel, 4 draws); Indradhanu Heavy 454 s per scene on 16 CPU threads; unmixing 5-13 min per scene on a 96 GB GPU |
| licence | no LICENSE on GitHub; MIT on Hugging Face - unresolved for a product |
| data | the ~265 GB corpus is on Baidu Netdisk only: no audit, no retrain with our scaling |
| paper vs code | base 100 vs 10,000; "layer-by-layer normalization" = per-scene min-max; 300 vs 200 million spectra; a 439-band set not in `train.py`; "endmember-free" code that starts from VCA; downstream scripts that never load the checkpoint; a last encoder layer with no skip connection |

## Step 3 - what would have to change

1. **Scaling:** train on reflectance with a fixed per-wavelength normalisation.
2. **Tokenizer:** read wavelengths, not band positions.
3. **Masking and context:** hide whole pixels and give the model neighbours.
4. **Data:** keep clouds in pretraining, and label them.
5. **Heads:** judge per-pixel heads at fills of 0.1-0.25; do not smooth abundances.
6. **Access:** publish the corpus and a licence.

Items 1-4 describe a wavelength-aware spatial masked autoencoder pretrained on reflectance with fixed statistics -
Indradhanu on its SpectralEarth stem (Indradhanu workings 31-42). **The changes HyperSL would need are the design we already
chose, for the same reasons.**

What we kept: wavelength as a coordinate (shared idea), and a discipline for outside models - hash the checkpoint, read the
code before the paper, reproduce a published number first, keep the outputs. The one claim in this series that cannot be
re-checked (the 28% of part 30) is the one whose output was not kept.

## Common confusions

**"HyperSL is a bad model."** A good model for one-scene land-cover description, measured on jobs it was not built for.

**"Fine-tuning would fix it."** Fine-tuning keeps the scaling, the band masking and the tokenizer.

**"The filter result means it helps detection."** Only with a cut chosen on the test labels.

**"MIT on Hugging Face settles the licence."** The code repository has none; settle it with the authors.

## Check yourself

1. For each job (matching, detection, unmixing, cloud), name HyperSL's design choice most responsible, and the part.
2. Which of the six changes could be made without retraining?
3. How many more objects does FTH3 ep 5 catch in the top 1% than HyperSL RM, both cloud-excluded? As a share of 909? Which
   bootstrap row agrees?
4. Name the five measures where HyperSL was ahead and why each does not change the decision.

<details>
<summary>Answers</summary>

1. Matching: per-scene min-max (30). Detection: band masking within a pixel (31). Unmixing: 3x3 head + TV (32). Cloud:
   cloud-free pretraining, compounded by min-max (33).
2. Only 6, and that needs the authors. 1-5 change what the weights were trained to read; part 29 measured how far a changed
   input moves the loss (micrometres: 0.0024 -> 0.173).
3. `449 - 369 = 80`; `80 / 909 = 8.8%`; "HyperSL RM, cloud excl. - FTH3": -8.8 [-11.7, -6.2] pts.
4. Library grouping is not naming anomalies; thick cloud is the easy class while cirrus fell to 28.9%; the plateau gain is
   small next to 39 points of cirrus; the filter's honest cut lost up to 58% of objects; FCLS references are themselves
   unmixings, and on exact truth it lost every set.

</details>

---

This is the last part of the course. The authoritative records are the experiment READMEs it cites and `docs/lld/`.
