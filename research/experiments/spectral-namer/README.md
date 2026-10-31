# Spectral namer - separability first (natural vs man-made, then the class)

**Status:** separability check done 2026-10-31 02:45 UTC on the laptop (no training). Next: design doc for the
two-level matching model, then training on H100 block 2 (`cr-0dccba00b34dc29e1`, 2026-10-31 11:30 -> 11-02 11:30 UTC).

## Design so far (user, 2026-10-31)
- **Two levels:** (1) natural or man-made? (2) if man-made, which material class? Outputs: natural / man-made, class
  unknown / man-made + class; P(class) = P(man-made) x P(class | man-made).
- Level 1 is to be trained as "not explained by nature" so it can flag materials no library holds (camouflage,
  military coatings).
- Level 2 as a **matching** model (pixel+ring encoder vs spectrum encoder) so a new library needs no retraining.
- Defence-oriented class list; at 30 m only large objects are nameable (a vehicle is ~1% of a pixel).

## Method ([separability.py](separability.py))
The 411-spectrum family library resampled to each scene's bands; each spectrum t mixed into real background pixels:
x = g (a t + (1 - a) b) + n, r = g x ring mean (ring 2-7 px), g a smooth gain curve within +-10%, n the scene's
shift-difference noise. Classified by **ACE** on d = x - r against every other spectrum's s_j = t_j - r, whitened with
the scene covariance (top-30 PCA + ridge); **leave-one-spectrum-out**. 12 backgrounds per spectrum; PRISMA: the 4 bench
scenes; EnMAP / AVIRIS-NG: 3 implant-bench hosts each, background pixels >= 8 px from implants.
Outputs: [separability_prisma.json](separability_prisma.json), [_enmap](separability_enmap.json),
[_aviris_ng](separability_aviris_ng.json), [separability_run.log](separability_run.log).

## Results
**Level 1** (man-made called man-made / natural called natural):

| fill | PRISMA | EnMAP | AVIRIS-NG |
|---|---|---|---|
| 0.10 | 0.61 / 0.78 | 0.48 / 0.72 | 0.59 / 0.77 |
| 0.25 | 0.77 / 0.83 | 0.66 / 0.79 | 0.76 / 0.81 |
| 0.50 | 0.85 / 0.86 | 0.80 / 0.84 | 0.84 / 0.85 |
| 1.00 | 0.87 / 0.86 | 0.87 / 0.87 | 0.87 / 0.85 |

**Level 2**, all man-made: exact family 0.38 / 0.55 / 0.61 / 0.64 at fill 0.1 / 0.25 / 0.5 / 1.0 on PRISMA (0.63 /
0.72 / 0.72 / 0.73 given level 1 right); EnMAP 0.23 / 0.44 / 0.58 / 0.64; AVIRIS-NG 0.36 / 0.53 / 0.59 / 0.62.

Per family at fill 0.25, PRISMA (EnMAP / AVIRIS-NG similar): oil 80%, plastic 70%, textile 69%, wood/paper 62%,
paint 45%, metal 41%, building 39% (16% -> pavement), roofing 31% (scattered: npv, plastic), pavement 10.4% (75 of 720; -> soil
15%, rock 13%, building 12%), glass untestable (1 spectrum: held out, nothing of its family remains).

## Conclusions for the class list
1. oil, plastic, textile, wood/paper separate well (hydrocarbon, polymer C-H, cellulose features).
2. "roofing" is a use, not a material (asphalt shingle, wood shake, metal, clear plastic) - dissolve it into materials.
3. building and pavement confuse each other and natural rock / soil - merge as **concrete/asphalt**; its difficulty
   is level 1.
4. metal (flat, bright - confused with soil) and paint (pigment over substrate - confused with plastic) are mid-range.
5. glass/PV needs spectra before it can be tested.
**Proposed level-2 classes:** oil/fuel, plastic, textile/fabric, wood/paper, paint/coatings, metal, concrete/asphalt,
(glass/PV when spectra exist); camouflage and military coatings when field spectra arrive. These classical numbers are
the baseline the learned namer must beat, mainly at 10-25% fill.
