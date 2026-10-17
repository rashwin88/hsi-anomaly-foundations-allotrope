# Public HAD benchmarks: HAD100 and ABU, acquired and vetted

**Purpose:** get two public hyperspectral anomaly detection (HAD) benchmarks onto the dev laptop so the
Indradhanu models can be evaluated on them, and record what a loader must do. Our models read any band list by
wavelength and expect reflectance (0-1; shards store int16 x 0.0001). So for every scene we need four things:
the cube, the band-centre wavelengths, a GT mask, and the value scale. Because HAD100 is cut from the public
AVIRIS-NG archive, we also checked it for overlap with the 79 AVIRIS-NG lines we trained on.

No product code was changed. The data and manifests live outside the repo:

| path | contents |
|---|---|
| `C:/es/benchmarks/had100/` | `HAD100.zip` (3,754,846,290 B) + extracted `HAD100/` (4.47 GB), `manifest.json` |
| `C:/es/benchmarks/abu/` | 13 `abu-*.mat` (40.6 MB total), `manifest.json`, `band_alignment_estimate.json` |
| `C:/es/benchmarks/_scripts/` | `safe_unzip.py`, `had100_manifest.py`, `abu_manifest.py`, `abu_band_align.py`, `cmr_footprints.py` (+ `cmr_footprints.json`, `abu_cdx.txt`, `our_manifest.json`) |

All reads used numpy, scipy.io.loadmat or pyproj, run as `C:/av/Scripts/python.exe -I`. Nothing shipped in
the downloads was executed. HAD100's `main.py` was read as text, and its `crop_dict` and `band_select` were
transcribed into `had100_manifest.py`.

## 1. Sources (retrieved 2026-10-16)

| dataset | official source | what was downloaded | integrity | licence |
|---|---|---|---|---|
| HAD100 | project page https://zhaoxuli123.github.io/HAD100/, linked from the code repo https://github.com/ZhaoxuLi123/AETNet | Google Drive file `10Z6f-8ELatxdAi31IDg2Mgbl92_4EnCJ` = `HAD100.zip`, sha256 `6f91035543b7bc7806ebc555cd5411d320f2810f0af8dfcf20fe4f331227c19f` (no version tag; the page gives no release date) | Test tiles + GT are byte-identical to the copy used in `dual-window-mahalanobis/` (`C:/Users/FITT/Documents/public_had_datasets/bench/HAD100`) | Data: none stated. AETNet code: MIT. Cite Li et al., IEEE TGRS 61, 2023, doi 10.1109/TGRS.2023.3258067 (arXiv 2303.18001) |
| ABU | Xudong Kang's page https://xudongkang.weebly.com/data-sets.html. **Returns 404 as of 2026-10-16**; it was live on 2026-09-17 | The 13 official upload URLs (`.../uploads/1/6/4/6/16465750/abu-<scene>.mat`), fetched as raw Wayback captures from 2026-09-17 (`web.archive.org/web/<ts>id_/<url>`) | The SHA-1 of every file equals its Wayback CDX digest. That digest is unchanged in every capture from 2018-11 to 2026-09, and all 13 sha256 match the earlier local copy in `public_had_datasets/bench` | None stated on the page. Cite Kang et al., "Hyperspectral anomaly detection with attribute and edge-preserving filters", IEEE TGRS 55(10):5600-5611, 2017 |

Other mirrors. HAD100: Baidu Disk https://pan.baidu.com/s/1S5p442t_jL_mPoL0m-useA (code 1234). ABU: IEEE
DataPort https://ieee-dataport.org/documents/hyperspectral-dataset-1 (doi 10.21227/98g3-b397, uploaded
2026-03 by a third party, Mingxing Wang). DataPort requires a logged-in subscription, so it was **not used**,
and nothing is lost by skipping it. HAD100's baselines and evaluation code are a separate download (Drive
`1d4Lr1jqD7hzhTpIWIU-49smFa14RREEe`), not fetched.

## 2. HAD100

**Contents of the zip:** `data/aviris_ng_target` (94 ENVI tiles), `gt/aviris_ng_gt` (94 `.mat`),
`data/aviris_ng_normal` (260 anomaly-free AVIRIS-NG tiles, all 81x81), `data/aviris_normal` (262
anomaly-free AVIRIS-Classic tiles, 224 bands, mostly 81x81), `main.py`. The zip holds the raw tiles, not the
"100 scenes". `main.py` builds the 100 test scenes by cropping 64x64 from the 94 tiles at hand-picked offsets.
Six tiles each give two crops, so 94 tiles become 100 scenes.

| item | value |
|---|---|
| sensor / level | AVIRIS-NG **L1B at-sensor radiance**, header says `uW nm-1 cm-2 sr-1`. float32, BIL, little-endian |
| raw test tiles | 94: 76 at 64x64, 11 at 100x100, 2 at 70x70, one each at 75, 80, 86x81, 101, **120x120**. All 425 bands. The papers' "64 to 101" omits the 120x120 tile (`ang20170908t225309_34`) |
| protocol test scenes | 100 crops of 64x64. 3,567 anomaly pixels, 460 objects (8-connected). Per scene: median 25.5 px (0.6%), min 6, max 251. No scene is empty |
| raw-tile GT | `.mat` key `gt`, uint8 {0,1}, same HxW as the tile. 3,568 px, 466 objects (8-conn), 471 (4-conn) |
| values | radiance, median per tile ~1.0, p99 ~7.8, max up to 78.7. Every tile has a few small negatives (min about -0.19). `data ignore value = -9999` is declared, but no pixel has it |
| wavelengths | in every `.hdr`: 425 centres, 376.7-2500.4 nm, about 5.01 nm spacing. **They differ slightly per flight line** (three calibrations: 376.86, 376.72 and 377.07 nm start), so read them per tile, not once. No `fwhm` and no `bbl` in the AVIRIS-NG headers |
| geo | full `map info` per tile (UTM, 1.9-8.4 m GSD, rotated). `x start` / `y start` give the tile's pixel offset in its source line |
| HAD100 "good bands" (`main.py`) | 0-based `np.r_[15:109, 118:145, 158:187, 227:274, 328:407]` = 276 bands: 452.0-917.8, 967.9-1098.1, 1168.2-1308.5, 1513.8-1744.2, 2019.7-2410.4 nm |

**HAD100's own train/test flight lines.** All 10 AVIRIS-NG lines supply both test tiles and anomaly-free
training tiles. HAD100's train and test sets therefore share flight lines, separated only by position within
each line.

| flight line | test tiles | normal tiles | where (tile centre) | GSD |
|---|---|---|---|---|
| ang20170821t183707 | 38 | 90 | Alberta, CA (51.44 N, 114.50 W) | 2.3 m |
| ang20170821t195229 | 8 | 20 | Alberta, CA (51.52 N, 113.10 W) | 2.5 m |
| ang20170825t173426 | 5 | 13 | Alberta, CA (52.04 N, 114.88 W) | 3.0 m |
| ang20170908t225309 | 10 | 30 | California (35.05 N, 118.91 W) | 2.0 m |
| ang20171012t194435 | 12 | 22 | Arizona/California border (34.93 N, 114.60 W) | 2.0 m |
| **ang20180227t082814** | **4** | 14 | **India, Andhra Pradesh, near Kakinada (17.07 N, 82.28 E)** | 4.0 m |
| ang20191004t185054 | 6 | 21 | Texas/Louisiana coast (30.03 N, 94.13 W) | 8.4 m |
| ang20191004t215336 | 3 | 10 | Texas coast (29.43 N, 95.02 W) | 8.4 m |
| ang20191027t204454 | 1 | 9 | Ohio (38.86 N, 84.23 W) | 5.1 m |
| ang20210614t141018 | 7 | 31 | Switzerland (47.47 N, 7.92 E) | 1.9 m |

The AVIRIS-Classic training tiles come from 14 lines (`f080709t01p00r15` ... `f180601t01p00r09`, 2008-2018,
US only, 13-17 m). They are int16 scaled radiance with 224 bands, and their wavelengths and fwhm are in the
headers. Full list in the manifest.

## 3. Leakage check against our 79 AVIRIS-NG training lines

Method: the line ids came from `patches/aviris_ng/native-v1/manifest.json` (73 train, 6 test). Footprints
came from NASA CMR (collection `AVIRIS-NG_L1B_radiance_2095`, public granule polygons), and HAD100 tile
footprints from each tile's own `map info`. Distances were measured in UTM 44N with shapely.

- **Exact line overlap: none.** None of HAD100's 10 AVIRIS-NG lines is among our 79.
- **Same campaign, same day, adjacent swath: yes, one line.** `ang20180227t082814` was flown on 2018-02-27,
  the same day as three of our **train** lines: `ang20180227t043325`, `t072844` and `t084640`. `t084640`
  was flown 18 minutes later on the parallel track immediately east. The CMR outlines are about 630 m apart
  and do not overlap. HAD100 tiles fall inside `t082814`'s own outline, as a sanity check.
- **Distance from each HAD100 tile to our nearest training footprint (`t084640`):** test tiles `_15` 1,284 m,
  `_16` 1,865 m, `_17` 1,020 m, `_18` 1,185 m. HAD100's normal tiles from this line are 1,087-3,314 m away.
  `t072844` is about 8.4-10.4 km west, and the others are 12-175 km away.
- Those 4 tiles become **6 of the 100 protocol test scenes** (`ang20180227t082814_15_1`, `_15_2`, `_16`,
  `_17_1`, `_17_2`, `_18`).

**Verdict: no pixel-level leakage (high confidence).** The line ids are exact. The footprint gap rests on CMR's
4-corner outlines, which are approximate, but a 1 km margin is several times any plausible outline error. There
is **near-domain proximity for 6/100 scenes (certain)**: same campaign, same day, same delta landscape, 1-2 km
from a training swath. The other 94 scenes come from North America and Switzerland in 2017-2021, which our
India-only training never saw. **Report HAD100 mAUC both with and without the 6 Kakinada scenes.** If we ever
train on HAD100's normal tiles, its train/test line sharing (above) is a separate leak to avoid.

## 4. ABU

| scene | sensor, place, GSD (Kang 2017, via secondary sources) | H x W x B | dtype | p1 / p50 / p99 / max | anomaly px | objects (8-conn) |
|---|---|---|---|---|---|---|
| airport-1 | AVIRIS, Los Angeles, 7.1 m | 100x100x205 | uint16 | 20 / 440 / 2713 / 6604 | 144 | 13 |
| airport-2 | AVIRIS, Los Angeles, 7.1 m | 100x100x205 | uint16 | 21 / 432 / 2736 / 6644 | 87 | 2 |
| airport-3 | AVIRIS, Los Angeles, 7.1 m | 100x100x205 | int16 | 13 / 315 / 2136 / 5651 | 170 | 16 |
| airport-4 | AVIRIS, Gulfport, 3.4 m | 100x100x191 | uint16 | 17 / 777 / 2987 / 5061 | 60 | 3 |
| beach-1 | AVIRIS, Cat Island, 17.2 m | 150x150x188 | int16 | 0 / 28 / 1858 / 6593 | 19 | 1 |
| beach-2 | AVIRIS, San Diego, 7.5 m | 100x100x193 | int16 | 0 / 16 / 1370 / 4100 | 202 | 58 |
| beach-3 | AVIRIS, Bay Champagne, 4.4 m | 100x100x188 | int16 | 10 / 321 / 2681 / 5404 | 11 | 1 |
| beach-4 | **ROSIS-03**, Pavia, 1.3 m | 150x150x102 | float64 | 0 / 0.051 / 0.552 / **1.0** | 68 | 7 |
| urban-1 | AVIRIS, Texas Coast, 17.2 m | 100x100x204 | int16 | -2 / 513 / 2950 / 6534 | 67 | 9 |
| urban-2 | AVIRIS, Texas Coast, 17.2 m | 100x100x207 | int16 | -3 / 433 / 3079 / 14060 | 155 | 20 |
| urban-3 | AVIRIS, Gainesville, 3.5 m | 100x100x191 | uint16 | 9 / 525 / 3193 / 7208 | 52 | 11 |
| urban-4 | AVIRIS, Los Angeles, 7.1 m | 100x100x205 | int16 | 14 / 389 / 2699 / 19492 | 272 | 25 |
| urban-5 | AVIRIS, Los Angeles, 7.1 m | 100x100x205 | int16 | 17 / 380 / 3006 / 7120 | 232 | 30 |

Every file has two variables: `data` (H, W, B) and `map` (H, W) uint8 {0,1}. They are MATLAB 5.0 files, so
scipy reads them and h5py is not needed. There are no all-zero spectra, and the minimum is -50 (small
negatives). Sizes and band counts match the brief. Two published place names conflict: one paper puts all
four airports in Gulfport, but the table used above has airports I-III in Los Angeles. The ABU page itself
gives no metadata.

**Scale: AVIRIS scenes look like scaled at-sensor radiance, not reflectance (medium-high confidence).** The
median spectra peak in the blue (about 1,800 counts), fall to about 20-100 near 2.4 um and have deep
1.4 / 1.9 um water dips. That shape and those magnitudes match HAD100's AVIRIS-Classic orthocorrected int16
radiance tiles (for example, 2009 tiles have a median of 1,880 at 414 nm and 22 at 2,416 nm). Reflectance x
10,000 of airports and cities could not be about 1% of its blue value at 2.4 um. The AVIRIS gain factors
(which differ by spectrometer and by era) are not given, so absolute radiance is unknown. **beach-4 is
min-max normalised to [0, 1] over the whole cube**, which is not physical.

**Wavelengths: none ship with ABU, and the band-removal lists are unpublished.** We tried to estimate them.
`abu_band_align.py` aligns each scene's median log spectrum to the 224-band AVIRIS-Classic reference
(HAD100 tiles, wavelengths from their headers) with a monotone DP that allows a few contiguous gaps. Result
(`abu/band_alignment_estimate.json`, removed 1-based AVIRIS band blocks):

- 205-band scenes (airport-1/2/3, urban-4/5): {43}, 108-112, 155-167. This is consistent across all five, but
  a lone removed band 43 is implausible.
- 191 (airport-4, urban-3): 108-117, 152/153-170, 221-224. 204 (urban-1): 108-118, 152-154, 165-170.
  207 (urban-2): 107-111, 118-122, 164-170.
- 188 and 193 (beach-1/2/3): ragged and different per scene, with RMS misfit 0.9-1.3 log units.

The removed blocks land on the expected water bands (about 1,350-1,450 and 1,800-1,950 nm) plus the SWIR
tail. However, the fits are loose (0.4-1.3 log RMS), and different AVIRIS years have slightly different
centres. **Treat these as low-confidence estimates, not wavelengths to evaluate with.** beach-4 (ROSIS-03, 102
bands, nominally 430-860 nm at about 4 nm) is visible/NIR only, and its dropped-band list is also unpublished.

## 5. How papers preprocess these sets (not comparable across rows)

| protocol | test set | bands | metric | used by |
|---|---|---|---|---|
| HAD100 official | 100 crops of 64x64 (`main.py` crop_dict) | 276 "good" bands, then the **first 50 / 100 / 200** of those. First 50 = 452-697 nm (visible only); first 100 = 452-918 + 968-993 nm; first 200 ends at 2,030 nm | mAUC = mean per-scene ROC-AUC | Li et al. 2023 (AETNet), our `dual-window-mahalanobis/step6` |
| "94 tiles, first 50 bands" | the 94 raw tiles, uncropped | first 50 of the **raw 425** = 377-622 nm | per-scene AUC, averaged | Ma et al. 2024 STAD (arXiv 2401.01093) |
| 94 tiles, full spectrum | 94 raw tiles | all, mapped to our grid | mean AUC | our `dual-window-mahalanobis/step3` |
| ABU | 13 scenes as shipped | as shipped (each scene its own band set) | per-scene AUC; ROC and box plots | essentially every HAD paper; AETNet's `detect_any_HSI.py` |

"First 50 bands" means two different things, depending on whether the paper takes them from the 276 good
bands or the raw 425. Both versions are visible-only, so our SWIR-aware models are handicapped against any
row that uses them. Quote the full-spectrum numbers, say which protocol, and do not mix them in one table.

## 6. What a loader must do

**HAD100**
1. Read the ENVI `.hdr` and `.dat` directly (`data type = 4` float32, `interleave = bil`, `byte order = 0`) and
   reshape (lines, bands, samples) to (H, W, B). Wavelengths come from that tile's own `.hdr`.
2. Units are radiance in uW nm-1 cm-2 sr-1, not reflectance. Choose one of three routes:
   (a) an empirical per-band affine map onto our training statistics, as in `dual-window-mahalanobis/step3`.
   This is crude and does not remove the atmosphere.
   (b) top-of-atmosphere reflectance = pi L / (E0 cos SZA). This needs the solar zenith per line, which is in
   the AVIRIS-NG `obs` files, not the tiles.
   (c) **preferred:** fetch the AVIRIS-NG L2 reflectance for the 10 lines from ORNL DAAC
   (`AVIRIS-NG_L2_Reflectance_2110`) and cut the same windows using each tile's `x start` / `y start`. This
   needs an Earthdata login (user action), and we would need to confirm that the L2 grid matches the L1
   ortho grid.
3. Band selection: drop the water-vapour bands (use `main.py`'s 276 good bands, or our own bad-band list
   against the wavelengths), then resample onto our model's band list by wavelength.
4. Small negatives: clip to 0 or leave as they are. There is no nodata in practice, but treat -9999 as
   nodata anyway.
5. GT: `loadmat(gt/aviris_ng_gt/<tile>.mat)["gt"] > 0`. For the official protocol, apply the 100 crops
   listed in `had100/manifest.json` -> `protocol_test_scenes` (row, col, 64x64).
6. Tag the 6 Kakinada scenes (`ang20180227t082814_*`) so results can be reported without them.

**ABU**
1. `loadmat(f)["data"]` (H, W, B) and `loadmat(f)["map"] > 0`. Cast to float, because dtype differs across
   files (uint16 / int16 / float64).
2. There are no wavelengths. A wavelength-driven model cannot run on ABU correctly without guessing them.
   Either skip ABU for Indradhanu (score only classical detectors, as before), or run it on the estimated
   band lists and **label the result as such**.
3. AVIRIS scenes are scaled radiance with an unknown per-scene gain, so reflectance is unavailable. beach-4
   is min-max normalised ROSIS covering 430-860 nm only. Exclude it from Indradhanu runs.

## 7. Open questions

1. Should we pull AVIRIS-NG L2 reflectance for HAD100's 10 lines from ORNL DAAC (needs an Earthdata login,
   which is the same blocker as the ACOLITE ancillary work)? It is the only way to evaluate HAD100 in the
   units our model was trained on.
2. ABU wavelengths. Kang's group may publish band lists in the AED paper's supplement. Alternatively, the
   source AVIRIS flight lines could be identified (Los Angeles 2011, Gulfport 2010, ...) and their
   radiance matched band by band. Until one of these is done, any Indradhanu number on ABU rests on a guess.
3. HAD100 has no version or DOI. The zip's sha256 above is the only pin; if the Drive file changes, re-vet it.
4. ABU's official page is now 404. Our copy is from the Wayback captures (identical since 2018). Cite the
   paper, and note the archive URL as the retrieval route.
