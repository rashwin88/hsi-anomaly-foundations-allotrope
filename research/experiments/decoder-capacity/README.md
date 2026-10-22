# Decoder capacity - is FT04's decoder too unexpressive?

**Status:** diagnosis done 2026-10-21 on the laptop (no training); follow-up FT09 (a richer decoder) in
research/experiments/indradhanu-finetune/FT09-decoder/. Prompted by the user: "Maybe a bigger model is the answer",
"Or maybe more layers of decoding", "I feel the decoder maybe too un-expressive given the sophistication of the encoder."

## Where FT04's parameters are
encoder 3.33 M, decoder 2.16 M, segmentation head 0.37 M, spectral_in 14.7 k, spectral_out 4.3 k. The decoder
(SegFormer all-MLP) works entirely at 1/4 resolution: per-stage 1x1 projections to 256, bilinear up to stage 1, a 1x1
fuse, one 3x3 refine, then one 3x3 conv to D x 16 channels and PixelShuffle(4) - each 4 x 4 pixel block is emitted at
once from one 256-wide vector. The spectral output is exactly linear: band n = U(lambda_n) . z + c(lambda_n), D = 32.

## 1. The spectral floor ([spectral_floor.py](spectral_floor.py))
Share of FT04's residual energy outside the 32-curve span its output layer can produce (that part no upstream change
can remove), PRISMA bench, normalised units:

| scene | background: actual / floor mean abs | floor share | targets: actual / floor | floor share |
|---|---|---|---|---|
| set1 | 0.048 / 0.013 | 0.07 | 0.189 / 0.022 | 0.02 |
| set2 | 0.122 / 0.017 | 0.02 | 0.204 / 0.025 | 0.02 |
| set3 | 0.121 / 0.033 | 0.09 | 0.149 / 0.043 | 0.10 |
| set4 | 0.115 / 0.015 | 0.03 | 0.177 / 0.025 | 0.03 |

**The spectral output is not the bottleneck** (2-10% of the error). Over 90% is in which code z the decoder predicts.

## 2. Visible vs hidden pixels ([visible_vs_hidden.py](visible_vs_hidden.py))
Mean abs error per band, 24 random 128 px tiles per scene (>= 95% valid), FT04:

| scene | background visible (no mask) | background hidden (product two-pass) | targets visible | targets hidden |
|---|---|---|---|---|
| set1 | 0.056 | 0.057 | 0.194 | 0.218 |
| set2 | 0.110 | 0.137 | 0.161 | 0.220 |
| set3 | 0.159 | 0.129 | 0.176 | 0.179 |
| set4 | 0.135 | 0.139 | 0.169 | 0.191 |

**FT04 cannot reproduce pixels it can see much better than ones it must predict.** A model limited by
unpredictability would autoencode visible pixels almost exactly; this one is limited by what its architecture can
render - most likely the 4 x 4-token bottleneck (512 numbers per token into a 32-wide stage-1 vector) with no
full-resolution path back in the decoder. The background error RM fights is largely detail FT04 cannot render; targets
stay 1.2-3.8x above it.

## Consequence
A decoder that renders sub-token detail - progressive upsampling plus a full-resolution skip from the input, with
hidden pixels zeroed in the skip so nothing leaks - should lower background error while unpredictable targets stay
high. FT09 tests it on FT04's encoder (no re-pretraining).
