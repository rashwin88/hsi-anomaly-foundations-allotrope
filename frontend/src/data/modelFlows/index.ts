// Registry of foundation-model flow charts. Looked up by architecture
// slug so ModelDetailPage can fetch the manifest from /api/models/:arch
// and pair it with the matching flow definition without a giant switch.

import { hyperspectralSegformerMaeFlow } from "./hyperspectral_segformer_mae";
import { normalizedMaskedAutoencoderFlow } from "./normalized_masked_autoencoder";
import { segformerMaeFlow } from "./segformer_mae";
import { spatialAutoencoderFlow } from "./spatial_autoencoder";
import { spatialMaskedAutoencoderFlow } from "./spatial_masked_autoencoder";
import { spatialMaskedAutoencoderL1Flow } from "./spatial_masked_autoencoder_l1";
import { spatialMaskedAutoencoderL1UnnormalizedFlow } from "./spatial_masked_autoencoder_l1_unnormalized";
import type { ModelFlow } from "./types";

export const MODEL_FLOWS: Record<string, ModelFlow> = {
  spatial_autoencoder: spatialAutoencoderFlow,
  spatial_masked_autoencoder: spatialMaskedAutoencoderFlow,
  spatial_masked_autoencoder_l1: spatialMaskedAutoencoderL1Flow,
  spatial_masked_autoencoder_l1_unnormalized:
    spatialMaskedAutoencoderL1UnnormalizedFlow,
  normalized_masked_autoencoder: normalizedMaskedAutoencoderFlow,
  segformer_mae: segformerMaeFlow,
  hyperspectral_segformer_mae: hyperspectralSegformerMaeFlow,
};

export type { ModelFlow, ModelFlowEdge, ModelFlowNode, StageKind } from "./types";
