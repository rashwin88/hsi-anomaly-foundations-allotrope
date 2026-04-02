"""
Mix-FFN -- position-aware feed-forward network for SegFormer.

In a standard transformer, the FFN is two linear layers with an activation:
    Linear(C, C*E) -> GELU -> Linear(C*E, C)

This treats each token independently -- no spatial awareness. The model must
rely entirely on attention (and positional encodings) to understand spatial
relationships.

SegFormer's Mix-FFN inserts a depthwise convolution between the two linear
layers. The depthwise conv operates on the 2D token grid with a 3x3 kernel,
so each token's representation gets mixed with its spatial neighbors. This
gives the model local spatial awareness WITHOUT explicit positional encodings.

    Linear(C, C*E) -> reshape to 2D -> DWConv3x3 -> reshape to seq -> GELU -> Linear(C*E, C)

"Depthwise" means groups=channels: each of the C*E channels gets its own
independent 3x3 spatial filter. No cross-channel mixing happens in the conv
(the surrounding Linear layers handle cross-channel interaction). This keeps
the parameter count low: C*E * 3 * 3 parameters instead of C*E * C*E * 3 * 3.

The expansion ratio E (default 4) gives the network more capacity in the
intermediate representation -- a wider hidden layer can represent more complex
nonlinear transformations before projecting back to the original dimension.

The DWConv uses stride=1, padding=1 so spatial dimensions are preserved.
Mix-FFN is a per-token transformation that enriches tokens with neighbor
context, not a spatial reduction operation. Token count in = token count out.
"""

import torch.nn as nn
import torch

class MixFFN(nn.Module):
    """
    Position-aware feed-forward network with depthwise convolution.

    Data flow:
        (B, N, C)
            -> Linear(C, C*E)              expand to hidden dim
        (B, N, C*E)
            -> transpose + reshape         convert to 2D spatial form
        (B, C*E, H, W)
            -> DWConv2d(3x3, groups=C*E)   local spatial mixing per channel
        (B, C*E, H, W)
            -> reshape + transpose         convert back to sequence form
        (B, N, C*E)
            -> GELU                        nonlinearity in the expanded space
        (B, N, C*E)
            -> Linear(C*E, C)              project back to embedding dim
        (B, N, C)

    The GELU is applied after the depthwise conv but before the final linear
    projection. This is standard ordering: the nonlinearity operates in the
    expanded space (C*E dims) where there are more degrees of freedom, giving
    the network richer nonlinear capacity.

    Args:
        embed_dim:        Token embedding dimension C
        expansion_ratio:  Hidden dim multiplier E (default 4)
                          Hidden dim = C * E
                          e.g., embed_dim=64, E=4 -> hidden_dim=256
    """

    def __init__(self, embed_dim, expansion_ratio=4):

        super().__init__()
        hidden_dim = embed_dim * expansion_ratio

        # Linear expansion: C -> C*E
        # Each token independently projected to a wider representation
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        # Depthwise convolution: each of the C*E channels gets its own 3x3 filter
        # groups=hidden_dim means no cross-channel mixing in the conv
        # kernel=3, stride=1, padding=1 preserves spatial dims: H_out = H_in, W_out = W_in
        # This is what gives SegFormer spatial awareness without positional encodings
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups = hidden_dim)

        # Linear projection: C*E -> C
        # Projects back to the original embedding dimension
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

        # GELU activation -- smooth nonlinearity applied in the expanded space
        self.act = nn.GELU()

    def forward(self, x, H, W):
        """
        Args:
            x: Token sequence (B, N, C) where N = H * W
            H: Spatial height of the token grid
            W: Spatial width of the token grid

        Returns:
            projected_x: (B, N, C) -- same shape as input, tokens enriched with
                         local spatial context via depthwise convolution
        """

        # x: (B, N, C)  where C = embed_dim
        #   N = H * W when processing the full grid (no masking, or Stages 2-4)
        #   N < H * W when processing visible tokens after MAE removal (Stage 1)

        # Expand each token to the hidden dimension
        expanded_x = self.fc1(x)
        # expanded_x: (B, N, C*E)
        B, N, C = expanded_x.shape
        # Note: C here is C*E (the hidden dim), not the original embed_dim

        # Depthwise conv requires reshaping to 2D: (B, C*E, H, W)
        # This only works when N == H * W. After MAE token removal, N < H * W.
        #
        # When N == H*W: apply depthwise conv for spatial mixing (standard path)
        # When N != H*W: skip depthwise conv, just use linear layers + activation
        #   The positional mixing from dwconv is a nice-to-have, not essential --
        #   the attention in ESA already provides spatial context. Skipping dwconv
        #   for the ~512 visible tokens at Stage 1 is a minor quality tradeoff
        #   for correct handling of sparse token sets.
        if N == H * W:
            # --- Standard path: reshape to 2D, apply depthwise conv ---
            # (B, N, C*E) -> transpose -> (B, C*E, N) -> reshape -> (B, C*E, H, W)
            reshaped_x = expanded_x.transpose(1, 2)
            reshaped_x = torch.reshape(reshaped_x, (B, C, H, W))
            # reshaped_x: (B, C*E, H, W)

            # Depthwise conv: each channel independently applies a 3x3 spatial filter
            # Token at position (h, w) gets mixed with its 8 spatial neighbors per channel
            convolved_x = self.dwconv(reshaped_x)
            # convolved_x: (B, C*E, H, W)  -- same spatial dims, no reduction

            # Reshape back to sequence form
            # (B, C*E, H, W) -> reshape -> (B, C*E, N) -> transpose -> (B, N, C*E)
            sequence_x = torch.reshape(convolved_x, (B, C, N)).transpose(1, 2)
            # sequence_x: (B, N, C*E)
        else:
            # --- Sparse path: skip depthwise conv (can't form 2D grid) ---
            # (B, N, C*E) passes through unchanged
            sequence_x = expanded_x

        # Nonlinearity in the expanded space
        activated_x = self.act(sequence_x)
        # activated_x: (B, N, C*E)

        # Project back to original embedding dimension
        projected_x = self.fc2(activated_x)
        # projected_x: (B, N, embed_dim)

        return projected_x
