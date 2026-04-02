"""
Token-level masking utilities for SegFormer MAE.

Converts pixel-level validity masks to token-level masks, generates
random prediction masks for training, and checkerboard masks for inference.

Token removal workflow:
    1. pixel_mask_to_token_mask: (B, 1, H, W) pixel mask -> (B, N) token mask
    2. generate_prediction_mask: from valid tokens, select prediction targets
    3. remove_tokens:  gather visible tokens from the full sequence
    4. restore_tokens: scatter encoded tokens back into full grid (zeros at masked positions)

Edge handling:
    erode_mask: shrinks the valid region by a buffer zone around invalid boundaries.
    Pixels near the edge of the valid region have their OPE receptive fields partially
    overlapping with invalid pixels, producing contaminated tokens and unreliable
    reconstructions. Eroding the mask excludes these border pixels from loss and
    anomaly scoring.
"""

import torch
import torch.nn.functional as F


class TokenMasking:
    """
    Static utility methods for token-level masking.
    No learnable parameters -- not an nn.Module.
    """

    @staticmethod
    def erode_mask(mask, kernel_size):
        """
        Erode a validity mask by shrinking the valid region inward from boundaries.

        Pixels within kernel_size//2 of any invalid pixel are marked invalid.
        This creates a buffer zone around invalid regions (clouds, nodata, scene
        edges) where the OPE's receptive field would overlap with invalid pixels,
        producing contaminated tokens and unreliable reconstructions.

        Implementation: dilate the INVALID region using max_pool2d, then invert.
        max_pool2d with stride=1, padding=kernel_size//2 expands any invalid pixel
        (value=1 in the inverted mask) into its neighbors.

        Args:
            mask:        (B, 1, H, W) -- pixel validity (1=valid, 0=invalid)
            kernel_size: erosion radius = kernel_size // 2 pixels
                         Should match OPE kernel_size (7 for Stage 1) so the
                         buffer zone covers the full receptive field overlap.

        Returns:
            eroded_mask: (B, 1, H, W) -- validity mask with border pixels removed

        Example (1D, kernel_size=7, so erosion radius=3):
            mask:        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
                                 ^-- invalid border
            eroded_mask: [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
                                        ^-- 3 pixels eroded from each edge

            The 3 valid pixels adjacent to the invalid boundary are now excluded.
            These are the pixels whose OPE 7x7 kernel would have partially
            overlapped with the invalid region.
        """
        padding = kernel_size // 2
        # Invert: invalid=1, valid=0
        invalid = 1.0 - mask
        # Dilate invalid region: any invalid pixel expands into its neighbors
        # max_pool2d with stride=1 preserves spatial dims
        dilated_invalid = F.max_pool2d(
            invalid, kernel_size=kernel_size, stride=1, padding=padding
        )
        # Invert back: valid=1, invalid=0 (now with eroded border)
        eroded_mask = 1.0 - dilated_invalid
        return eroded_mask

    @staticmethod
    def pixel_mask_to_token_mask(pixel_mask, kernel_size, stride):
        """
        Convert pixel-level validity mask to token-level validity mask.

        Uses average pooling with the SAME kernel_size, stride, and padding
        as the OverlapPatchEmbedding. This ensures each token's validity is
        determined by the exact same pixel region that OPE used to compute
        that token's value. No mismatch between what the token "sees" and
        what determines its validity.

        A token is considered valid if >50% of its receptive field is valid.

        Args:
            pixel_mask:  (B, 1, H, W) -- pixel validity (1=valid, 0=invalid)
            kernel_size: OPE kernel size (7 for Stage 1, 3 for Stages 2-4)
            stride:      OPE stride (4 for Stage 1, 2 for Stages 2-4)

        Returns:
            token_mask: (B, N) -- token validity (1=valid, 0=invalid)
                        where N = (H // stride) * (W // stride)

        Example (B=1, H=16, W=16, kernel_size=7, stride=4):
            OPE Token 0 sees pixels 0-6 (7x7 receptive field)
            OPE Token 1 sees pixels 4-10
            Overlap = kernel - stride = 3 pixels

            avg_pool2d(kernel=7, stride=4, padding=3) -> (1, 1, 4, 4)
            Each value = fraction of valid pixels in that token's 7x7 receptive field
            Threshold > 0.5 -> token valid or not
            Flatten -> (1, 16) token mask
        """
        # Average pool with same params as OPE: fraction of valid pixels
        # in each token's overlapping receptive field.
        # padding = kernel_size // 2 ensures output count = H // stride (same as OPE)
        # (B, 1, H, W) -> (B, 1, H/stride, W/stride)
        #
        # Example: (2, 1, 128, 128) with kernel=7, stride=4, padding=3 -> (2, 1, 32, 32)
        token_fractions = F.avg_pool2d(
            pixel_mask.float(),
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )

        # Threshold: >50% valid pixels -> valid token
        # (B, 1, H/stride, W/stride) -> same shape, but binary
        #
        # Example: 0.75 -> 1.0 (valid), 0.3 -> 0.0 (invalid)
        token_valid = (token_fractions > 0.5).float()

        # Flatten spatial dims and remove channel dim -> (B, N)
        # (B, 1, 32, 32) -> squeeze -> (B, 32, 32) -> flatten -> (B, 1024)
        token_mask = token_valid.squeeze(1).flatten(1)

        return token_mask

    @staticmethod
    def generate_prediction_mask(token_mask, mask_ratio, device):
        """
        From valid tokens alone, randomly select a mask_ratio fraction as
        prediction targets. Invalid tokens are untouched -- they stay in the
        encoder input as zero-signal markers.

        Args:
            token_mask:  (B, N) -- 1=valid, 0=invalid
            mask_ratio:  float, e.g. 0.5 -- fraction of VALID tokens to mask
            device:      torch device

        Returns:
            keep_mask: (B, N) -- 1=keep in encoder (visible valid + all invalid), 0=remove
            pred_mask: (B, N) -- 1=prediction target, 0=everything else

        Example (B=1, N=6, mask_ratio=0.5):
            token_mask = [1, 1, 0, 1, 1, 0]     4 valid tokens, 2 invalid

            noise      = [0.3, 0.7, 0.1, 0.5, 0.2, 0.8]   random
            + invalid*2= [0.3, 0.7, 2.1, 0.5, 0.2, 2.8]   invalid pushed to end

            num_valid = 4, num_to_mask = 4 * 0.5 = 2

            argsort    = [4, 0, 3, 1, 2, 5]    indices sorted by noise ascending
                          ^  ^                   first 2 are prediction targets
                          0.2 0.3                (tokens 4 and 0)

            pred_mask_sorted = [1, 1, 0, 0, 0, 0]  first num_to_mask=2 are 1

            scatter back to original positions:
            pred_mask  = [1, 0, 0, 0, 1, 0]    tokens 0 and 4 are prediction targets
            keep_mask  = [0, 1, 1, 1, 0, 1]    everything else (including invalid) is kept
        """
        B, N = token_mask.shape

        # Generate random noise in [0, 1] for each token
        # (B, N) -- each token gets a random priority
        noise = torch.rand(B, N, device=device)

        # Push invalid tokens to the end of the sort order by adding 2.0
        # Valid tokens have noise in [0, 1], invalid tokens have noise in [2, 3]
        # This guarantees invalid tokens are never selected for masking
        #
        # Example: token_mask = [1, 1, 0, 1]
        #          noise      = [0.3, 0.7, 0.1, 0.5]
        #          adjusted   = [0.3, 0.7, 2.1, 0.5]   <- invalid token 2 pushed to end
        noise = noise + (1 - token_mask) * 2

        # Count valid tokens per batch item
        # (B,) -- e.g. [500, 480] for two batch items
        num_valid = token_mask.sum(dim=1)

        # How many valid tokens to mask as prediction targets
        # (B,) -- e.g. [250, 240] at mask_ratio=0.5
        num_to_mask = (num_valid * mask_ratio).long()

        # Sort tokens by noise -- valid tokens with lowest noise come first
        # (B, N) -- indices that would sort noise in ascending order
        #
        # Example: noise   = [0.3, 0.7, 2.1, 0.5]
        #          argsort = [0, 3, 1, 2]   (0.3 < 0.5 < 0.7 < 2.1)
        sorted_indices = noise.argsort(dim=1)

        # Create position indices [0, 1, 2, ..., N-1] for each batch item
        # (1, N) -> expand -> (B, N)
        #
        # Example (N=4): [[0, 1, 2, 3],
        #                  [0, 1, 2, 3]]
        positions = torch.arange(N, device=device).unsqueeze(0).float().expand(B, -1)

        # In sorted order, the first num_to_mask positions are prediction targets
        # positions < num_to_mask broadcasts: (B, N) < (B, 1) -> (B, N)
        #
        # Example: num_to_mask = [2], positions = [0, 1, 2, 3]
        #          [0<2, 1<2, 2<2, 3<2] = [True, True, False, False]
        #          pred_mask_sorted = [1, 1, 0, 0]  (first 2 sorted tokens are targets)
        pred_mask_sorted = (positions < num_to_mask.unsqueeze(1)).float()

        # Unsort: scatter the sorted prediction mask back to original token positions
        # scatter_(dim=1, index, src): at each position index[b,i], place src[b,i]
        #
        # Example: sorted_indices = [0, 3, 1, 2]
        #          pred_mask_sorted = [1, 1, 0, 0]
        #          scatter: position 0 <- 1, position 3 <- 1, position 1 <- 0, position 2 <- 0
        #          pred_mask = [1, 0, 0, 1]   tokens 0 and 3 are prediction targets
        pred_mask = torch.zeros(B, N, device=device)
        pred_mask.scatter_(1, sorted_indices, pred_mask_sorted)

        # Keep mask: everything that is NOT a prediction target
        # Invalid tokens always get keep=1 (they were never selected for masking)
        # Valid visible tokens get keep=1, prediction targets get keep=0
        keep_mask = 1.0 - pred_mask

        return keep_mask, pred_mask

    @staticmethod
    def checkerboard_token_mask(H_tokens, W_tokens, cell_size, device, invert=False):
        """
        Deterministic checkerboard at token level for inference.
        Invert=True flips the pattern for the second pass.

        Args:
            H_tokens:  token grid height (e.g. 32 for 128x128 input with stride 4)
            W_tokens:  token grid width
            cell_size: checkerboard cell size in tokens (1=individual, 2=2x2 blocks)
            device:    torch device
            invert:    flip pattern for second inference pass

        Returns:
            mask: (1, N) -- 1=visible, 0=masked. Broadcastable over batch dim.

        Example (H_tokens=4, W_tokens=4, cell_size=1, invert=False):
            rows = [0, 1, 2, 3]
            cols = [0, 1, 2, 3]

            rows[:, None] + cols[None, :]:     Broadcasting: (4,1) + (1,4) -> (4,4)
                [[0+0, 0+1, 0+2, 0+3],        [[0, 1, 2, 3],
                 [1+0, 1+1, 1+2, 1+3],    =    [1, 2, 3, 4],
                 [2+0, 2+1, 2+2, 2+3],         [2, 3, 4, 5],
                 [3+0, 3+1, 3+2, 3+3]]         [3, 4, 5, 6]]

            % 2:
                [[0, 1, 0, 1],     Checkerboard!
                 [1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [1, 0, 1, 0]]

            flatten -> (1, 16) = [0,1,0,1, 1,0,1,0, 0,1,0,1, 1,0,1,0]

        Example (cell_size=2):
            rows // 2 = [0, 0, 1, 1]
            cols // 2 = [0, 0, 1, 1]

            sum + % 2:
                [[0, 0, 1, 1],     2x2 block checkerboard!
                 [0, 0, 1, 1],
                 [1, 1, 0, 0],
                 [1, 1, 0, 0]]
        """
        # Integer division by cell_size groups consecutive indices into blocks
        # before the % 2 creates the alternating pattern
        rows = torch.arange(H_tokens, device=device) // cell_size
        cols = torch.arange(W_tokens, device=device) // cell_size

        # Broadcasting: (H,1) + (1,W) -> (H,W)
        # row+col is even -> 0 (masked), odd -> 1 (visible)
        grid = (rows[:, None] + cols[None, :]) % 2  # (H_tokens, W_tokens)

        if invert:
            grid = 1 - grid

        # Flatten to (1, N) for easy broadcasting over batch dimension
        # (H_tokens, W_tokens) -> (H_tokens * W_tokens,) -> (1, N)
        return grid.flatten().unsqueeze(0).float()

    @staticmethod
    def remove_tokens(tokens, keep_mask):
        """
        Gather only the kept tokens from the full sequence, removing prediction targets.

        Args:
            tokens:    (B, N, C) -- full token sequence
            keep_mask: (B, N)    -- 1=keep, 0=remove

        Returns:
            kept_tokens:    (B, max_kept, C) -- only tokens where keep_mask=1
            gather_indices: (B, max_kept)    -- original positions of kept tokens
                            (needed by restore_tokens to scatter back)

        Note: max_kept = max across batch of keep_mask.sum(). With uniform
        masking ratios and similar validity patterns, this is typically the
        same for all batch items. If not, shorter items get a few extra
        tokens from the sort tail (minor leakage).

        Example (B=1, N=6, C=2):
            tokens    = [[t0, t1, t2, t3, t4, t5]]     6 tokens
            keep_mask = [[1,  0,  1,  1,  0,  1]]      keep 4, remove 2

            argsort descending: [0, 2, 3, 5, 1, 4]     1s sort before 0s
            max_kept = 4
            gather_indices = [[0, 2, 3, 5]]             first 4 sorted indices

            expand to (1, 4, 2) for gathering across C dim
            gather from tokens:
                position 0 -> t0
                position 2 -> t2
                position 3 -> t3
                position 5 -> t5

            kept_tokens = [[t0, t2, t3, t5]]            (1, 4, 2)
        """
        B, N, C = tokens.shape

        # Count how many tokens to keep per batch item
        # (B,) -- e.g. [780, 720]
        num_kept = keep_mask.sum(dim=1).long()
        max_kept = num_kept.max().item()

        # Sort indices so kept tokens (keep_mask=1) come first
        # Descending: 1s before 0s
        # (B, N) -- e.g. [3, 0, 7, 2, ...] (indices of 1s first, then 0s)
        sorted_indices = keep_mask.argsort(dim=1, descending=True)

        # Take only the first max_kept indices -- these are all the kept token positions
        # (B, N) -> (B, max_kept)
        #
        # Example: sorted = [0, 2, 3, 5, 1, 4], max_kept=4
        #          gather_indices = [0, 2, 3, 5]
        gather_indices = sorted_indices[:, :max_kept]

        # Expand indices for gathering across the C (embedding) dimension
        # torch.gather needs index shape to match output shape
        # (B, max_kept) -> unsqueeze -> (B, max_kept, 1) -> expand -> (B, max_kept, C)
        #
        # Example (max_kept=4, C=2):
        #   gather_indices = [[0, 2, 3, 5]]
        #   expanded       = [[[0,0], [2,2], [3,3], [5,5]]]
        #   Each index is repeated C times so gather pulls the full embedding vector
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, C)

        # Gather: pull tokens at the specified positions along dim=1 (token axis)
        # (B, N, C) indexed by (B, max_kept, C) -> (B, max_kept, C)
        #
        # Example: tokens[0, 0, :] -> kept[0, 0, :] = t0
        #          tokens[0, 2, :] -> kept[0, 1, :] = t2
        #          tokens[0, 3, :] -> kept[0, 2, :] = t3
        #          tokens[0, 5, :] -> kept[0, 3, :] = t5
        kept_tokens = torch.gather(tokens, dim=1, index=gather_indices_expanded)

        return kept_tokens, gather_indices

    @staticmethod
    def restore_tokens(kept_tokens, gather_indices, N, C):
        """
        Scatter encoded tokens back into the full grid, zeros at removed positions.

        This is the inverse of remove_tokens: place the encoded visible tokens
        back at their original positions in the full N-token grid.

        Args:
            kept_tokens:    (B, N_kept, C) -- encoded visible tokens
            gather_indices: (B, N_kept)    -- original positions (from remove_tokens)
            N:              total token count in the full grid (e.g. 1024)
            C:              embedding dimension

        Returns:
            full_tokens: (B, N, C) -- full grid with encoded tokens at kept positions,
                         zeros at prediction target positions

        Example (B=1, N_kept=4, N=6, C=2):
            kept_tokens    = [[t0', t2', t3', t5']]     4 encoded tokens
            gather_indices = [[0,   2,   3,   5  ]]     their original positions

            full_tokens starts as all zeros: [[0, 0, 0, 0, 0, 0]]   (1, 6, 2)

            scatter_(dim=1):
                position 0 <- t0'
                position 2 <- t2'
                position 3 <- t3'
                position 5 <- t5'

            full_tokens = [[t0', 0, t2', t3', 0, t5']]
                                ^              ^
                            prediction targets remain zero -- decoder must fill these
        """
        B = kept_tokens.shape[0]
        device = kept_tokens.device

        # Start with zeros -- prediction positions will stay zero
        # (B, N, C) -- e.g. (2, 1024, 32)
        full_tokens = torch.zeros(B, N, C, device=device)

        # Expand indices for scattering across C dimension (same pattern as gather)
        # (B, N_kept) -> (B, N_kept, C)
        #
        # Example: [[0, 2, 3, 5]] -> [[[0,0], [2,2], [3,3], [5,5]]]
        scatter_indices = gather_indices.unsqueeze(-1).expand(-1, -1, C)

        # Scatter: place kept_tokens[b, i, :] at position scatter_indices[b, i, :] in full_tokens
        # scatter_(dim=1, index, src): along dim 1 (token axis), at index positions, write src values
        #
        # Example: full_tokens[0, 0, :] = kept_tokens[0, 0, :] = t0'  (because index=0)
        #          full_tokens[0, 2, :] = kept_tokens[0, 1, :] = t2'  (because index=2)
        #          full_tokens[0, 3, :] = kept_tokens[0, 2, :] = t3'  (because index=3)
        #          full_tokens[0, 5, :] = kept_tokens[0, 3, :] = t5'  (because index=5)
        full_tokens.scatter_(1, scatter_indices, kept_tokens)

        return full_tokens
