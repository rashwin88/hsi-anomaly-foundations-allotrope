"""Train the spectral namer: python scripts/train_namer.py configs/namer/N001.json (app/spectral_match/namer/train.py)."""

import sys

from app.spectral_match.namer.train import main

if __name__ == "__main__":
    main(sys.argv[1])
