"""
CLI entry point for foundation model training.

Usage:
    python scripts/train_foundation_model.py path/to/config.json
"""

import json
import sys
import logging

from app.models.training.training_config import TrainingConfig
from app.foundation_models.trainers.trainer_factory import get_trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/train_foundation_model.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        raw = json.load(f)

    config = TrainingConfig(**raw)
    trainer = get_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
