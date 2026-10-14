import json, sys, csv, shutil, logging
from pathlib import Path
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
for n in ("FoundationTrainer", "HyperspectralSegFormerMAETrainer", "allotrope.pixel_stats"): logging.getLogger(n).setLevel(logging.WARNING)
from app.models.training.training_config import TrainingConfig
from app.foundation_models.trainers.trainer_factory import get_trainer
SP = Path(sys.argv[1]); S3 = "s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/_resume_test/"
base = json.loads(Path("research/experiments/indradhanu-se-pretrain/S01-laptop-smoke/config_cross_attention.json").read_text())
base["device"] = "cuda"; base["model_config"]["amp_bf16"] = True
def run(name, epochs, **mc):
    cfg = json.loads(json.dumps(base)); cfg["data"]["num_epochs"] = epochs; cfg["model_config"].update(mc)
    cfg["model_config"]["metrics_csv"] = str(SP / f"rt_{name}.csv"); cfg["checkpoint"]["checkpoint_dir"] = str(SP / f"rt_ckpt_{name}")
    get_trainer(TrainingConfig(**cfg)).train()
    return [(r["epoch"], round(float(r["val_l1"]), 4), f'{float(r["lr"]):.2e}') for r in csv.DictReader(open(SP / f"rt_{name}.csv"))]
for f in SP.glob("rt_*"): shutil.rmtree(f) if f.is_dir() else f.unlink()
print("(a) early stop, patience 1, needs 50% gains:", run("es", 10, early_stop_patience=1, early_stop_min_epochs=1, early_stop_min_delta=0.5))
print("(b) 2 of 4 epochs, S3:", run("rs", 2, checkpoint_s3_uri=S3))
(SP / "rt_ckpt_rs" / "last.pt").unlink()
print("(c) resume to 4 from S3:", run("rs", 4, checkpoint_s3_uri=S3, auto_resume=True))
