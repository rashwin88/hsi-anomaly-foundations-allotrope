import json, sys, csv, shutil, logging
from pathlib import Path
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)
from app.models.training.training_config import TrainingConfig
from app.foundation_models.trainers.trainer_factory import get_trainer
from app.foundation_models.trainers.spectral_stem_mae_trainer import SpectralStemMAETrainer
SP = Path(sys.argv[1])
base = json.loads(Path("research/experiments/indradhanu-se-pretrain/S01-laptop-smoke/config_cross_attention.json").read_text())
base["device"] = "cuda"; base["model_config"]["amp_bf16"] = True; base["data"]["num_epochs"] = 4
def cfg(name, **mc):
    c = json.loads(json.dumps(base)); c["model_config"].update(mc, metrics_csv=str(SP / f"rt2_{name}.csv")); c["checkpoint"]["checkpoint_dir"] = str(SP / f"rt2_ckpt_{name}")
    return TrainingConfig(**c)
lrs = lambda name: [f'{float(r["lr"]):.3e}' for r in csv.DictReader(open(SP / f"rt2_{name}.csv"))]
for f in SP.glob("rt2_*"): shutil.rmtree(f) if f.is_dir() else f.unlink()
get_trainer(cfg("straight")).train()
class Killed(Exception): pass
orig = SpectralStemMAETrainer._run_epoch
def dying(self, epoch, *a):
    if epoch == 2: raise Killed()
    return orig(self, epoch, *a)
SpectralStemMAETrainer._run_epoch = dying
try: get_trainer(cfg("killed", auto_resume=True)).train()
except Killed: print("killed before epoch 3")
SpectralStemMAETrainer._run_epoch = orig
get_trainer(cfg("killed", auto_resume=True)).train()
print("uninterrupted lr:", lrs("straight")); print("killed+resumed lr:", lrs("killed"))
