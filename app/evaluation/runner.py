"""
Evaluation: run every (dataset scene x model x scorer) of an EvaluationConfig and write the results
(docs/lld/evaluation.md).

    Evaluation(EvaluationConfig.from_file("configs/eval/<name>.json")).run()

Loop order is scene-major, then model-major: a scene is loaded once; each model reconstructs it once, every
scorer for that model runs, and the reconstruction is freed before the next model - so memory holds one scene
and one reconstruction. Class maps (for class_aware / exclude) are computed once per scene per segmenter.
Score maps are cached (float16, <output>/cache/scores) by scene, model file and scorer settings, so adding a
metric or a group re-runs no model; reconstructions are not cached (0.25-1 GB each).

Writes to <output>: config.json, provenance.json (git commit, checkpoint sha256s, dataset manifests),
per_object.csv, per_scene.csv, summary.json and report.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from app.evaluation.config import EvaluationConfig
from app.evaluation.datasets import make_dataset
from app.evaluation.metrics import scene_metrics
from app.evaluation.models import EvalModel, sha256
from app.evaluation.report import summarise, write_report
from app.evaluation.scorers import score

logger = logging.getLogger("Evaluation")
NO_MODEL = "-"


class Evaluation:
    def __init__(self, config: EvaluationConfig, datasets: list | None = None, models: dict | None = None):
        """datasets / models override what the config would build (tests, notebooks): iterables of EvalScene in
        config order, and name -> EvalModel-like objects."""
        self.cfg = config
        self.out = Path(config.output)
        self.cache = self.out / "cache"
        self._datasets, self._models = datasets, models

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(self.cfg.model_dump_json(indent=1))
        models = self._models if self._models is not None else {m.name: EvalModel(m, self.cache) for m in self.cfg.models}
        self._write_provenance(models)
        object_rows, scene_rows = [], []
        sources = self._datasets if self._datasets is not None else [make_dataset(d) for d in self.cfg.datasets]
        for source in sources:
            for scene in source:
                t = time.time()
                if scene.units != "reflectance":
                    raise ValueError(f"{scene.key}: units {scene.units!r}; models read reflectance")
                o, s = self._run_scene(scene, models)
                object_rows += o
                scene_rows += s
                self._write_tables(object_rows, scene_rows)
                logger.info(f"{scene.key}: {len(s)} score maps in {time.time() - t:.0f} s")
                print(f"{scene.key}: {len(s)} score maps in {time.time() - t:.0f} s", flush=True)
                del scene
        summary = summarise(object_rows, scene_rows, self.cfg)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1))
        write_report(summary, self.cfg, self.out / "report.md")
        for m in models.values():
            getattr(m, "release", lambda: None)()
        return summary

    # --- one scene ----------------------------------------------------------------------------------------------

    def _run_scene(self, scene, models: dict) -> tuple[list, list]:
        classes: dict = {}

        def classes_for(segmenter: str):
            if segmenter not in classes:
                p = self.cache / "classes" / scene.dataset / f"{scene.name}__{self._model_id(models[segmenter])}.npz"
                if self.cfg.cache and p.exists():
                    classes[segmenter] = np.load(p)["cls"].astype(np.int64)
                else:
                    classes[segmenter] = models[segmenter].segment(scene)
                    if self.cfg.cache:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(p, cls=classes[segmenter].astype(np.uint8))
            return classes[segmenter]

        object_rows, scene_rows = [], []
        jobs = [(name, [s for s in self.cfg.scorers if s.needs_model]) for name in models]
        jobs.append((NO_MODEL, [s for s in self.cfg.scorers if not s.needs_model]))
        for mname, scorers in jobs:
            if not scorers:
                continue
            model = models.get(mname)
            read = None                                       # (cube, recon, valid), computed only if needed
            for spec in scorers:
                if spec.needs_scale and not getattr(model, "has_scale", False):
                    continue                                  # e.g. scaled_rm on a model without an uncertainty head
                path = self._score_path(scene, model, spec)
                if self.cfg.cache and path.exists():
                    s = np.load(path)["score"].astype(np.float32)
                else:
                    if read is None:
                        read = model.reconstruct(scene) if model else (scene.cube, None, scene.valid, None)
                    cube, recon, valid, *scale = read          # log b only from a model with an uncertainty head
                    log_b = scale[0] if scale else None
                    cls = classes_for(spec.segmenter) if spec.segmenter else None
                    s = score(spec, cube, recon, valid, cls, log_b)
                    if self.cfg.cache:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        # float16 halves the cache; scores past its range (blurred / mixture squared distances
                        # reach 1e5+) are kept float32 so a reload does not turn the top of the ranking into inf.
                        small = np.isfinite(s).all() and np.abs(s).max(initial=0) < 6e4
                        np.savez_compressed(path, score=s.astype(np.float16 if small else np.float32))
                rows, measures = scene_metrics(s, scene.valid, scene.objects, scene.attrs, self.cfg.metrics,
                                               self.cfg.group_by)
                tag = {"dataset": scene.dataset, "scene": scene.name, "sensor": scene.sensor, "model": mname,
                       "scorer": spec.label}
                object_rows += [{**tag, **r} for r in rows]
                scene_rows.append({**tag, **measures})
            del read
        return object_rows, scene_rows

    # --- bookkeeping ---------------------------------------------------------------------------------------------

    def _model_id(self, model) -> str:
        if model is None:
            return NO_MODEL
        if getattr(model, "identity", None):
            return model.identity
        st = model.path.stat()
        key = f"{model.path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{model.mode}"
        s = getattr(model, "spec", None)
        if s is not None and s.adapt_steps:           # adapted variants differ by their settings, not their file
            key += f"|adapt {s.adapt_steps} {s.adapt_lr} {s.adapt_gate_pct} {s.adapt_halo}"
        return f"{model.name}-" + hashlib.sha1(key.encode()).hexdigest()[:10]

    def _score_path(self, scene, model, spec) -> Path:
        key = hashlib.sha1((spec.model_dump_json() + "|" + scene.fingerprint).encode()).hexdigest()[:10]
        return self.cache / "scores" / scene.dataset / scene.name / self._model_id(model) / f"{key}.npz"

    def _write_provenance(self, models: dict) -> None:
        def git(*a):
            try:
                return subprocess.run(["git", *a], capture_output=True, text=True,
                                      cwd=Path(__file__).resolve().parents[2]).stdout.strip()   # repo root
            except OSError:
                return None
        prov = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "app", "scripts")),
                "models": {n: {"checkpoint": m.spec.checkpoint, "sha256": sha256(m.path), "epoch": m.epoch,
                               "type": m.kind, "reading": m.mode} for n, m in models.items() if hasattr(m, "spec")},
                "datasets": [d.model_dump() for d in self.cfg.datasets]}
        for d in self.cfg.datasets:
            mf = Path(d.path) / "manifest.json"
            if mf.exists():
                prov.setdefault("dataset_manifests", {})[d.path] = sha256(mf)
        (self.out / "provenance.json").write_text(json.dumps(prov, indent=1))

    def _write_tables(self, object_rows: list, scene_rows: list) -> None:
        for name, rows in (("per_object.csv", object_rows), ("per_scene.csv", scene_rows)):
            if not rows:
                continue
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(self.out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
