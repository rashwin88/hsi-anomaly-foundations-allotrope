# Chapter 02 — Patch Generation and Sharding

This chapter covers how Allotrope converts a fully built
`VendableDataset` (a normalized scene-level cube with a validity
mask) into the training-ready WebDataset shards consumed by the
foundation-model trainers.

The pipeline has four conceptual stages:

1. **Plan**: given a cube shape `(C, H, W)` and a patch geometry
   (height, width, stride), compute the set of top-left corners.
2. **Cut**: slice the cube and its validity mask at every planned
   corner to produce per-patch dictionaries.
3. **Intermediate shard**: stream patches into per-scene tar shards
   on disk, filter by validity, and upload to S3 under a structured
   prefix.
4. **Final shard**: stream all intermediate shards for a sensor/split
   back, shuffle across scene boundaries with a buffer, and write the
   "final" mixed shards that the trainer actually consumes via
   `pipe: aws s3 cp ...` URLs.

Each component gets its own section. Read in order for a guided
walk, or jump to the section that matches your task.

## Section index

- [01_patch_request_and_plan.md](02_patching_and_sharding/01_patch_request_and_plan.md) — the two Pydantic models that pin the planner/cutter contract.
- [02_patch_plan_generator.md](02_patching_and_sharding/02_patch_plan_generator.md) — sliding-window tiling with snap-to-edge, counting formula, alternatives we rejected.
- [03_per_sensor_cutters.md](02_patching_and_sharding/03_per_sensor_cutters.md) — the two generator functions that turn a plan + vendable into patch dicts.
- [04_intermediate_sharder.md](02_patching_and_sharding/04_intermediate_sharder.md) — the abstract base class and the S3-prefix-as-geometry convention.
- [05_per_sensor_sharders.md](02_patching_and_sharding/05_per_sensor_sharders.md) — the three concrete sharders, validity-filter theory, patches-per-shard numerics.
- [06_final_shuffling.md](02_patching_and_sharding/06_final_shuffling.md) — cross-scene shuffling, single vs cross-sensor shufflers, write-back-to-disk tradeoff.
- [07_training_time_pipe.md](02_patching_and_sharding/07_training_time_pipe.md) — the read-side URL builder and why `pipe: aws s3 cp ... -`.
- [08_end_to_end_recap.md](02_patching_and_sharding/08_end_to_end_recap.md) — the whole flow in one place, knob cheat-sheet, sensor-addition litmus test.
