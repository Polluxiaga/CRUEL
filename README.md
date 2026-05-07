# Gaze2Nav Code

This repository contains the code for **Learning from Human Gaze: Human-like Robot Social Navigation in Dense Crowds (AAAI 2026)**. Gaze2Nav learns from egocentric RGB video, human gaze, pedestrian instance masks, and 2D trajectories to produce more human-like navigation in dense crowds.

## Method Overview

Gaze2Nav follows a three-stage pipeline:

1. **Gaze Predictor** predicts where a human would look in the current egocentric frame from recent RGB frames and previous gaze maps.
2. **Semantic Saliency Matching** matches predicted gaze peaks with tracked pedestrian masks and keeps recently attended pedestrians in a sliding window.
3. **Motion Planner** predicts future waypoints from RGB history plus gaze maps or salient pedestrian masks.

The code also includes ViNT/GNM baselines and gaze/person-aware variants.

## Repository Layout

```text
configs/                  Training and detector/tracker configs
gaze2nav/data/            Dataset loaders and preprocessing scripts
gaze2nav/models/          Gaze, saliency-matching, and action models
gaze2nav/training/        Losses, training loops, logging, visualization
detector/, deep_sort/     Mask R-CNN and Deep SORT tracking components
tracking_utils/           Shared detector/tracker drawing, IO, and logging helpers
run_deep_sort.py          Batch tracking script that exports mask CSV files
train.py                  Main train/generate entrypoint
evaluate_salient_ids.py   Evaluation for generated salient-person IDs
create_data_splits.py     Utility for train/test trajectory splits
```

## Expected Data Format

Each trajectory folder should contain frame-level data:

```text
data/<trajectory_name>/
  0.jpg, 1.jpg, ...
  0.csv, 1.csv, ...          # first row: tracked person IDs; following rows: flattened masks
  traj_data.pkl              # 2D trajectory dataframe
  fixations.pkl              # fixation dataframe, x/y per frame
  person_ids.pkl             # generated from mask CSVs
  select_ids.pkl             # ground-truth salient IDs
```

Split folders contain `traj_names.txt`:

```text
data_splits/train/traj_names.txt
data_splits/test/traj_names.txt
```

Create splits with:

```bash
python create_data_splits.py --data_dir /path/to/data --data_splits_dir /path/to --dataset_name data_splits
```

## Acknowledgements

Parts of the navigation model structure and training workflow are inspired by
[robodhruv/visualnav-transformer](https://github.com/robodhruv/visualnav-transformer).
The tracking stack includes Deep SORT-derived components together with Mask R-CNN
detection code.
