#!/bin/bash
# Loop map, stage 2 controls: earlier windows on the GPU, then the checks that need no model.
PY=~/ml/eval-venv/bin/python
$PY ~/ml/harness/map/stage2_read.py --offset 1024 --budget 1200
$PY ~/ml/harness/map/stage2_read.py --offset 4096 --budget 1200
$PY ~/ml/harness/map/stage2_controls.py
