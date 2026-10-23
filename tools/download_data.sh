#!/bin/bash
# Download the 40 G1-retargeted LAFAN1 motions into data/lafan1_g1 (the robot model ships with mjlab).
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0. With a local copy, link it instead:
#   ln -s /path/to/LAFAN1_Retargeting_Dataset/g1 data/lafan1_g1
set -euo pipefail
cd "$(dirname "$0")/.."
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
mkdir -p data/lafan1_g1
if [ "$(ls data/lafan1_g1/*.csv 2>/dev/null | wc -l)" -ge 40 ]; then
  echo "data/lafan1_g1 already has $(ls data/lafan1_g1/*.csv | wc -l) clips"
  exit 0
fi
for f in $(curl -s "$API/g1" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin)]"); do
  curl -sfL -o "data/lafan1_g1/$(basename "$f")" "$HF/resolve/main/$f"
done
echo "motions: $(ls data/lafan1_g1 | wc -l) clips"
