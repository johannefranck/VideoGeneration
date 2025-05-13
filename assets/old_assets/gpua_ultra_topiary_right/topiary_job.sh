#!/bin/bash
#BSUB -J vidgen1_gpua_ultra_topiary_right
#BSUB -q gpuv100
#BSUB -W 03:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/gpua_ultra_topiary_right/logs/gpua_ultra_topiary_right.out
#BSUB -e ./assets/gpua_ultra_topiary_right/logs/gpua_ultra_topiary_right.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/gpua_ultra_topiary_right/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/gpua_ultra_topiary_right/generation_config.yaml

