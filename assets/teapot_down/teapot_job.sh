#!/bin/bash
#BSUB -J vidgen1_teapot_down
#BSUB -q gpuv100
#BSUB -W 03:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/teapot_down/logs/teapot_down.out
#BSUB -e ./assets/teapot_down/logs/teapot_down.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/teapot_down/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/teapot_down/generation_config.yaml

