#!/bin/bash
#BSUB -J vidgen1_apple_farright
#BSUB -q gpuv100
#BSUB -W 03:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/apple_farright/logs/apple_farright.out
#BSUB -e ./assets/apple_farright/logs/apple_farright.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1
module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate
mkdir -p assets/apple_farright/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/apple_farright/generation_config.yaml
