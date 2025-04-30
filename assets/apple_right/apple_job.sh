#!/bin/bash
#BSUB -J vidgen1_apple_right
#BSUB -q gpua100
#BSUB -W 01:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -o ./assets/apple_right/logs/apple_right.out
#BSUB -e ./assets/apple_right/logs/apple_right.err

#BSUB -n 1
#BSUB -R "span[hosts=1]"

module load cuda/11.7
source motion_guidance_env/bin/activate
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py /work3/s204129/adlicv/project/VideoGeneration/assets/apple_right/generation_config.yaml
