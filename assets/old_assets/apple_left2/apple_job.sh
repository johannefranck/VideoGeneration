#!/bin/bash
#BSUB -J vidgen1_apple_left2
#BSUB -q gpuv100
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/apple_left2/logs/apple_left2.out
#BSUB -e ./assets/apple_left2/logs/apple_left2.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

module load cuda/11.7
source motion_guidance_env/bin/activate
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py /work3/s204129/adlicv/project/VideoGeneration/assets/apple_left2/generation_config.yaml
