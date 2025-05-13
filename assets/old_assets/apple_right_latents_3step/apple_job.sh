#!/bin/bash
#BSUB -J vidgen1_apple_right_latents_3step
#BSUB -q gpuv100
#BSUB -W 03:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/apple_right_latents_3step/logs/apple_right_latents_3step.out
#BSUB -e ./assets/apple_right_latents_3step/logs/apple_right_latents_3step.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/apple_right_latents_3step/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/apple_right_latents_3step/generation_config.yaml

