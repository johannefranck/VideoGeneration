#!/bin/bash
#BSUB -J vidgen1_deluxe_apple_3step_latents
#BSUB -q gpuv100
#BSUB -W 03:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/deluxe_apple_3step_latents/logs/deluxe_apple_3step_latents.out
#BSUB -e ./assets/deluxe_apple_3step_latents/logs/deluxe_apple_3step_latents.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/deluxe_apple_3step_latents/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/deluxe_apple_3step_latents/generation_config.yaml

