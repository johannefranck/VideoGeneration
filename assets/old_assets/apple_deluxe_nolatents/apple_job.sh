#!/bin/bash
#BSUB -J vidgen1_apple_deluxe_nolatents
#BSUB -q gpuv100
#BSUB -W 20:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/apple_deluxe_nolatents/logs/apple_deluxe_nolatents.out
#BSUB -e ./assets/apple_deluxe_nolatents/logs/apple_deluxe_nolatents.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/apple_deluxe_nolatents/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/apple_deluxe_nolatents/generation_config.yaml

