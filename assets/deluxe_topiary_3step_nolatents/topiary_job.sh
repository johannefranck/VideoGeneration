#!/bin/bash
#BSUB -J vidgen1_deluxe_topiary_3step_nolatents
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/deluxe_topiary_3step_nolatents/logs/deluxe_topiary_3step_nolatents.out
#BSUB -e ./assets/deluxe_topiary_3step_nolatents/logs/deluxe_topiary_3step_nolatents.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/deluxe_topiary_3step_nolatents/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/deluxe_topiary_3step_nolatents/generation_config.yaml

