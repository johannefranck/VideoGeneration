#!/bin/bash
#BSUB -J vidgen1_brandhane
#BSUB -q gpuv100
#BSUB -W 06:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/brandhane\logs/brandhane.out
#BSUB -e ./assets/brandhane\logs/brandhane.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/brandhane/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/brandhane/generation_config.yaml

