#!/bin/bash
#BSUB -J vidgen1_teapot_down
#BSUB -q gpua100
#BSUB -W 02:00
#BSUB -R "rusage[mem=15GB]"
#BSUB -o ./assets/teapot_down/logs/teapot_down.out
#BSUB -e ./assets/teapot_down/logs/teapot_down.err

#BSUB -n 1
#BSUB -R "span[hosts=1]"

module load cuda/11.7
source motion_guidance_env/bin/activate
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py /work3/s204129/adlicv/project/VideoGeneration/assets/teapot_down/generation_config.yaml
