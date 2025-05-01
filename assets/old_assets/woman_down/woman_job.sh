#!/bin/bash
#BSUB -J vidgen1_woman_down
#BSUB -q gpuv100
#BSUB -W 02:00
#BSUB -R "rusage[mem=15GB]"
#BSUB -o ./assets/woman_down/logs/woman_down.out
#BSUB -e ./assets/woman_down/logs/woman_down.err

#BSUB -n 1
#BSUB -R "span[hosts=1]"

module load cuda/11.7
source motion_guidance_env/bin/activate
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py /work3/s204129/adlicv/project/VideoGeneration/assets/woman_down/generation_config.yaml
