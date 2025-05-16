#!/bin/bash
#BSUB -J vidgen1_yoyo_topiary_5step
#BSUB -q gpuv100
#BSUB -W 06:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -gpu "num=1"
#BSUB -o ./assets/yoyo_topiary_5step/logs/yoyo_topiary_5step.out
#BSUB -e ./assets/yoyo_topiary_5step/logs/yoyo_topiary_5step.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

module load cuda/11.7
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

mkdir -p assets/yoyo_topiary_5step/logs
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py assets/yoyo_topiary_5step/generation_config.yaml

