#!/bin/bash
#BSUB -J vidgen1_test_woman_of_ultragui3
#BSUB -q gpua100
#BSUB -W 08:30
#BSUB -R "rusage[mem=20GB]"
#BSUB -o ./assets/test_woman_of_ultragui3/logs/test_woman_of_ultragui3.out
#BSUB -e ./assets/test_woman_of_ultragui3/logs/test_woman_of_ultragui3.err

#BSUB -n 1
#BSUB -R "span[hosts=1]"

module load cuda/11.7
source motion_guidance_env/bin/activate
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers
python ./generate.py /work3/s204129/adlicv/project/VideoGeneration/assets/test_woman_of_ultragui3/generation_config.yaml
