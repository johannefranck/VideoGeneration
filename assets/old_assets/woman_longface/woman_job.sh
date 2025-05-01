#!/bin/bash
#BSUB -J vidgen1_woman_longface
#BSUB -q gpuv100
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -o /work3/s204129/adlicv/project/VideoGeneration/assets/woman_longface/logs/woman_longface.out
#BSUB -e /work3/s204129/adlicv/project/VideoGeneration/assets/woman_longface/logs/woman_longface.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"

# Navigate to project root
cd /work3/s204129/adlicv/project/VideoGeneration || exit 1

# Load dependencies
module load cuda/11.7

# Activate env using absolute path
source /work3/s204129/adlicv/project/VideoGeneration/motion_guidance_env/bin/activate

# Ensure logs dir exists
mkdir -p assets/woman_longface/logs

# Run generation
python ./generate.py assets/woman_longface/generation_config.yaml
