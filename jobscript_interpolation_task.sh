#!/bin/bash
#BSUB -J vidgen1           
#BSUB -q gpuv100           
#BSUB -W 01:30                     # Wall time: 30 minutes
#BSUB -R "rusage[mem=20GB]"    
#BSUB -o cool%J.out            
#BSUB -e cool%J.err            

#BSUB -n 1
#BSUB -R "span[hosts=1]"          

# Load CUDA module (update if needed)
module load cuda/11.7

# Activate the local virtual environment
source /work3/s201390/ADLCV/VideoGeneration/
#source motion_guidance_env/bin/activate

# Optional: set PYTHONPATH if custom modules are needed
#export PYTHONPATH=/work3/s204104/ADLCV/VideoGeneration:$PYTHONPATH

## For memory
export HOME=$PWD

# Ensure cache & config folders exist (just in case)
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"


python generate_interpolation_task.py
