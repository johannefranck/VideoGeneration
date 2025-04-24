#!/bin/bash
#BSUB -J jojojob_latents              
#BSUB -q gpuv100                  
#BSUB -W 00:30                     
#BSUB -R "rusage[mem=10000MB]"    
#BSUB -o latent_job%J.out            
#BSUB -e latent_job%J.err            
#BSUB -u jcafranck@gmail.com      
#BSUB -B                           
#BSUB -N                           
#BSUB -n 4 
#BSUB -R "span[hosts=1]"

# Load CUDA module (adjust if needed)
module load cuda/11.7

# Activate your virtual environment
source /dtu/blackhole/1d/155613/venv_video/bin/activate

# Optional: set PYTHONPATH
export PYTHONPATH=/dtu/blackhole/1d/155613/VideoGeneration

# Ensure cache/config folders exist
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"

# Set input/output info
INPUT_IMG=./data/teapot/pred.png
LATENT_OUTPUT_DIR=./data/teapot/latents
DDIM_STEPS=200

# Optional: clean up existing latents
if [ -d "$LATENT_OUTPUT_DIR" ]; then
    rm -f "$LATENT_OUTPUT_DIR"/zt.*.pth
    echo "Old latents cleared from $LATENT_OUTPUT_DIR"
fi

# Generate latents
python ./generate_latents.py \
    --input "$INPUT_IMG" \
    --output_dir "$LATENT_OUTPUT_DIR" \
    --ddim_steps "$DDIM_STEPS"
