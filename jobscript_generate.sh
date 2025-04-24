#!/bin/bash
#BSUB -J jojojob_generate             
#BSUB -q gpuv100                  
#BSUB -W 00:30                     
#BSUB -R "rusage[mem=10000MB]"    
#BSUB -o cool%J.out            
#BSUB -e cool%J.err            
#BSUB -u jcafranck@gmail.com      
#BSUB -B                           # Send email at start
#BSUB -N                           # Send email at end
#BSUB -n 4
#BSUB -R "span[hosts=1]"          

# Load CUDA module (update if needed)
module load cuda/11.7

# Activate your virtual environment
source /dtu/blackhole/1d/155613/venv_video/bin/activate

# Optional: set PYTHONPATH
export PYTHONPATH=/dtu/blackhole/1d/155613/VideoGeneration

## For memory
# Ensure cache/config folders exist
export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"


# Paths
INPUT_DIR=./data/teapot
SAVE_DIR=results/teapot.right.rec_step_1_teapot_floating
DDIM_STEPS=200

# Run your script
python ./generate.py \
    --prompt "a reddish teapot floating in water" \
    --input_dir "$INPUT_DIR" \
    --edit_mask_path down150.mask.pth \
    --target_flow_name down150.pth \
    --use_cached_latents \
    --ddim_steps "$DDIM_STEPS" \
    --save_dir "$SAVE_DIR" \
    --log_freq 5 \
    --num_recursive_steps 1