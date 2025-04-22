#!/bin/bash
#BSUB -J jojojob_videogen              
#BSUB -q gpuv100                  
#BSUB -W 01:30                     # Wall time: 30 minutes
#BSUB -R "rusage[mem=10000MB]"    
#BSUB -o cool%J.out            
#BSUB -e cool%J.err            
#BSUB -u bbjar2303@gmail.com      
#BSUB -B                           # Send email at start
#BSUB -N                           # Send email at end
#BSUB -n 10 
#BSUB -R "span[hosts=1]"          

# Load CUDA module (update if needed)
module load cuda/11.7

# Activate the local virtual environment
source /work3/s204104/ADLCV/VideoGeneration/vidgen_venv/bin/activate

# Optional: set PYTHONPATH if custom modules are needed
export PYTHONPATH=/work3/s204104/ADLCV/VideoGeneration:$PYTHONPATH

## For memory
export HOME=$PWD

# Ensure cache & config folders exist (just in case)
mkdir -p "$HOME/.cache" "$HOME/.config" "$HOME/.huggingface"


# Run your script
#python ./generate.py \
#    --prompt "an apple on a wooden table" \
#    --input_dir ./data/apple \
#    --edit_mask_path right.mask.pth \
#    --target_flow_name right.pth \
#    --use_cached_latents \
#    --save_dir results/apple.right.rec_step_1 \
#    --log_freq 5 \
#    --num_recursive_steps 1



# Run your script
python ./generate.py \
    --prompt "" \
    --input_dir ./data/teapot \
    --edit_mask_path down150.mask.pth \
    --target_flow_name down150.pth \
    --use_cached_latents \
    --save_dir results/teapot.right.rec_step_1_no_text_prompt \
    --log_freq 5 \
    --num_recursive_steps 1