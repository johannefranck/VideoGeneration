#!/bin/bash
#BSUB -J jojojob_videogen             
#BSUB -q gpuv100                  
#BSUB -W 2                        
#BSUB -R "rusage[mem=10000MB]"    
#BSUB -o jojojob%J.out            
#BSUB -e jojojob%J.err            
#BSUB -u jcafranck@gmail.com      
#BSUB -B                          
#BSUB -N                          
#BSUB -n 10 
#BSUB -R "span[hosts=1]"  
#BSUB -W 00:30        

# Load required modules (if necessary)
module load cuda/11.7  # Load CUDA (modify based on your cluster)

# Activate Virtual Environment (venv)
source /dtu/blackhole/1d/155613/venv_video/bin/activate
export PYTHONPATH=/dtu/blackhole/1d/155613/VideoGeneration

# Debugging: Check if venv is active
echo "Activated venv environment: $(which python)"

# Run the Python script
python generate.py --prompt "a teapot floating in water" --input_dir ./data/teapot --input_src pred_128x128.png --edit_mask_path down150.mask.pth --target_flow_name down150.pth --use_cached_latents --save_dir results/teapot128x128.down150 --log_freq 25 --ddim_steps 10

