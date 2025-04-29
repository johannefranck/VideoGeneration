#!/bin/bash
#BSUB -J vidgen1_apple4     
#BSUB -q gpua100           
#BSUB -W 08:30                     # Wall time: 30 minutes
#BSUB -R "rusage[mem=20GB]"    
#BSUB -o apple4(%J.out            
#BSUB -e apple4(%J.err            

#BSUB -n 1
#BSUB -R "span[hosts=1]"  
        

# Load CUDA module (update if needed)
module load cuda/11.7

# Activate the local virtual environment
#source /work3/s201390/ADLCV/VideoGeneration/
source motion_guidance_env/bin/activate

# Optional: set PYTHONPATH if custom modules are needed
#export PYTHONPATH=/work3/s204104/ADLCV/VideoGeneration:$PYTHONPATH

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


#export PYTHONPATH=$PYTHONPATH:/work3/s201390/VideoGeneration/motion_guidance_env/src/taming-transformers
export PYTHONPATH=/VideoGeneration/motion_guidance_env/src/taming-transformers

export PYTHONPATH=/work3/s201390/VideoGeneration/motion_guidance_env/src/taming-transformers
# Run your script
python ./generate.py