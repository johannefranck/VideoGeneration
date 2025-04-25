# Video Generation

Original Motion Flow code from https://dangeng.github.io/motion_guidance/ by Daniel Geng and Andrew Owens.


remember the taming git repo!

then remember to run ``export PYTHONPATH=$PYTHONPATH:$(pwd)`` to set the path correctly





Run the resample script to downsample or inspect dimension of pred.png or other source images.


Use the generate_latents.py without the --use_cached_latents flag to create new latents using the ``Stable Diffusion v1.4, from CompVis.`` diffusion model. This calls a script ``VideoGeneration/ldm/models/diffusion/ddim_with_grad.py`` using the pretrained weights sd-v1-4.ckpt model architecture is v1-inference.yaml 