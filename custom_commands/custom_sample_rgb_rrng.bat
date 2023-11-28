@echo off
set bat_dir=%~dp0
set model_params=--image_size 64 --num_channels 128 --num_res_blocks 3
set diffusion_params=--diffusion_steps 1500 --noise_schedule linear
set sample_params=--num_samples 216 --batch_size 8
call python %bat_dir%..\scripts\image_sample_rgb_rrng.py --model_path %bat_dir%..\models\custom_model\model.pt %model_params% %diffusion_params% %sample_params%
