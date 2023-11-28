@echo off
set bat_dir=%~dp0
set model_params=--image_size 64 --num_channels 128 --num_res_blocks 3
set diffusion_params=--diffusion_steps 1500 --noise_schedule linear
set train_params=--lr 2e-4 --batch_size 6
call python %bat_dir%..\scripts\image_train.py --data_dir %bat_dir%..\datasets\custom_dataset\ %model_params% %diffusion_params% %train_params%
