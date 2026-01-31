#!/bin/bash

# Batch processing script for EchoMimicV3 Flash-Pro
# Keeps models hot in memory for faster processing of multiple videos

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python batch_infer_flash_pro.py \
    --json_path "/workspace/echomimic_v3/media_encoded_05.json" \
    --config_path "/workspace/echomimic_v3/config/config.yaml" \
    --model_name "/workspace/echomimic_v3/flash-pro/Wan2.1-Fun-V1.1-1.3B-InP" \
    --transformer_path "/workspace/echomimic_v3/flash-pro/transformer/diffusion_pytorch_model.safetensors" \
    --wav2vec_model_dir "/workspace/echomimic_v3/flash-pro/chinese-wav2vec2-base" \
    --save_path "outputs" \
    --prompt "Jesus mercyfully speaks to Christian followers" \
    --num_inference_steps 8 \
    --sampler_name "Flow_Unipc" \
    --video_length 330 \
    --guidance_scale 7.0 \
    --audio_guidance_scale 8 \
    --seed 43 \
    --enable_teacache \
    --weight_dtype "bfloat16" \
    --sample_size 512 512 \
    --fps 25

