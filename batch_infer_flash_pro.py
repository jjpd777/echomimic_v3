"""
Batch Processing Script for EchoMimicV3 Flash-Pro
==================================================

This script loads models once and keeps them "hot" in memory for processing
multiple videos efficiently. It reads a JSON file with signed URLs for images
and audio files, downloads them, processes videos, and saves outputs.

JSON Format:
-----------
[
  {
    "portrait_image": "https://signed-url.com/image.png?signature=...",
    "portrait_audio": "https://signed-url.com/audio.wav?signature=...",
    "output_name": "video_001"  // optional, defaults to video_0001, video_0002, etc.
  },
  ...
]

Usage:
------
python batch_infer_flash_pro.py --json_path batch_jobs.json [other options]

Or use the shell script:
bash run_batch_flash_pro.sh

Benefits:
---------
- Models loaded once (saves 2-5 minutes per video)
- Automatic download from signed URLs
- Batch processing with progress tracking
- Automatic cleanup of temporary files
- Error handling and retry logic
"""

import os
import sys
import argparse
import json
import tempfile
import shutil
import requests
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

from src.dist import set_multi_gpus_devices, shard_model
from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as WanTransformer
from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline

from src.utils import (filter_kwargs, get_image_to_video_latent, get_image_to_video_latent2,
                       save_videos_grid)

from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.cache_utils import get_teacache_coefficients

import math
import librosa
from moviepy import VideoFileClip, AudioFileClip
import pyloudnorm as pyln
from transformers import Wav2Vec2FeatureExtractor
from src.wav2vec2 import Wav2Vec2Model
from einops import rearrange


def parse_args():
    parser = argparse.ArgumentParser(description="Batch WanFun Inference with Hot Models")
    
    # JSON input
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file with image/audio URLs")
    
    # Model paths and config
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Config path")
    parser.add_argument("--model_name", type=str, default="/workspace/echomimic_v3/flash-pro/Wan2.1-Fun-V1.1-1.3B-InP", help="Model name")
    parser.add_argument("--transformer_path", type=str, default="/workspace/echomimic_v3/flash-pro/transformer/diffusion_pytorch_model.safetensors", help="Transformer path")
    parser.add_argument("--wav2vec_model_dir", type=str, default="/workspace/echomimic_v3/flash-pro/chinese-wav2vec2-base", help="Wav2Vec model directory")
    parser.add_argument("--save_path", type=str, default="outputs", help="Save path")
    
    # Inference parameters
    parser.add_argument("--prompt", type=str, default="Speaker speaks clearly with clear mouth movement and facial expressions.", help="Text prompt")
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"], help="Sampler name")
    parser.add_argument("--video_length", type=int, default=300, help="Video length")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--audio_guidance_scale", type=float, default=6.0, help="Audio guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    
    # TeaCache parameters
    parser.add_argument("--enable_teacache", action="store_true", default=True, help="Enable TeaCache")
    parser.add_argument("--teacache_threshold", type=float, default=0.1, help="TeaCache threshold")
    parser.add_argument("--num_skip_start_steps", type=int, default=5, help="Number of skip start steps")
    
    # GPU and memory
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Weight dtype")
    parser.add_argument("--sample_size", type=int, nargs=2, default=[512, 512], help="Sample size")
    parser.add_argument("--fps", type=int, default=25, help="FPS")
    parser.add_argument("--negative_prompt", type=str, default="Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. Unclear gestures, broken hands, fused fingers. 手指融合，", help="Negative prompt")
    
    # Download settings
    parser.add_argument("--temp_dir", type=str, default=None, help="Temporary directory for downloads (default: system temp)")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files after processing")
    
    return parser.parse_args()


def download_file(url, local_path, max_retries=3):
    """Download a file from URL to local path with retries."""
    for attempt in range(max_retries):
        try:
            print(f"  Downloading {os.path.basename(local_path)}... (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            print(f"  ✓ Downloaded {os.path.basename(local_path)} ({os.path.getsize(local_path) / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
            else:
                print(f"  ✗ Failed after {max_retries} attempts")
                return False
    return False


def download_all_media(data, temp_dir):
    """Download all media files upfront before processing to avoid URL expiration."""
    print(f"\n{'='*80}")
    print("DOWNLOADING ALL MEDIA FILES (to avoid URL expiration during processing)")
    print(f"{'='*80}\n")
    
    media_files = {}  # Maps output_name -> {'image': path, 'audio': path}
    download_failed = []
    
    for idx, item in enumerate(data, 1):
        # Validate JSON structure
        if 'portrait_image' not in item or 'portrait_audio' not in item:
            print(f"  ✗ Skipping item {idx}: missing 'portrait_image' or 'portrait_audio'")
            download_failed.append(idx)
            continue
        
        image_url = item['portrait_image']
        audio_url = item['portrait_audio']
        output_name = item.get('output_name', f"video_{idx:04d}")
        
        print(f"\n[{idx}/{len(data)}] Downloading media for: {output_name}")
        
        # Determine file extensions from URLs or use defaults
        image_ext = os.path.splitext(image_url.split('?')[0])[1] or '.png'
        audio_ext = os.path.splitext(audio_url.split('?')[0])[1] or '.wav'
        
        local_image_path = os.path.join(temp_dir, f"{output_name}_image{image_ext}")
        local_audio_path = os.path.join(temp_dir, f"{output_name}_audio{audio_ext}")
        
        # Download image
        if not download_file(image_url, local_image_path):
            print(f"  ✗ Failed to download image for {output_name}")
            download_failed.append(idx)
            continue
        
        # Download audio
        if not download_file(audio_url, local_audio_path):
            print(f"  ✗ Failed to download audio for {output_name}")
            download_failed.append(idx)
            # Clean up image if audio failed
            try:
                os.remove(local_image_path)
            except:
                pass
            continue
        
        # Store paths
        media_files[output_name] = {
            'image': local_image_path,
            'audio': local_audio_path,
            'index': idx
        }
    
    print(f"\n{'='*80}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully downloaded: {len(media_files)}/{len(data)}")
    if download_failed:
        print(f"Failed downloads: {len(download_failed)} items")
        print(f"Failed indices: {download_failed}")
    print(f"{'='*80}\n")
    
    return media_files, download_failed


def get_sample_size(pil_img, sample_size):
    w, h = pil_img.size
    ori_a = w * h
    default_a = sample_size[0] * sample_size[1]
    if default_a < ori_a:
        ratio_a = math.sqrt(ori_a / sample_size[0] / sample_size[1])
        w = w / ratio_a // 16 * 16
        h = h / ratio_a // 16 * 16
    else:
        w = w // 16 * 16
        h = h // 16 * 16
    return [int(h), int(w)]


def get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length=81, sr=16000, fps=25, device='cpu'):
    audio_feature = np.squeeze(wav2vec_feature_extractor(mel_input, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)
    
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
    
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    audio_emb = audio_emb.cpu().detach()
    return audio_emb


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def load_models(args):
    """Load all models once - this is the expensive operation."""
    print("=" * 80)
    print("LOADING MODELS (this happens once and keeps them hot)")
    print("=" * 80)
    
    weight_dtype = torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16
    
    # Load audio models
    print("\n[1/6] Loading audio encoder (Wav2Vec2)...")
    audio_encoder = Wav2Vec2Model.from_pretrained(args.wav2vec_model_dir, local_files_only=True).to('cpu')
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model_dir, local_files_only=True)
    print("  ✓ Audio encoder loaded")
    
    # Setup device
    device = set_multi_gpus_devices(1, 1)  # Single GPU
    config = OmegaConf.load(args.config_path)
    
    # Load transformer
    print("\n[2/6] Loading transformer...")
    transformer = WanTransformer.from_pretrained(
        os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    if args.transformer_path:
        print(f"  Loading checkpoint: {args.transformer_path}")
        from safetensors.torch import load_file
        state_dict = load_file(args.transformer_path)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Transformer loaded (missing: {len(m)}, unexpected: {len(u)})")
    else:
        print("  ✓ Transformer loaded")
    
    # Load VAE
    print("\n[3/6] Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    print("  ✓ VAE loaded")
    
    # Load tokenizer
    print("\n[4/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    print("  ✓ Tokenizer loaded")
    
    # Load text encoder
    print("\n[5/6] Loading text encoder (T5)...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    print("  ✓ Text encoder loaded")
    
    # Load CLIP image encoder
    print("\n[6/6] Loading CLIP image encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype)
    clip_image_encoder = clip_image_encoder.eval()
    print("  ✓ CLIP image encoder loaded")
    
    # Create scheduler
    Choosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    if args.sampler_name == "Flow_Unipc" or args.sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    # Create pipeline
    print("\n[Pipeline] Creating pipeline...")
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder
    )
    pipeline.enable_sequential_cpu_offload()
    print("  ✓ Pipeline created with sequential CPU offload")
    
    # Setup TeaCache
    if args.enable_teacache:
        print("\n[TeaCache] Enabling TeaCache...")
        coefficients = get_teacache_coefficients(args.model_name)
        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients, args.num_inference_steps, args.teacache_threshold,
                num_skip_start_steps=args.num_skip_start_steps, offload=False
            )
            print(f"  ✓ TeaCache enabled (threshold: {args.teacache_threshold})")
    
    print("\n" + "=" * 80)
    print("✓ ALL MODELS LOADED - Ready for batch processing!")
    print("=" * 80 + "\n")
    
    return {
        'audio_encoder': audio_encoder,
        'wav2vec_feature_extractor': wav2vec_feature_extractor,
        'pipeline': pipeline,
        'vae': vae,
        'device': device,
        'weight_dtype': weight_dtype,
    }


def process_single_video(args, models, image_path, audio_path, output_name, temp_dir):
    """Process a single video generation."""
    pipeline = models['pipeline']
    audio_encoder = models['audio_encoder']
    wav2vec_feature_extractor = models['wav2vec_feature_extractor']
    vae = models['vae']
    device = models['device']
    weight_dtype = models['weight_dtype']
    
    # Check if output already exists - skip expensive processing if done
    output_video_path = os.path.join(args.save_path, f"{output_name}_output.mp4")
    if os.path.exists(output_video_path):
        print(f"  ⏭️  Skip: {output_video_path} already exists.")
        return True, output_video_path
    
    try:
        # Load reference image
        ref_image = Image.open(image_path).convert("RGB")
        ref_start = np.array(ref_image)
        
        # Load audio
        audio_clip = AudioFileClip(audio_path)
        video_length_actual = min(int(audio_clip.duration * args.fps), args.video_length)
        video_length_actual = int((video_length_actual - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length_actual != 1 else 1
        
        # Get audio features
        mel_input, sr = librosa.load(audio_path, sr=16000)
        mel_input = loudness_norm(mel_input, sr)
        mel_input = mel_input[:int(video_length_actual / 25 * sr)]
        
        print(f"  Audio length: {int(len(mel_input)/ sr * 25)} frames, Video length: {video_length_actual} frames")
        audio_feature_wav2vec = get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length_actual, sr=16000, fps=25, device='cpu')
        
        # Get audio batch
        audio_embeds = audio_feature_wav2vec.to(device=device, dtype=weight_dtype)
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        center_indices = torch.arange(0, video_length_actual, 1).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0]-1)
        audio_embeds = audio_embeds[center_indices]
        audio_embeds = audio_embeds.unsqueeze(0).to(device=device)
        
        validation_image_start = Image.fromarray(ref_start).convert("RGB")
        validation_image_end = None
        latent_frames = (video_length_actual - 1) // vae.config.temporal_compression_ratio + 1
        
        sample_size_0, sample_size_1 = get_sample_size(validation_image_start, args.sample_size)
        input_video, input_video_mask, clip_image = get_image_to_video_latent2(
            validation_image_start, validation_image_end, 
            video_length=video_length_actual, 
            sample_size=[sample_size_0, sample_size_1]
        )
        
        # Generate video
        print("  Generating video...")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        with torch.no_grad():
            sample = pipeline(
                args.prompt,
                num_frames=video_length_actual,
                negative_prompt=args.negative_prompt,
                audio_embeds=audio_embeds,
                audio_scale=1.0,
                ip_mask=None,
                use_un_ip_mask=False,
                height=sample_size_0,
                width=sample_size_1,
                generator=generator,
                neg_scale=1.0,
                neg_steps=0,
                use_dynamic_cfg=False,
                use_dynamic_acfg=False,
                guidance_scale=args.guidance_scale,
                audio_guidance_scale=args.audio_guidance_scale,
                num_inference_steps=args.num_inference_steps,
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
                cfg_skip_ratio=0.0,
                shift=5.0,
            ).videos
        
        # Save video
        tmp_video_path = os.path.join(temp_dir, f"{output_name}_tmp.mp4")
        
        save_videos_grid(sample[:,:,:video_length_actual], tmp_video_path, fps=args.fps)
        
        # Add audio to video
        video_clip = VideoFileClip(tmp_video_path)
        audio_clip = audio_clip.subclipped(0, video_length_actual / args.fps)
        video_clip = video_clip.with_audio(audio_clip)
        video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", threads=2)
        
        # Clean up temporary video
        os.remove(tmp_video_path)
        video_clip.close()
        audio_clip.close()
        
        print(f"  ✓ Saved: {output_video_path}")
        return True, output_video_path
        
    except Exception as e:
        print(f"  ✗ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    args = parse_args()
    
    # Set CUDA memory config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Create temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="echomimic_batch_")
    
    print(f"Temporary directory: {temp_dir}")
    
    # Load JSON file
    print(f"\nReading JSON file: {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print("Error: JSON file must contain a list of objects")
        return
    
    print(f"Found {len(data)} video(s) to process\n")
    
    # STEP 1: Download ALL media files upfront (before models load, to avoid URL expiration)
    media_files, download_failed = download_all_media(data, temp_dir)
    
    if not media_files:
        print("✗ No media files were successfully downloaded. Exiting.")
        return
    
    # STEP 2: Load models once (expensive operation)
    models = load_models(args)
    
    # STEP 3: Process each video using pre-downloaded files
    successful = 0
    failed = len(download_failed)  # Count download failures
    
    for output_name, file_paths in media_files.items():
        idx = file_paths['index']
        local_image_path = file_paths['image']
        local_audio_path = file_paths['audio']
        
        print(f"\n{'='*80}")
        print(f"Processing video {idx}/{len(data)}: {output_name}")
        print(f"{'='*80}")
        print(f"  Using pre-downloaded files:")
        print(f"  Image: {os.path.basename(local_image_path)}")
        print(f"  Audio: {os.path.basename(local_audio_path)}")
        
        # Process video
        success, output_path = process_single_video(
            args, models, local_image_path, local_audio_path, output_name, temp_dir
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Clean up downloaded files after processing
        if not args.keep_temp:
            try:
                os.remove(local_image_path)
                os.remove(local_audio_path)
            except:
                pass
    
    # Clean up temporary directory
    if not args.keep_temp and not args.temp_dir:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(data)}")
    print(f"Failed: {failed}/{len(data)}")
    print(f"Output directory: {args.save_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

