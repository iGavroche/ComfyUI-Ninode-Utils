import torch
import gc
import logging
import os
import time
import uuid
import folder_paths
from typing import Optional, Dict, Any

import comfy.model_management as model_management
from comfy.utils import ProgressBar

# Import VibeVoice components if available
try:
    from ComfyUI-VibeVoice.modules.model_info import AVAILABLE_VIBEVOICE_MODELS
    from ComfyUI-VibeVoice.modules.loader import VibeVoiceModelHandler, ATTENTION_MODES, VIBEVOICE_PATCHER_CACHE, cleanup_old_models
    from ComfyUI-VibeVoice.modules.patcher import VibeVoicePatcher
    from ComfyUI-VibeVoice.modules.utils import parse_script_1_based, preprocess_comfy_audio, set_vibevoice_seed, check_for_interrupt
    VIBEVOICE_AVAILABLE = True
except ImportError:
    VIBEVOICE_AVAILABLE = False
    AVAILABLE_VIBEVOICE_MODELS = {}
    ATTENTION_MODES = ["eager"]

logger = logging.getLogger(__name__)

class VibeVoiceDesignNode:
    """
    VibeVoice-based Voice Design node - Drop-in replacement for Minimax Voice Design
    Generates custom voices from text descriptions using VibeVoice TTS
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        if not VIBEVOICE_AVAILABLE:
            return {
                "required": {
                    "error": ("STRING", {
                        "default": "VibeVoice not available. Please install ComfyUI-VibeVoice custom node.",
                        "multiline": True
                    })
                }
            }
        
        model_names = list(AVAILABLE_VIBEVOICE_MODELS.keys())
        if not model_names:
            model_names.append("No models found in models/tts/VibeVoice")
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "讲述悬疑故事的播音员，声音低沉富有磁性，语速时快时慢，营造紧张神秘的氛围。",
                    "placeholder": "描述您想要的音色特征，如：性别、年龄、情感、语速、音调、使用场景等"
                }),
                "preview_text": ("STRING", {
                    "multiline": True,
                    "default": "夜深了，古屋里只有他一人。窗外传来若有若无的脚步声，他屏住呼吸，慢慢地，慢慢地，走向那扇吱呀作响的门……",
                    "placeholder": "用于试听的文本内容（可选，不超过200字）"
                }),
                "model_name": (model_names, {
                    "tooltip": "Select the VibeVoice model to use. Official models will be downloaded automatically."
                }),
                "attention_mode": (ATTENTION_MODES, {
                    "default": "sdpa",
                    "tooltip": "Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest)"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.3, "min": 0.1, "max": 50.0, "step": 0.05,
                    "tooltip": "Classifier-Free Guidance scale. Higher values increase adherence to the voice prompt."
                }),
                "inference_steps": ("INT", {
                    "default": 10, "min": 1, "max": 500,
                    "tooltip": "Number of diffusion steps for audio generation."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True,
                    "tooltip": "Seed for reproducibility. Set to 0 for a random seed on each run."
                }),
            },
            "optional": {
                "custom_voice_id": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "自定义音色ID（可选）。如果为空，将自动生成唯一ID"
                }),
                "reference_audio": ("AUDIO", {
                    "tooltip": "Reference audio for voice cloning (optional). If provided, will be used as speaker voice."
                }),
                "quantize_llm_4bit": ("BOOLEAN", {
                    "default": False, "label_on": "Q4 (LLM only)", "label_off": "Full precision",
                    "tooltip": "Quantize the Qwen2.5 LLM to 4-bit NF4 via bitsandbytes."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Controls randomness in generation."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling (Top-P)."
                }),
                "force_offload": ("BOOLEAN", {
                    "default": False, "label_on": "Force Offload", "label_off": "Keep in VRAM",
                    "tooltip": "Force model to be offloaded from VRAM after generation."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("voice_id", "trial_audio")
    FUNCTION = "design_voice"
    CATEGORY = "Ninode Utils/Voice Design"
    
    def design_voice(self, prompt, preview_text, model_name, attention_mode, cfg_scale, inference_steps, seed, custom_voice_id="", reference_audio=None, quantize_llm_4bit=False, temperature=0.95, top_p=0.95, force_offload=False):
        if not VIBEVOICE_AVAILABLE:
            raise RuntimeError("VibeVoice not available. Please install ComfyUI-VibeVoice custom node.")
        
        if not prompt.strip():
            raise ValueError("音色描述不能为空")
        
        # Generate or use custom voice_id
        if custom_voice_id and custom_voice_id.strip():
            voice_id = custom_voice_id.strip()
            print(f"🎯 使用自定义音色ID: {voice_id}")
        else:
            # Auto-generate unique voice_id
            current_timestamp = int(time.time())
            unique_id = str(uuid.uuid4()).replace('-', '')[:8]
            voice_id = f"vibevoice_{current_timestamp}_{unique_id}"
            print(f"🔄 自动生成音色ID: {voice_id}")
        
        try:
            print(f"🎙️ VibeVoice Design: 正在根据描述生成定制音色")
            print(f"📝 音色描述: {prompt}")
            if preview_text:
                print(f"🔊 预览文本: {preview_text}")
            
            # Load VibeVoice model
            actual_attention_mode = attention_mode
            if quantize_llm_4bit and attention_mode in ["eager", "flash_attention_2"]:
                actual_attention_mode = "sdpa"
            
            cache_key = f"{model_name}_attn_{actual_attention_mode}_q4_{int(quantize_llm_4bit)}"
            
            if cache_key not in VIBEVOICE_PATCHER_CACHE:
                cleanup_old_models(keep_cache_key=cache_key)
                
                model_handler = VibeVoiceModelHandler(model_name, attention_mode, use_llm_4bit=quantize_llm_4bit)
                patcher = VibeVoicePatcher(
                    model_handler,
                    attention_mode=attention_mode,
                    load_device=model_management.get_torch_device(), 
                    offload_device=model_management.unet_offload_device(),
                    size=model_handler.size
                )
                VIBEVOICE_PATCHER_CACHE[cache_key] = patcher
            
            patcher = VIBEVOICE_PATCHER_CACHE[cache_key]
            model_management.load_model_gpu(patcher)
            model = patcher.model.model
            processor = patcher.model.processor
            
            if model is None or processor is None:
                raise RuntimeError("VibeVoice model and processor could not be loaded. Check logs for errors.")
            
            # Prepare script for generation
            if preview_text and preview_text.strip():
                # Use preview text as the generation script
                script_text = f"[1] {preview_text.strip()}"
            else:
                # Use a default script based on the prompt
                script_text = f"[1] Hello, this is a voice generated from the description: {prompt[:100]}"
            
            # Parse script
            parsed_lines_0_based, speaker_ids_1_based = parse_script_1_based(script_text)
            if not parsed_lines_0_based:
                raise ValueError("Script is empty or invalid. Please provide text to generate.")
            
            # Prepare voice samples
            speaker_inputs = {1: reference_audio}  # Use reference audio if provided
            voice_samples_np = [preprocess_comfy_audio(speaker_inputs.get(sid)) for sid in speaker_ids_1_based]
            
            set_vibevoice_seed(seed)
            
            # Prepare inputs
            inputs = processor(
                parsed_scripts=[parsed_lines_0_based],
                voice_samples=[voice_samples_np], 
                speaker_ids_for_prompt=[speaker_ids_1_based],
                padding=True,
                return_tensors="pt", 
                return_attention_mask=True
            )
            
            # Validate inputs
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                        logger.error(f"Input tensor '{key}' contains NaN or Inf values")
                        raise ValueError(f"Invalid values in input tensor: {key}")
            
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            model.set_ddpm_inference_steps(num_steps=inference_steps)
            
            # Generation config
            generation_config = {
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p
            }
            
            # Generate audio
            with torch.no_grad():
                pbar = ProgressBar(inference_steps)
                
                def progress_callback(step, total_steps):
                    pbar.update(1)
                    if model_management.interrupt_current_processing:
                        raise model_management.InterruptProcessingException()
                
                try:
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=None, 
                        cfg_scale=cfg_scale,
                        tokenizer=processor.tokenizer, 
                        generation_config=generation_config,
                        verbose=False, 
                        stop_check_fn=check_for_interrupt
                    )
                    pbar.update(inference_steps - pbar.current)
                    
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "assertion" in error_msg or "cuda" in error_msg:
                        logger.error(f"CUDA assertion failed with {attention_mode} attention: {e}")
                        logger.error("Try restarting ComfyUI or switching to 'eager' attention mode.")
                    raise e
                except model_management.InterruptProcessingException:
                    logger.info("VibeVoice generation interrupted by user")
                    raise
                finally:
                    pbar.update_absolute(inference_steps)
            
            # Process output
            output_waveform = outputs.speech_outputs[0]
            if output_waveform.ndim == 1: 
                output_waveform = output_waveform.unsqueeze(0)
            if output_waveform.ndim == 2: 
                output_waveform = output_waveform.unsqueeze(0)
            
            # Save trial audio
            trial_audio_path = ""
            if output_waveform is not None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                trial_filename = f"vibevoice_design_trial_{voice_id}_{timestamp}.wav"
                trial_filepath = os.path.join(self.output_dir, trial_filename)
                
                # Convert tensor to audio file
                import torchaudio
                output_waveform_cpu = output_waveform.detach().cpu()
                torchaudio.save(trial_filepath, output_waveform_cpu, 24000)
                
                trial_audio_path = os.path.abspath(trial_filepath)
                print(f"💾 试听音频保存至: {trial_audio_path}")
            
            # Force offload if requested
            if force_offload:
                logger.info(f"Force offloading VibeVoice model '{model_name}' from VRAM...")
                if patcher.is_loaded:
                    patcher.unpatch_model(unpatch_weights=True)
                model_management.unload_all_models()
                gc.collect()
                model_management.soft_empty_cache()
                logger.info("Model force offload completed")
            
            print(f"✅ 音色生成成功！最终音色ID: {voice_id}")
            return (voice_id, trial_audio_path)
            
        except model_management.InterruptProcessingException:
            logger.info("VibeVoice TTS generation was cancelled")
            return (voice_id, "")
        
        except Exception as e:
            logger.error(f"Error during VibeVoice generation: {e}")
            if "interrupt" in str(e).lower() or "cancel" in str(e).lower():
                logger.info("Generation was interrupted")
                return (voice_id, "")
            raise RuntimeError(f"音色设计失败: {str(e)}")


# Update the main nodes.py to include audio nodes
def get_audio_node_mappings():
    """Get audio node mappings for the audio-nodes branch"""
    if VIBEVOICE_AVAILABLE:
        return {
            "VibeVoiceDesign": VibeVoiceDesignNode,
        }
    else:
        return {}

def get_audio_display_name_mappings():
    """Get audio node display name mappings for the audio-nodes branch"""
    if VIBEVOICE_AVAILABLE:
        return {
            "VibeVoiceDesign": "VibeVoice Voice Design",
        }
    else:
        return {}
