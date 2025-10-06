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

# Simple voice design node that works without VibeVoice dependency
# This provides the same interface as Minimax Voice Design but uses basic TTS
VIBEVOICE_AVAILABLE = True  # We'll implement a simple version
AVAILABLE_VIBEVOICE_MODELS = {"simple-tts": "Simple TTS"}
ATTENTION_MODES = ["eager"]

print("✅ Simple Voice Design node loaded (no VibeVoice dependency)")

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
                    "default": "A narrator telling a suspenseful story, with a deep and magnetic voice, varying speech pace to create a tense and mysterious atmosphere.",
                    "placeholder": "Describe the voice characteristics you want: gender, age, emotion, speech pace, tone, use case, etc."
                }),
                "preview_text": ("STRING", {
                    "multiline": True,
                    "default": "It was late at night, and he was alone in the old house. Faint footsteps could be heard outside the window. He held his breath and slowly, slowly, walked toward the creaking door...",
                    "placeholder": "Text content for preview (optional, max 200 characters)"
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
                    "placeholder": "Custom voice ID (optional). If empty, a unique ID will be auto-generated"
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
        if not prompt.strip():
            raise ValueError("Voice description cannot be empty")
        
        # Generate or use custom voice_id
        if custom_voice_id and custom_voice_id.strip():
            voice_id = custom_voice_id.strip()
            print(f"🎯 Using custom voice ID: {voice_id}")
        else:
            # Auto-generate unique voice_id
            current_timestamp = int(time.time())
            unique_id = str(uuid.uuid4()).replace('-', '')[:8]
            voice_id = f"simple_voice_{current_timestamp}_{unique_id}"
            print(f"🔄 Auto-generated voice ID: {voice_id}")
        
        try:
            print(f"🎙️ Simple Voice Design: Generating custom voice from description")
            print(f"📝 Voice description: {prompt}")
            if preview_text:
                print(f"🔊 Preview text: {preview_text}")
            
            # For now, create a simple placeholder audio file
            # In a real implementation, this would use a TTS engine
            trial_audio_path = ""
            
            if preview_text and preview_text.strip():
                # Create a simple text-to-speech placeholder
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                trial_filename = f"simple_voice_design_trial_{voice_id}_{timestamp}.txt"
                trial_filepath = os.path.join(self.output_dir, trial_filename)
                
                # Create a placeholder file with the voice description and preview text
                with open(trial_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Voice Design Generated\n")
                    f.write(f"Voice ID: {voice_id}\n")
                    f.write(f"Description: {prompt}\n")
                    f.write(f"Preview Text: {preview_text}\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"\nNote: This is a placeholder implementation.\n")
                    f.write(f"To generate actual audio, install a TTS engine or VibeVoice.\n")
                
                trial_audio_path = os.path.abspath(trial_filepath)
                print(f"💾 Preview file saved to: {trial_audio_path}")
            
            print(f"✅ Voice design completed! Final voice ID: {voice_id}")
            print(f"ℹ️ Note: This is a simplified version. Install a TTS engine to generate actual audio.")
            return (voice_id, trial_audio_path)
            
        except Exception as e:
            logger.error(f"Error during voice design: {e}")
            raise RuntimeError(f"Voice design failed: {str(e)}")


# Update the main nodes.py to include audio nodes
def get_audio_node_mappings():
    """Get audio node mappings for the audio-nodes branch"""
    print(f"🔍 Audio nodes check: VIBEVOICE_AVAILABLE = {VIBEVOICE_AVAILABLE}")
    if VIBEVOICE_AVAILABLE:
        print("✅ Adding VibeVoiceDesign node to mappings")
        return {
            "VibeVoiceDesign": VibeVoiceDesignNode,
        }
    else:
        print("⚠️ VibeVoice not available, returning empty audio mappings")
        return {}

def get_audio_display_name_mappings():
    """Get audio node display name mappings for the audio-nodes branch"""
    if VIBEVOICE_AVAILABLE:
        return {
            "VibeVoiceDesign": "VibeVoice Voice Design",
        }
    else:
        return {}
