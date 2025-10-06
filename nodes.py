"""
Nino-utils: Utility Nodes for ComfyUI
OpenAI Compatible Chat Node - Compatible with OpenAI API endpoints
"""

import json
import uuid
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Tuple
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
import torch
from PIL import Image
import base64
import io


class HistoryEntry:
    """Represents a single entry in the chat history."""
    def __init__(self, role: str, content: str, response_id: Optional[str] = None):
        self.role = role
        self.content = content
        self.response_id = response_id


class OpenAICompatibleChatNode(ComfyNodeABC):
    """
    OpenAI Compatible Chat Node - Works with any OpenAI-compatible API endpoint
    Supports local servers like Lemonade, Ollama, and other OpenAI-compatible services
    """

    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "Ninode Utils/Chat"
    DESCRIPTION = "OpenAI Compatible Chat Node - Works with any OpenAI-compatible API endpoint"
    FUNCTION = "api_call"

    def __init__(self) -> None:
        """Initialize the chat node with a new session ID and empty history."""
        self.current_session_id: str = str(uuid.uuid4())
        self.history: Dict[str, List[HistoryEntry]] = {}
        self.previous_response_id: Optional[str] = None

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text inputs to the model, used to generate a response.",
                    },
                ),
                "api_url": (
                    IO.STRING,
                    {
                        "default": "http://127.0.0.1:8000/api/v1/chat/completions",
                        "tooltip": "OpenAI-compatible API endpoint URL (e.g., http://127.0.0.1:8000/api/v1/chat/completions for Lemonade Server, or http://localhost:11434/v1/chat/completions for Ollama)",
                    },
                ),
                "model": (
                    IO.STRING,
                    {
                        "default": "gpt-4o-mini",
                        "tooltip": "Model name to use (e.g., gpt-4o-mini, llama3.2, etc.)",
                    },
                ),
                "api_key": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": "API key for authentication (leave empty if not required)",
                    },
                ),
                "persist_context": (
                    IO.BOOLEAN,
                    {
                        "default": True,
                        "tooltip": "Persist chat context between calls (multi-turn conversation)",
                    },
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional image(s) to use as context for the model. To include multiple images, you can use the Batch Images node.",
                    },
                ),
                "files": (
                    "OPENAI_INPUT_FILES",
                    {
                        "default": None,
                        "tooltip": "Optional file(s) to use as context for the model. Accepts inputs from the OpenAI Chat Input Files node.",
                    },
                ),
                "advanced_options": (
                    "OPENAI_CHAT_CONFIG",
                    {
                        "default": None,
                        "tooltip": "Optional configuration for the model. Accepts inputs from the OpenAI Chat Advanced Options node.",
                    },
                ),
                "max_tokens": (
                    IO.INT,
                    {
                        "default": 1000,
                        "min": 1,
                        "max": 100000,
                        "tooltip": "Maximum number of tokens to generate",
                    },
                ),
                "temperature": (
                    IO.FLOAT,
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Controls randomness in the response (0.0 = deterministic, 2.0 = very random)",
                    },
                ),
                "top_p": (
                    IO.FLOAT,
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Controls diversity via nucleus sampling (0.0 = only most likely tokens, 1.0 = all tokens)",
                    },
                ),
                "system_message": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional system message to set the behavior of the assistant",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """Convert a tensor to base64 string for image transmission."""
        try:
            # Convert tensor to PIL Image
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # Remove batch dimension if present
            
            # Handle different tensor dimensions
            if tensor.dim() == 3:
                # Check if it's a valid image tensor (H, W, C) or (C, H, W)
                h, w, c = tensor.shape
                if c in [1, 3, 4]:  # Grayscale, RGB, or RGBA
                    # Already in HWC format
                    pass
                elif h in [1, 3, 4] and w > 1:  # Likely CHW format
                    # Convert from CHW to HWC
                    tensor = tensor.permute(1, 2, 0)
                else:
                    # Unusual shape, try to reshape or skip
                    raise ValueError(f"Unsupported tensor shape for image conversion: {tensor.shape}")
            elif tensor.dim() == 2:
                # Grayscale image, add channel dimension
                tensor = tensor.unsqueeze(-1)
            else:
                raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")
            
            # Convert to numpy and scale to 0-255
            numpy_array = tensor.cpu().numpy()
            
            # Ensure the array is contiguous and has the right dtype
            if not numpy_array.flags.c_contiguous:
                numpy_array = numpy_array.copy()
            
            # Handle different data types and ranges
            if numpy_array.dtype == 'uint8':
                # Already in the right format
                pass
            elif numpy_array.dtype in ['float32', 'float64']:
                if numpy_array.max() <= 1.0:
                    numpy_array = (numpy_array * 255).astype('uint8')
                else:
                    numpy_array = numpy_array.astype('uint8')
            else:
                # Convert to uint8
                numpy_array = numpy_array.astype('uint8')
            
            # Ensure the array has the right shape for PIL
            if numpy_array.shape[-1] == 1:
                # Grayscale image
                numpy_array = numpy_array.squeeze(-1)
            elif numpy_array.shape[-1] == 3:
                # RGB image
                pass
            elif numpy_array.shape[-1] == 4:
                # RGBA image
                pass
            else:
                raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}")
            
            # Convert to PIL Image
            if len(numpy_array.shape) == 2:
                # Grayscale
                image = Image.fromarray(numpy_array, mode='L')
            elif len(numpy_array.shape) == 3 and numpy_array.shape[-1] == 3:
                # RGB
                image = Image.fromarray(numpy_array, mode='RGB')
            elif len(numpy_array.shape) == 3 and numpy_array.shape[-1] == 4:
                # RGBA
                image = Image.fromarray(numpy_array, mode='RGBA')
            else:
                raise ValueError(f"Unsupported array shape for PIL: {numpy_array.shape}")
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            # If image conversion fails, create a placeholder or skip
            print(f"Warning: Could not convert tensor to image: {e}")
            print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            
            # Create a small placeholder image
            placeholder = Image.new('RGB', (64, 64), color='gray')
            buffer = io.BytesIO()
            placeholder.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

    def create_messages(self, prompt: str, images: Optional[torch.Tensor] = None, 
                       files: Optional[List[Dict[str, Any]]] = None,
                       system_message: str = "", session_id: str = "") -> List[Dict[str, Any]]:
        """Create the messages array for the API call."""
        messages = []
        
        # Add system message if provided
        if system_message.strip():
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add conversation history if persist_context is enabled
        if session_id in self.history:
            for entry in self.history[session_id]:
                messages.append({
                    "role": entry.role,
                    "content": entry.content
                })
        
        # Create current user message
        user_content = []
        
        # Add text content
        if prompt.strip():
            user_content.append({
                "type": "text",
                "text": prompt
            })
        
        # Add image content if provided
        if images is not None:
            try:
                if images.dim() == 4:  # Batch of images
                    for i in range(images.shape[0]):
                        img_b64 = self.tensor_to_base64(images[i])
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": img_b64
                            }
                        })
                else:  # Single image
                    img_b64 = self.tensor_to_base64(images)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": img_b64
                        }
                    })
            except Exception as e:
                print(f"Warning: Skipping image processing due to error: {e}")
                # Continue without images rather than failing completely
        
        # Add file content if provided
        if files is not None:
            for file_content in files:
                if isinstance(file_content, dict) and "file_data" in file_content:
                    user_content.append({
                        "type": "text",
                        "text": f"File: {file_content.get('filename', 'unknown')}\n{file_content['file_data']}"
                    })
        
        messages.append({
            "role": "user",
            "content": user_content if len(user_content) > 1 else user_content[0]["text"] if user_content else ""
        })
        
        return messages

    def add_to_history(self, session_id: str, prompt: str, response: str, response_id: Optional[str] = None):
        """Add a conversation turn to the history."""
        if session_id not in self.history:
            self.history[session_id] = []
        
        # Add user message
        self.history[session_id].append(HistoryEntry("user", prompt))
        
        # Add assistant response
        self.history[session_id].append(HistoryEntry("assistant", response, response_id))

    def display_history_on_node(self, session_id: str, unique_id: Optional[str] = None):
        """Display the conversation history on the node (placeholder for future implementation)."""
        if session_id in self.history:
            history_text = f"Session: {session_id}\n"
            for entry in self.history[session_id][-6:]:  # Show last 6 entries
                history_text += f"{entry.role}: {entry.content[:100]}...\n"
            print(f"Chat History for {unique_id}:\n{history_text}")

    async def make_api_call(self, api_url: str, api_key: str, model: str, 
                           messages: List[Dict[str, Any]], max_tokens: int, 
                           temperature: float, top_p: float, 
                           advanced_options: Optional[Any] = None) -> Tuple[str, Optional[str]]:
        """Make the actual API call to the OpenAI-compatible endpoint."""
        headers = {
            "Content-Type": "application/json",
        }
        
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Start with basic payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        # Override with advanced_options if provided
        if advanced_options:
            # Convert Pydantic model to dictionary if needed
            if hasattr(advanced_options, 'model_dump'):
                # It's a Pydantic model
                options_dict = advanced_options.model_dump(exclude_none=True)
            elif hasattr(advanced_options, 'items'):
                # It's already a dictionary
                options_dict = advanced_options
            else:
                # Try to convert to dict, but handle edge cases gracefully
                try:
                    options_dict = dict(advanced_options)
                except (TypeError, ValueError):
                    # If conversion fails, skip advanced options
                    print(f"Warning: Could not convert advanced_options to dictionary: {type(advanced_options)}")
                    options_dict = {}
            
            for key, value in options_dict.items():
                if value is not None:
                    payload[key] = value
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract response content
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            response_id = result.get("id")
                            return content, response_id
                        else:
                            raise Exception(f"No choices in response: {result}")
                    else:
                        error_text = await response.text()
                        raise Exception(f"API call failed with status {response.status}: {error_text}")
            except Exception as e:
                raise Exception(f"API call failed: {str(e)}")

    def api_call(
        self,
        prompt: str,
        api_url: str,
        model: str,
        api_key: str,
        persist_context: bool,
        unique_id: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
        files: Optional[List[Dict[str, Any]]] = None,
        advanced_options: Optional[Any] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        system_message: str = "",
        **kwargs,
    ) -> Tuple[str]:
        """Main API call method."""
        # Validate inputs
        if not prompt.strip() and images is None:
            raise ValueError("Either prompt or images must be provided")
        
        if not api_url.strip():
            raise ValueError("API URL is required")
        
        # Determine session ID
        if persist_context:
            session_id = self.current_session_id
        else:
            session_id = str(uuid.uuid4())
        
        # Get previous response ID for context
        if persist_context and self.previous_response_id:
            previous_response_id = self.previous_response_id
        else:
            previous_response_id = None
        
        # Create messages
        messages = self.create_messages(prompt, images, files, system_message, session_id)
        
        # Make API call
        try:
            # Run the async function in the event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we need to use a different approach
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.make_api_call(api_url, api_key, model, messages, max_tokens, temperature, top_p, advanced_options)
                    )
                    output_text, response_id = future.result()
            else:
                output_text, response_id = loop.run_until_complete(
                    self.make_api_call(api_url, api_key, model, messages, max_tokens, temperature, top_p, advanced_options)
                )
        except Exception as e:
            raise Exception(f"Failed to make API call: {str(e)}")
        
        # Update history
        if persist_context:
            self.add_to_history(session_id, prompt, output_text, response_id)
            self.display_history_on_node(session_id, unique_id)
            self.previous_response_id = response_id
        
        return (output_text,)


# Import audio nodes
try:
    from .nodes_audio import get_audio_node_mappings, get_audio_display_name_mappings
    audio_node_mappings = get_audio_node_mappings()
    audio_display_name_mappings = get_audio_display_name_mappings()
    print("✅ Audio nodes loaded successfully")
except Exception as e:
    print(f"⚠️ Audio nodes import failed: {e} - continuing without audio nodes")
    audio_node_mappings = {}
    audio_display_name_mappings = {}

# Node mappings
NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleChatNode": OpenAICompatibleChatNode,
    **audio_node_mappings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleChatNode": "OpenAI Compatible Chat",
    **audio_display_name_mappings
}
