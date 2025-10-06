"""
ComfyUI-Ninode-Utils: Utility Nodes for ComfyUI
A collection of utility nodes including OpenAI Compatible Chat
"""

__version__ = "0.9.0"
__author__ = "iGavroche"
__email__ = "nino2k@proton.me"
__description__ = "Utility nodes for ComfyUI including OpenAI Compatible Chat"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

