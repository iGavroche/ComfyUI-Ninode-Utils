# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.2] - 2024-12-19

### Changed
- Updated display name to "Ninode Utils" for cleaner ComfyUI sidebar appearance
- Simplified node library label from "ComfyUI Ninode Utils" to "Ninode Utils"

## [0.9.1] - 2024-12-19

### Changed
- Updated default API URL to Lemonade Server standard (`http://127.0.0.1:8000/api/v1/chat/completions`)
- Added link to [Lemonade Server documentation](https://lemonade-server.ai/docs/server) in README
- Updated tooltip to reference Lemonade Server instead of generic Lemonade

## [0.9.0] - 2024-12-19

### Added
- Initial release of ComfyUI-Ninode-Utils
- OpenAI Compatible Chat Node with complete input compatibility
- Support for multiple OpenAI-compatible API endpoints:
  - Lemonade server (`http://127.0.0.1:35841/v1/chat/completions`)
  - Ollama (`http://localhost:11434/v1/chat/completions`)
  - OpenAI API (`https://api.openai.com/v1/chat/completions`)
  - LM Studio (`http://localhost:1234/v1/chat/completions`)
  - Any other OpenAI-compatible service
- Multi-turn conversation support with context persistence
- Image support for vision models
- File support for document analysis
- Advanced options configuration
- System message support
- Comprehensive test suite:
  - Basic functionality tests
  - Input compatibility tests
  - Advanced options handling tests
  - Tensor processing tests
- Robust error handling and edge case management
- Complete documentation and examples

### Features
- **Complete Input Compatibility**: All inputs from the original ChatGPT node are supported
- **Flexible API Configuration**: Works with any OpenAI-compatible endpoint
- **Multi-turn Conversations**: Context persistence for ongoing conversations
- **Image Support**: Vision model support with image inputs
- **File Support**: Document analysis with file inputs
- **Advanced Options**: Fine-tuned control with advanced configuration
- **Configurable Parameters**: Temperature, top_p, max_tokens, system messages
- **Error Handling**: Robust error handling for API failures and edge cases

### Technical Details
- Built with Python 3.9+
- Uses aiohttp for async HTTP requests
- Supports various tensor shapes and data types
- Handles Pydantic models and dictionary inputs
- Graceful fallback for unsupported tensor shapes
- Comprehensive logging and error reporting

### Dependencies
- aiohttp>=3.8.0
- torch>=2.0.0
- Pillow>=9.0.0
- numpy>=1.25.0
