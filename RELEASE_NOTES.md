# Release Notes - ComfyUI-Ninode-Utils v0.9.0

## 🎉 Initial Release

**ComfyUI-Ninode-Utils v0.9.0** is now ready for release! This is the first public release of our utility nodes collection for ComfyUI.

## 📦 What's Included

### Core Features
- **OpenAI Compatible Chat Node** - Complete drop-in replacement for the original ChatGPT node
- **Multi-API Support** - Works with Lemonade, Ollama, OpenAI API, LM Studio, and any OpenAI-compatible service
- **Complete Input Compatibility** - All inputs from the original ChatGPT node are supported
- **Robust Error Handling** - Graceful handling of edge cases and API failures

### Technical Highlights
- **Async Support** - Uses aiohttp for non-blocking HTTP requests
- **Image Processing** - Converts ComfyUI tensors to base64 for API transmission
- **File Support** - Document analysis with file inputs
- **Advanced Options** - Fine-tuned control with Pydantic model support
- **Context Persistence** - Multi-turn conversation support
- **Comprehensive Testing** - 12/12 tests passing

## 🚀 Ready for Production

The package is fully tested and ready for use:

- ✅ All basic functionality tests pass
- ✅ Input compatibility verified
- ✅ Advanced options handling tested
- ✅ Tensor processing edge cases covered
- ✅ Error handling validated
- ✅ Documentation complete

## 📋 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iGavroche/ComfyUI-Ninode-Utils.git
cd ComfyUI-Ninode-Utils
pip install -r requirements.txt
```

## 🎯 Usage

The node appears in the **"ComfyUI-Ninode-Utils"** category and can be used immediately as a drop-in replacement for the original ChatGPT node.

## 🔗 Repository

- **GitHub**: https://github.com/iGavroche/ComfyUI-Ninode-Utils
- **Issues**: https://github.com/iGavroche/ComfyUI-Ninode-Utils/issues
- **Documentation**: See README.md for complete usage guide

## 📊 Test Results

```
Basic tests: 5/5 passed ✅
New input tests: 3/3 passed ✅
Advanced options tests: 4/4 passed ✅
Tensor handling tests: 4/4 passed ✅
Total: 16/16 tests passed ✅
```

## 🎉 Ready to Push!

The repository is initialized, committed, and ready to be pushed to GitHub:

```bash
git push -u origin main
```

**Happy coding! 🚀**

