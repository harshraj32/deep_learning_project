#!/bin/bash

# Uninstall existing llama-cpp-python
pip uninstall -y llama-cpp-python

# Install cmake if not present
pip install cmake

# Install with specific options for Metal support (for Mac)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# For CUDA (if you're on Linux/Windows with NVIDIA GPU):
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Install other requirements
pip install streamlit torch sentence-transformers scikit-learn tqdm