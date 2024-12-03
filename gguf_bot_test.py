# test_llm.py
import os
from llama_cpp import Llama

def test_model():
    model_path = "Llama-3.2-3B-Instruct-Q5_K_S.gguf"
    
    # Print current working directory and model path
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for model at: {os.path.abspath(model_path)}")
    
    try:
        # Try loading with minimal parameters
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_batch=8,
            n_threads=4,
            n_gpu_layers=0  # Start with CPU only
        )
        
        # Test basic inference
        output = llm("Tell me a joke about programming.", max_tokens=64)
        print("Model loaded successfully!")
        print("Test output:", output)
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    test_model()