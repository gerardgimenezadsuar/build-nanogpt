def check_dependencies():
    missing_deps = []
    
    # Check core dependencies
    try:
        import torch
        print(f"✓ PyTorch installed (version {torch.__version__})")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import tiktoken
        print("✓ Tiktoken installed")
    except ImportError:
        missing_deps.append("tiktoken")
    
    try:
        from transformers import GPT2LMHeadModel
        print("✓ Transformers installed")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import numpy
        print(f"✓ NumPy installed (version {numpy.__version__})")
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print("\n❌ Missing dependencies:")
        print("Run: pip install " + " ".join(missing_deps))
    else:
        print("\n✓ All core dependencies are installed!")
        
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available (version {torch.version.cuda})")
    else:
        print("ℹ️ CUDA is not available - will run on CPU")

if __name__ == "__main__":
    check_dependencies() 