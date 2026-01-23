#!/usr/bin/env python3
import torch
import sys

# Check if terminal supports colors
def supports_color():
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

# ANSI color codes
RED = '\033[91m' if supports_color() else ''
GREEN = '\033[92m' if supports_color() else ''
RESET = '\033[0m' if supports_color() else ''

failed = False

# Test CUDA
try:
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        pytorch_version = torch.__version__
        # Quick CUDA test
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print(f"{GREEN}✓ CUDA OK{RESET} - PyTorch {pytorch_version}, CUDA {cuda_version}")
    else:
        print(f"{RED}❌ CUDA not available{RESET}")
        failed = True
except Exception as e:
    print(f"{RED}❌ CUDA test failed: {e}{RESET}")
    failed = True

# Test timm
try:
    import timm
    model = timm.create_model("resnet18", pretrained=False)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).cuda() if torch.cuda.is_available() else torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"{GREEN}✓ timm OK{RESET} - version {timm.__version__}")
except Exception as e:
    print(f"{RED}❌ timm test failed: {e}{RESET}")
    failed = True

# Test transformers
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    inputs = tokenizer("test", return_tensors="pt")
    inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"{GREEN}✓ transformers OK{RESET} - version {transformers.__version__}")
except Exception as e:
    print(f"{RED}❌ transformers test failed: {e}{RESET}")
    failed = True

sys.exit(1 if failed else 0)
