#!/bin/bash
# Quick start script for Qwen3.5 Tool-Calling SFT

set -e  # Exit on error

echo "======================================"
echo "Qwen3.5 Tool-Calling SFT Setup"
echo "======================================"

# 1. Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install -r requirements.txt

# 3. Verify installation
echo "✅ Verifying installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. Run training (uncomment to auto-start)
# echo "🚀 Starting training..."
# python train_qwen35_toolcalling.py

echo "======================================"
echo "✅ Setup complete!"
echo "======================================"
echo ""
echo "To start training, run:"
echo "  source venv/bin/activate"
echo "  python train_qwen35_toolcalling.py"
echo ""
echo "Or use the LoRA version for GPU-constrained setups:"
echo "  python train_qwen35_toolcalling_lora.py"
