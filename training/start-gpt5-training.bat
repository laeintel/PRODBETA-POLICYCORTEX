@echo off
echo =====================================
echo PolicyCortex GPT-5 Training Launcher
echo =====================================
echo.

REM Check Python
python --version
echo.

REM Check PyTorch and CUDA
echo Checking PyTorch and GPU...
python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo.

REM Run the quick start
echo Starting GPT-5 training setup...
python quick_start.py

echo.
echo =====================================
echo Setup complete! 
echo To start actual training, run: python train_model.py
echo =====================================
pause