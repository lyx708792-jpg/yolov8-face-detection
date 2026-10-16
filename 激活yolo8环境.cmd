@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YOLOv8 yolo8
echo ============================================================
echo   正在激活 conda 环境: yolo8
echo ============================================================
call "D:\miniconda3\Scripts\activate.bat" yolo8
echo.
echo   Python 版本与路径:
python -c "import sys; print('     ', sys.version.split()[0], '|', sys.executable)"
echo   GPU 状态:
python -c "import torch; print('      torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
echo.
echo   当前目录: %CD%
echo ------------------------------------------------------------
echo   常用命令:
echo     python check_env.py            环境自检
echo     python train.py --help         查看训练参数
echo     python webcam.py --source 0   摄像头实时检测
echo     python predict.py --source 你的文件夹
echo ============================================================
cmd /k
