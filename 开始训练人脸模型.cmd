@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 训练人脸检测模型 - YOLOv8

REM ============================================================
REM  可以改下面这 3 个数字(用记事本打开本文件改就行)
REM ============================================================
set EPOCHS=150
set BATCH=8
set IMGSZ=640
REM ============================================================

echo ============================================================
echo    YOLOv8 人脸检测模型训练
echo ============================================================
call "D:\miniconda3\Scripts\activate.bat" yolo8

echo.
echo   [1/3] 环境检查
python -c "import torch; print('        torch', torch.__version__); print('        GPU  ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '不可用, 将用 CPU(极慢)')"

echo.
echo   [2/3] 数据集
echo        位置: D:\yolov8\datasets\mydata
echo        配置: datasets\data.yaml  (4000 张训练 / 400 验证)

echo.
echo   [3/3] 开始训练
echo        轮数=%EPOCHS%  batch=%BATCH%  尺寸=%IMGSZ%
echo        大约需要 20-40 分钟, 训练中请不要关闭本窗口
echo ------------------------------------------------------------
echo.

python train.py --data "D:\yolov8\datasets\data.yaml" --epochs %EPOCHS% --batch %BATCH% --imgsz %IMGSZ% --name face_4000
echo.
echo ============================================================
echo   训练结束
echo.
echo   最佳模型: D:\yolov8\runs\detect\face\weights\best.pt
echo   指标曲线: D:\yolov8\runs\detect\face\results.png
echo   混淆矩阵: D:\yolov8\runs\detect\face\confusion_matrix.png
echo ============================================================
pause
