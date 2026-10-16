# -*- coding: utf-8 -*-
"""
YOLOv8 推理/预测脚本

本脚本负责「批量 / 离线」推理: 图片、文件夹、视频文件。
摄像头实时检测请用 webcam.py (带帧率统计和按键调参)。

用法示例:
    # 用人脸模型跑整个文件夹 (默认权重已是第三轮 4000 张的 best.pt)
    python predict.py --source datasets/mydata/images/val

    # 跑视频文件
    python predict.py --source D:/videos/test.mp4

    # 调低置信度阈值, 同时导出 YOLO 格式的 txt 标签
    python predict.py --source datasets/mydata/images/val --conf 0.25 --save-txt

    # 换用别的权重
    python predict.py --model runs/detect/face/weights/best.pt --source datasets/mydata/images/val
"""
import argparse
import sys
from pathlib import Path

# 控制台默认编码可能是 GBK, 万一遇到 GBK 打不出的字符也不要让脚本崩掉
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# 脚本所在目录(项目根目录)
BASE_DIR = Path(__file__).resolve().parent


def _abs(p):
    """相对路径 -> 基于项目根目录的绝对路径"""
    p = Path(p)
    return p if p.is_absolute() else (BASE_DIR / p)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 推理脚本")
    p.add_argument("--model", default="runs/detect/face_4000/weights/best.pt",
                   help="权重文件, 默认用第三轮(4000张)训练出的人脸模型 best.pt")
    p.add_argument("--source", default="datasets/mydata/images/val",
                   help="图片/文件夹/视频路径。摄像头实时检测请用 webcam.py")
    p.add_argument("--imgsz", type=int, default=960, help="推理尺寸")
    p.add_argument("--conf", type=float, default=0.42, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.60, help="NMS IoU 阈值")
    p.add_argument("--max-det", type=int, default=1000, help="单图最大检测框数")
    p.add_argument("--device", default="0", help="GPU 编号 0, 或 cpu")
    p.add_argument("--save-txt", action="store_true", help="同时保存预测结果的 txt 标签")
    p.add_argument("--show", action="store_true", help="弹窗显示结果(本地有桌面时可用)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[X] 依赖缺失: {e}\n    请先激活环境: D:\\yolov8\\激活yolo8环境.cmd")
        return 1

    model_path = str(_abs(args.model))
    if not Path(model_path).exists():
        if len(Path(args.model).parts) > 1:   # 形如 runs/detect/xxx/weights/best.pt
            print(f"[X] 找不到权重文件: {model_path}")
            print("    训练产物的路径形如 runs/detect/<实验名>/weights/best.pt")
            return 1
        model_path = args.model  # 官方权重名(如 yolov8n.pt), 交给 ultralytics 自动下载
    src = int(args.source) if args.source.isdigit() else str(_abs(args.source))

    model = YOLO(model_path)
    results = model.predict(
        source=src,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        show=args.show,
        project=str(BASE_DIR / "runs" / "detect"),
        name="predict",
        exist_ok=True,
    )

    total = 0
    for r in results:
        boxes = getattr(r, "boxes", None)
        n = 0 if boxes is None else len(boxes)
        total += n
        names = {int(c): r.names[int(c)] for c in (boxes.cls.tolist() if n else [])}
        if n:
            print(f"  {getattr(r, 'path', '?')}: 检测到 {n} 个目标 {list(names.values())}")
    print(f"\n共检测到 {total} 个目标, 结果已保存到 runs/detect/predict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
