# -*- coding: utf-8 -*-
"""
YOLOv8 环境自检脚本
用法:
    python check_env.py          # 基础自检(不联网)
    python check_env.py --full   # 完整自检(会下载 yolov8n.pt 并做一次推理)
"""
import argparse
import platform
import sys

# 控制台默认编码可能是 GBK, 万一遇到 GBK 打不出的字符也不要让脚本崩掉
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


def line(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="额外下载 yolov8n.pt 并跑一次真实推理")
    args = ap.parse_args()

    ok = True

    line("1. Python 解释器")
    print(f"  版本      : {sys.version.split()[0]}")
    print(f"  路径      : {sys.executable}")
    print(f"  系统      : {platform.system()} {platform.release()} ({platform.machine()})")
    if sys.version_info[:2] != (3, 12):
        print("  [!] 建议使用 Python 3.12(当前 conda 环境 yolo8 已固定为 3.12)")

    line("2. PyTorch / CUDA")
    try:
        import torch
        print(f"  torch 版本        : {torch.__version__}")
        print(f"  编译用 CUDA 版本  : {torch.version.cuda}")
        print(f"  cuDNN 版本        : {torch.backends.cudnn.version()}")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA 是否可用     : {cuda_ok}")
        if cuda_ok:
            print(f"  GPU 数量          : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  GPU[{i}]           : {p.name}")
                print(f"    显存            : {p.total_memory / 1024**3:.1f} GB")
                print(f"    计算能力(sm)    : {p.major}.{p.minor}")
            # 真正做一次 GPU 矩阵运算,确认不是"假的可用"
            a = torch.randn(1000, 1000, device="cuda")
            b = torch.randn(1000, 1000, device="cuda")
            c = (a @ b).sum().item()
            torch.cuda.synchronize()
            print(f"  GPU 矩阵运算测试  : 通过 (结果 {c:.2f})")
        else:
            ok = False
            print("  [X] CUDA 不可用! 将退化为 CPU 训练(速度慢 20~50 倍)")
    except ImportError as e:
        ok = False
        print(f"  [X] 导入 torch 失败: {e}")
        return finish(ok)

    line("3. Ultralytics (YOLOv8)")
    try:
        import ultralytics
        from ultralytics import YOLO  # noqa: F401
        print(f"  ultralytics 版本  : {ultralytics.__version__}")
        import ultralytics.utils as _u
        print(f"  YOLO 类导入       : 成功")
    except Exception as e:
        ok = False
        print(f"  [X] 导入 ultralytics 失败: {e}")
        return finish(ok)

    line("4. 关键依赖")
    for mod, name in [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("yaml", "pyyaml"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
    ]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "")
            print(f"  {name:<18}: {v}")
        except Exception as e:
            print(f"  {name:<18}: [!] 未安装 ({type(e).__name__})")

    if args.full:
        line("5. 真实推理测试 (yolov8n.pt)")
        try:
            import numpy as np
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            img = (np.random.rand(480, 640, 3) * 255).astype("uint8")
            r = model.predict(img, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
            print(f"  推理成功, 输出张量形状: {r[0].boxes.data.shape}")
            print("  [OK] 环境完全可用, 可以开始训练了!")
        except Exception as e:
            ok = False
            print(f"  [X] 推理失败: {type(e).__name__}: {e}")
            print("      如果只是下载 yolov8n.pt 失败, 属于网络问题, 可手动下载后放到 D:\\yolov8\\weights\\")

    return finish(ok)


def finish(ok):
    line("结论")
    if ok:
        print("  [OK] 环境检查全部通过, 可以开始训练了!")
    else:
        print("  [!!] 存在问题, 请看上面标 [X] 的项")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
