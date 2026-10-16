# -*- coding: utf-8 -*-
"""
YOLOv8 训练脚本(适合训练自己的数据集)

用法示例:
    # 用默认参数训练(需要先改好 datasets/data.yaml)
    python train.py

    # 指定数据集配置和轮数
    python train.py --data datasets/data.yaml --epochs 100 --batch 8 --imgsz 640

    # 从已有的 best.pt 继续训练
    python train.py --model runs/detect/train/weights/last.pt --resume

显存参考(RTX 4070 Laptop 8GB, 640x640):
    yolov8n  batch=16   ~5.5GB
    yolov8s  batch=8    ~6GB
    yolov8m  batch=4    ~6.5GB   (偏紧, 建议 batch=4 并开 cache=False)
    显存不够就把 --batch 调小(会自动按 AMP 缩放), 或加 --imgsz 480
"""
import argparse
import os
import sys
from pathlib import Path

# 控制台默认编码可能是 GBK, 万一遇到 GBK 打不出的字符也不要让脚本崩掉
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# 脚本所在目录(项目根目录)。所有相对路径都相对它解析,
# 这样无论从哪个盘/哪个终端运行都不会写错地方。
BASE_DIR = Path(__file__).resolve().parent

# 关闭 ultralytics 的联网检查/遥测, 避免训练时卡顿
os.environ.setdefault("YOLO_VERBOSE", "false")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "false")


def _abs(p: Path) -> Path:
    """相对路径 -> 转成基于项目根目录的绝对路径"""
    p = Path(p)
    return p if p.is_absolute() else (BASE_DIR / p)


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def check_dataset(data_path: Path):
    """
    训练前先检查数据集, 有问题就给出人话提示, 而不是让 ultralytics 抛一长串报错。
    返回 (是否通过, 提示信息列表)
    """
    import yaml

    msgs = []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        return False, [f"读取数据集配置失败: {e}"]

    if "names" not in cfg:
        return False, ["data.yaml 里缺少 names(类别名), 请按模板补全"]

    # nc 可以省略, ultralytics 会从 names 推导
    names = cfg.get("names") or {}
    nc = cfg.get("nc", len(names))
    if not names:
        return False, ["data.yaml 里的 names 是空的, 至少要写一个类别名"]
    if nc != len(names):
        msgs.append(f"    [!] nc={nc} 和 names 数量({len(names)})不一致, 以 names 为准")

    root = Path(str(cfg.get("path", data_path.parent)))
    if not root.is_absolute():
        root = data_path.parent / root

    ok = True
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            if split == "train":
                ok = False
                msgs.append("[X] data.yaml 里没有配置 train")
            continue
        d = Path(str(rel))
        d = d if d.is_absolute() else root / d
        if not d.exists():
            ok = False
            msgs.append(f"[X] {split} 目录不存在: {d}")
            continue
        n_img = sum(1 for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)
        # 标签目录: 把路径里的 images 换成 labels
        lab_dir = Path(str(d).replace("images", "labels", 1))
        n_lab = sum(1 for p in lab_dir.rglob("*.txt")) if lab_dir.exists() else 0
        flag = "OK " if n_img else "[X]"
        if not n_img:
            ok = False
        msgs.append(f"{flag} {split:<5}: {n_img:>6} 张图片, {n_lab:>6} 个标签  ({d})")
        if n_img and n_lab == 0:
            msgs.append(f"    [!] 在 {lab_dir} 没找到标签文件, 训练会报错")
            ok = False

    name_list = list(names.values()) if isinstance(names, dict) else list(names)
    msgs.append(f"    类别数 nc={nc}, 类别名={name_list[:8]}{' ...' if len(name_list) > 8 else ''}")
    return ok, msgs


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    p.add_argument("--model", default="yolov8n.pt",
                   help="预训练权重或模型结构 (yolov8n/s/m/l/x.pt), 默认 yolov8n.pt")
    p.add_argument("--data", default="datasets/data.yaml",
                   help="数据集配置文件。默认 datasets/data.yaml (相对本脚本所在目录解析), "
                        "指向 D:/yolov8/datasets/mydata 的 4000 训练 / 400 验证 人脸数据集")
    p.add_argument("--epochs", type=int, default=150, help="训练轮数, 默认 150 (配合 patience 早停)")
    p.add_argument("--batch", type=int, default=8,
                   help="batch size; 640 下 8GB 显存建议 8(n) / 4(s)。填 -1 自动估计")
    # 注意: argparse 会把 help 文本当作 % 格式化字符串, 里面的字面 % 必须写成 %%
    p.add_argument("--imgsz", type=int, default=640,
                   help="输入图片尺寸, 默认 640 (960 实测无收益且慢 70%%, 推理时再放大到 960)")
    p.add_argument("--device", default="0", help="GPU 编号 0; 用 CPU 则填 cpu")
    p.add_argument("--workers", type=int, default=4, help="数据加载线程数(Windows 建议 4)")
    p.add_argument("--project", default="runs/detect",
                   help="结果保存根目录(相对路径会自动转成绝对路径)")
    p.add_argument("--name", default="train", help="本次实验名, 结果存到 project/name")
    p.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    p.add_argument("--cache", action="store_true", help="把图片缓存到内存(数据集小可加速)")
    p.add_argument("--patience", type=int, default=30, help="早停耐心值, 默认 30 (两轮最佳点均在 ep54~79)")
    p.add_argument("--seed", type=int, default=0, help="随机种子")

    # ---- 优化器与学习率 (2026-09-26 新增) ----
    p.add_argument("--optimizer", default="auto",
                   help="优化器 auto/SGD/AdamW/Adam/NAdam/RAdam/RMSprop。"
                        "注意: auto 会忽略下面的 lr0/momentum, 由 ultralytics 自己推导")
    p.add_argument("--lr0", type=float, default=0.002,
                   help="初始(峰值)学习率; 用 AdamW 建议 0.001~0.002; optimizer=auto 时被忽略")
    p.add_argument("--lrf", type=float, default=0.01, help="末期学习率 = lr0*lrf (cos_lr 余弦衰减)")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD 动量 / Adam 的 beta1")
    p.add_argument("--weight-decay", type=float, default=0.001, help="权重衰减(防过拟合), 默认 0.001")
    p.add_argument("--warmup-epochs", type=float, default=3.0, help="前 N 轮学习率从 0 线性升到 lr0")
    p.add_argument("--warmup-bias-lr", type=float, default=0.0,
                   help="预热期 bias 组学习率; 用 Adam 系建议 0.0")

    # ---- 损失权重 ----
    p.add_argument("--box", type=float, default=7.5, help="框回归损失权重")
    p.add_argument("--dfl", type=float, default=1.5, help="DFL 损失权重(直接影响定位精度)")
    p.add_argument("--cls", type=float, default=0.5, help="分类损失权重")

    # ---- 数据增强 (人脸检测调优重点) ----
    p.add_argument("--mosaic", type=float, default=0.8,
                   help="4 图拼接增强概率。第二轮降到 0.5 后 val 损失反而上升(过拟合), 故回复到 0.8")
    p.add_argument("--scale", type=float, default=0.4,
                   help="随机缩放幅度, 0.4 => 0.6~1.4 倍。第二轮 0.2 导致正则不足, 故回复到 0.4")
    p.add_argument("--erasing", type=float, default=0.2,
                   help="随机擦除概率。第二轮 0.1 导致正则不足, 故回复到 0.2")
    p.add_argument("--degrees", type=float, default=0.0, help="随机旋转角度(±度)")
    p.add_argument("--close-mosaic", type=int, default=10, help="最后 N 轮关闭 mosaic")

    # ---- 其他 ----
    p.add_argument("--fraction", type=float, default=1.0,
                   help="只用前百分之几的训练数据(0~1), 用于快速试参数")
    p.add_argument("--max-det", type=int, default=300, help="验证/推理时单图最多保留多少个框")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[X] 依赖缺失: {e}\n    请先激活环境: D:\\yolov8\\激活yolo8环境.cmd")
        return 1

    print("=" * 62)
    print("  YOLOv8 训练")
    print("=" * 62)

    # 关键: 全部转成绝对路径, 避免 ultralytics 对相对路径 project 的解析坑
    model_path = str(_abs(args.model))
    if not os.path.exists(model_path):
        model_path = args.model  # 本地没有就让 ultralytics 自己下载官方权重
    data_path = str(_abs(args.data))
    project_path = str(_abs(args.project))

    print(f"  torch      : {torch.__version__}")
    print(f"  CUDA 可用  : {torch.cuda.is_available()}")
    if torch.cuda.is_available() and args.device != "cpu":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  模型       : {model_path}")
    print(f"  数据集配置 : {data_path}")
    print(f"  结果目录   : {project_path}\\{args.name}")
    print(f"  轮数/尺寸  : {args.epochs} 轮 / imgsz={args.imgsz} / batch={args.batch}")
    if str(args.optimizer).lower() == "auto":
        print("  优化器     : auto  (lr0/momentum 由 ultralytics 自动推导, 命令行写的不生效)")
    else:
        print(f"  优化器     : {args.optimizer}  lr0={args.lr0}  lrf={args.lrf}  momentum={args.momentum}")
    print(f"  数据增强   : mosaic={args.mosaic}  scale={args.scale}  erasing={args.erasing}")
    if not os.path.exists(data_path) and not args.resume:
        print(f"\n[X] 找不到数据集配置文件: {data_path}")
        print("    请按 README 里的说明准备好数据集, 并修改 datasets/data.yaml")
        return 1

    # 数据集预检查, 提前发现问题
    if not args.resume:
        print("-" * 62)
        print("  数据集检查")
        ok, msgs = check_dataset(Path(data_path))
        for m in msgs:
            print("  " + m)
        if not ok:
            print("\n[X] 数据集还没准备好, 已停止。")
            print("    先把图片和标签放到 datasets/mydata/ 下, 并改好 datasets/data.yaml")
            print("    目录结构见 README.md 或 datasets/data.yaml 底部的注释")
            return 1
    print("=" * 62)

    model = YOLO(model_path)

    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=project_path,
        name=args.name,
        resume=args.resume,
        cache=args.cache,
        patience=args.patience,
        seed=args.seed,
        # ---- 优化器与学习率 ----
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        # ---- 损失权重 ----
        box=args.box,
        dfl=args.dfl,
        cls=args.cls,
        # ---- 数据增强 ----
        mosaic=args.mosaic,
        scale=args.scale,
        erasing=args.erasing,
        degrees=args.degrees,
        close_mosaic=args.close_mosaic,
        # ---- 其他 ----
        fraction=args.fraction,
        max_det=args.max_det,
        amp=True,          # 混合精度, 省显存提速
        cos_lr=True,       # 余弦学习率
        plots=True,        # 训练结束画曲线图
        val=True,
    )

    save_dir = getattr(results, "save_dir", os.path.join(project_path, args.name))
    print("\n" + "=" * 62)
    print("  训练完成!")
    print(f"  结果目录   : {save_dir}")
    print(f"  最佳权重   : {os.path.join(save_dir, 'weights', 'best.pt')}")
    print(f"  指标曲线   : {os.path.join(save_dir, 'results.png')}")
    print(f"  混淆矩阵   : {os.path.join(save_dir, 'confusion_matrix.png')}")
    print("\n  下一步 —— 用训练好的模型推理:")
    print(f"    python predict.py --model \"{os.path.join(save_dir, 'weights', 'best.pt')}\" --source images")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
