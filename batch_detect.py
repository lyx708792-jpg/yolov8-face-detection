# -*- coding: utf-8 -*-
"""
批量人脸检测 —— 对文件夹内所有图片跑推理，结果存到你指定的目录。

与 predict.py 的区别
    predict.py   : 通用推理，结果固定写到 runs/detect/predict (每次覆盖)
    本脚本        : 专为批量场景，多了 4 件事
                    1. --out       输出目录完全自定义
                    2. --recursive 递归扫描子文件夹
                    3. --csv       导出每张图的检出统计表
                    4. --crop      把人脸小图裁出来 (做人脸识别/建库用)

用法示例
    # 最简: 结果默认存到 "<源文件夹>_detected"
    python batch_detect.py --source "D:\\photos"

    # 指定输出目录
    python batch_detect.py --source "D:\\photos" --out "D:\\结果"

    # 递归子文件夹 + 导出统计表
    python batch_detect.py --source "D:\\photos" --recursive --csv

    # 顺便把人脸裁出来 (外扩 20%)
    python batch_detect.py --source "D:\\photos" --crop --crop-pad 0.2

    # 先拿 10 张试一下
    python batch_detect.py --source "D:\\photos" --limit 10

输出结构
    <out>\\
    ├── 图片名.jpg          画好框的结果图 (递归时子目录用 __ 拼进文件名)
    ├── 结果汇总.csv        --csv 时生成
    ├── labels\\图片名.txt    --save-txt 时生成 (YOLO 格式)
    └── crops\\图片名_1.jpg   --crop 时生成 (裁出的人脸)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# 控制台默认编码可能是 GBK, 遇到打不出的字符不要让脚本崩掉
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _abs(p):
    """相对路径 -> 基于项目根目录的绝对路径"""
    p = Path(p)
    return p if p.is_absolute() else (BASE_DIR / p)


def parse_args():
    p = argparse.ArgumentParser(
        description="批量人脸检测 (文件夹内图片)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True,
                   help="【必填】要处理的文件夹路径 (也可以直接给单个图片文件)")
    p.add_argument("--out", default=None,
                   help="输出目录。默认 = <源文件夹>_detected (单个文件时 = 它所在目录\\detected)")
    p.add_argument("--model", default="runs/detect/face_4000/weights/best.pt",
                   help="权重文件, 默认第三轮(4000张)的人脸模型")
    p.add_argument("--imgsz", type=int, default=960, help="推理尺寸, 默认 960")
    p.add_argument("--conf", type=float, default=0.42, help="置信度阈值, 默认 0.42")
    p.add_argument("--iou", type=float, default=0.60, help="NMS IoU 阈值, 默认 0.60")
    p.add_argument("--max-det", type=int, default=1000, help="单图最多保留多少个框")
    p.add_argument("--device", default="0", help="GPU 编号 0, 或 cpu")
    p.add_argument("--recursive", action="store_true", help="递归处理子文件夹")
    p.add_argument("--csv", action="store_true", help="导出 结果汇总.csv")
    p.add_argument("--save-txt", action="store_true", help="给每张图导出 YOLO 格式标签")
    p.add_argument("--crop", action="store_true", help="把人脸裁剪成小图另存")
    p.add_argument("--crop-pad", type=float, default=0.15,
                   help="裁剪时四周外扩比例, 默认 0.15 (15%%)")
    p.add_argument("--line-width", type=int, default=0, help="框线宽, 0=自动")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张, 0=全部 (调试用)")
    return p.parse_args()


def find_images(src: Path, recursive: bool):
    """收集待处理图片。返回 [(绝对路径, 输出用的名字)]"""
    if src.is_file():
        return [(src, src.name)]
    it = src.rglob("*") if recursive else src.glob("*")
    out = []
    for f in sorted(it):
        if not f.is_file() or f.suffix.lower() not in IMG_EXT:
            continue
        name = f.name if not recursive else str(f.relative_to(src)).replace("\\", "__").replace("/", "__")
        out.append((f, name))
    return out


def save_image(path: Path, img):
    """写图。用 imencode+tofile 而不是 cv2.imwrite, 避免中文路径写不出去"""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    buf.tofile(str(path))


def main():
    args = parse_args()
    src = _abs(args.source)

    if not src.exists():
        print(f"[X] 找不到路径: {src}")
        return 1

    images = find_images(src, args.recursive)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"[X] 在 {src} 里没有找到图片 (支持: {', '.join(sorted(IMG_EXT))})")
        if not args.recursive:
            print("    如果有子文件夹, 加 --recursive 试试")
        return 1

    out_dir = _abs(args.out) if args.out else (
        src.parent / f"{src.stem}_detected" if src.is_file() else src.with_name(src.name + "_detected"))

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[X] 依赖缺失: {e}\n    请先激活环境: D:\\yolov8\\激活yolo8环境.cmd")
        return 1

    model_path = str(_abs(args.model))
    if not Path(model_path).exists():
        print(f"[X] 找不到权重: {model_path}")
        return 1

    print("=" * 66)
    print("  批量人脸检测")
    print("=" * 66)
    print(f"  源        : {src}{'  (递归)' if args.recursive else ''}")
    print(f"  待处理    : {len(images)} 张")
    print(f"  模型      : {model_path}")
    print(f"  参数      : imgsz={args.imgsz}  conf={args.conf}  iou={args.iou}  max_det={args.max_det}")
    print(f"  输出      : {out_dir}")
    print(f"  附件产物  : {'CSV ' if args.csv else ''}{'TXT ' if args.save_txt else ''}{'crops' if args.crop else ''}"
          f"{'（无）' if not (args.csv or args.save_txt or args.crop) else ''}")
    print("-" * 66, flush=True)

    model = YOLO(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_faces = 0
    t_start = time.time()

    for i, (f, name) in enumerate(images, 1):
        t0 = time.time()
        try:
            res = model.predict(str(f), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                max_det=args.max_det, device=args.device, verbose=False)[0]
        except Exception as e:
            print(f"  [{i}/{len(images)}] [X] {name}  推理失败: {type(e).__name__}: {e}")
            rows.append([name, -1, "", "", "", "", f"{type(e).__name__}"])
            continue

        n = len(res.boxes)
        total_faces += n
        confs = res.boxes.conf.tolist() if n else []
        h, w = res.orig_shape[:2]
        dt = (time.time() - t0) * 1000

        # 1) 结果图 (画好框)
        save_image(out_dir / name, res.plot(line_width=args.line_width or None))

        # 2) YOLO 格式标签
        if args.save_txt:
            (out_dir / "labels").mkdir(parents=True, exist_ok=True)
            lines = []
            for b in res.boxes:
                cx, cy, bw, bh = b.xywhn[0].tolist()
                lines.append(f"{int(b.cls[0])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out_dir / "labels" / (Path(name).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # 3) 裁人脸 (从原图裁, 不是带框图)
        if args.crop and n:
            (out_dir / "crops").mkdir(parents=True, exist_ok=True)
            H, W = res.orig_img.shape[:2]
            for j, b in enumerate(res.boxes, 1):
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                pw, ph = (x2 - x1) * args.crop_pad, (y2 - y1) * args.crop_pad
                x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
                x2 = min(W, int(x2 + pw)); y2 = min(H, int(y2 + ph))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                save_image(out_dir / "crops" / f"{Path(name).stem}_{j}.jpg",
                           res.orig_img[y1:y2, x1:x2])

        rows.append([name, n,
                     f"{max(confs):.3f}" if confs else "",
                     f"{min(confs):.3f}" if confs else "",
                     w, h, f"{dt:.0f}"])
        if i % 20 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] 已处理, 累计检出 {total_faces} 张脸", flush=True)

    wall = time.time() - t_start

    if args.csv:
        csv_path = out_dir / "结果汇总.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp)
            w.writerow(["图片", "检出人脸数", "最高置信度", "最低置信度", "宽", "高", "耗时ms"])
            w.writerows(rows)

    n_ok = sum(1 for r in rows if r[1] >= 0)
    print("=" * 66)
    print("  完成!")
    print(f"  处理图片  : {n_ok} 张" + (f" (失败 {len(rows) - n_ok} 张)" if n_ok != len(rows) else ""))
    print(f"  检出人脸  : {total_faces} 张")
    print(f"  有脸的图  : {sum(1 for r in rows if r[1] > 0)} 张   /   无脸的图: {sum(1 for r in rows if r[1] == 0)} 张")
    print(f"  平均每图  : {total_faces / n_ok:.2f} 张脸" if n_ok else "")
    print(f"  总耗时    : {wall:.1f} 秒   ({wall / max(n_ok, 1) * 1000:.0f} ms/张)")
    print(f"  结果目录  : {out_dir}")
    if args.csv:
        print(f"  统计表    : {out_dir / '结果汇总.csv'}")
    if args.crop:
        print(f"  人脸小图  : {out_dir / 'crops'}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
