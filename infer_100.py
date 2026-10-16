# -*- coding: utf-8 -*-
"""
第三轮模型 (runs/detect/face_4000) 的独立测试：100 张新人脸照片 + 漏检/错检统计。

为什么用 WIDER FACE 的 val 划分？
  - 你的 4400 张全部来自 WIDER FACE 的 train 划分；val 划分(3226 张)一张都没用过
  - 它是真正独立的外部测试集，而且自带官方标注 → 「漏检/错检」可以客观数出来
  - 若用网上下载的无标注照片，只能数出「检出多少」，无法判断漏了多少、错在哪里

输出目录（按你的要求）：
  D:\\yolov8\\推理用图片\\原图\\        100 张原始照片
  D:\\yolov8\\推理用图片\\推理结果\\    100 张画好框的结果图
  D:\\yolov8\\推理用图片\\标注\\        100 个真值标签(便于人工复核)
  D:\\yolov8\\推理用图片\\结果汇总.csv  每张图的 TP / 漏检 / 错检
"""
import csv
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"D:\yolov8")
TMP = ROOT / "_val_tmp"
OUT = ROOT / "推理用图片"
IMG_OUT = OUT / "原图"
RES_OUT = OUT / "推理结果"
GT_OUT = OUT / "标注"
WEIGHTS = ROOT / "runs" / "detect" / "face_4000" / "weights" / "best.pt"
N_IMG = 100
IMGSZ, CONF_SHOW, IOU_NMS = 960, 0.42, 0.60      # 显示用（与 predict.py 默认一致）
CONF_SWEEP = (0.10, 0.25, 0.42, 0.50)            # 统计用的阈值
IOU_MATCH = 0.5                                   # 判定「检对」的 IoU 门槛
VAL_URL = "https://hf-mirror.com/datasets/wider_face/resolve/main/data/WIDER_val.zip"
# 训练集分布（来自 add_wider.py 的实测统计），用于让 100 张测试图的构成与训练一致
TRAIN_DIST = {1: 1146, 2: 242, 3: 62, 4: 52, 5: 23, 6: 13, 7: 17, 8: 7, 9: 2,
              10: 7, 11: 6, 12: 5, 14: 1, 15: 1, 16: 1, 18: 5, 20: 1, 22: 1,
              25: 1, 28: 1, 39: 1, 40: 1, 47: 1, 53: 1}


def log(msg):
    print(msg, flush=True)


def ensure_val_images():
    """下载并解压 WIDER FACE val 划分"""
    gt_dir = TMP / "ann" / "wider_face_split"
    img_dir = TMP / "WIDER_val" / "images"
    if img_dir.exists() and (gt_dir / "wider_face_val_bbx_gt.txt").exists():
        log("[跳过] val 数据已存在")
        return img_dir, gt_dir / "wider_face_val_bbx_gt.txt"
    TMP.mkdir(parents=True, exist_ok=True)
    zip_path = TMP / "WIDER_val.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
        log(f"[下载] {VAL_URL}")
        r = subprocess.run(["curl.exe", "-L", "--retry", "5", "--retry-delay", "3",
                            "-C", "-", "--connect-timeout", "30", "-o", str(zip_path), VAL_URL])
        if r.returncode != 0:
            log("[X] 下载失败。可手动下载 WIDER_val.zip 放到 " + str(TMP))
            sys.exit(1)
    log(f"[解压] {zip_path.name}  ({zip_path.stat().st_size/1e6:.0f} MB)")
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(TMP)
    if not img_dir.exists():
        log(f"[X] 解压后未找到 {img_dir}")
        sys.exit(1)
    if not (gt_dir / "wider_face_val_bbx_gt.txt").exists():
        log("[下载] 标注文件")
        ann_zip = TMP / "wider_face_split.zip"
        subprocess.run(["curl.exe", "-L", "-o", str(ann_zip),
                        "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"])
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(TMP / "ann")
    return img_dir, gt_dir / "wider_face_val_bbx_gt.txt"


def load_gt(p):
    """解析官方标注。注意: count=0 的条目后面有一行全 0 占位, 必须跳过"""
    lines = [x.strip() for x in p.open(encoding="utf-8", errors="replace")]
    out, i = {}, 0
    while i < len(lines):
        if not lines[i].endswith(".jpg"):
            i += 1
            continue
        name = lines[i]; i += 1
        n = int(lines[i]); i += 1
        boxes = [[float(v) for v in lines[i + j].split()] for j in range(n)]
        i += n
        if n == 0 and i < len(lines) and not lines[i].endswith(".jpg"):
            i += 1
        out[name] = boxes
    return out


def valid(raw):
    return [b[:4] for b in raw if not (len(b) >= 8 and int(b[7]) == 1) and b[2] > 0 and b[3] > 0]


def pick_100(gt):
    """按训练集的「每图框数」分布分层抽 100 张"""
    need = N_IMG
    tot = sum(TRAIN_DIST.values())
    ex = {k: need * v / tot for k, v in TRAIN_DIST.items()}
    quota = {k: int(v) for k, v in ex.items()}
    for k, _ in sorted(ex.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[: need - sum(quota.values())]:
        quota[k] += 1
    pool = {}
    for k, raw in gt.items():
        n = len(valid(raw))
        pool.setdefault(n, []).append(k)
    for v in pool.values():
        v.sort()
    picked = []
    for n in sorted(quota, key=lambda x: -quota[x]):
        for _ in range(quota[n]):
            if pool.get(n):
                picked.append((n, pool[n].pop(0)))
    log(f"[选图] 目标 {need} 张，实得 {len(picked)} 张；框数分布 "
        f"{dict(sorted(Counter(n for n, _ in picked).items()))}")
    return picked


def boxes_xyxy(labels, W, H):
    out = []
    for cls, cx, cy, w, h in labels:
        out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(gts, preds, thr):
    """贪心匹配: 返回 (命中数, 漏检数, 错检数)"""
    used = set()
    hit = 0
    for g in gts:
        best, bi = thr, -1
        for j, p in enumerate(preds):
            if j in used:
                continue
            v = iou(g, p)
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            used.add(bi)
            hit += 1
    return hit, len(gts) - hit, len(preds) - len(used)


def main():
    if not WEIGHTS.exists():
        log(f"[X] 找不到模型: {WEIGHTS}")
        sys.exit(1)

    img_dir, gt_file = ensure_val_images()
    gt = load_gt(gt_file)
    picked = pick_100(gt)

    for d in (IMG_OUT, RES_OUT, GT_OUT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    log("[复制] 原图 + 写真值标签")
    names, gts = [], {}
    for n, key in picked:
        src = img_dir / key
        dst = IMG_OUT / key.replace("/", "__")
        shutil.copy2(src, dst)
        names.append(dst.name)
        W, H = Image.open(src).size
        px, lines = [], []
        for x, y, w, h in valid(gt[key]):
            px.append([x, y, x + w, y + h])                 # 像素 xyxy -> 匹配用
            lines.append("0 %.6f %.6f %.6f %.6f" %          # 归一化 -> 写标签文件
                         ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        gts[dst.name] = px
        (GT_OUT / (dst.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    from ultralytics import YOLO
    m = YOLO(str(WEIGHTS))

    log("[推理1/2] conf=0.01 收集原始检测(用于多阈值统计)")
    raw = {}
    for r in m.predict(source=str(IMG_OUT), imgsz=IMGSZ, conf=0.01, iou=IOU_NMS,
                       device="0", verbose=False, save=False, stream=True):
        raw[Path(r.path).name] = (r.boxes.xyxy.cpu().numpy().tolist(),
                                  r.boxes.conf.cpu().numpy().tolist())

    log(f"[推理2/2] conf={CONF_SHOW} 生成结果图 -> {RES_OUT}")
    m.predict(source=str(IMG_OUT), imgsz=IMGSZ, conf=CONF_SHOW, iou=IOU_NMS,
              device="0", verbose=False, save=True,
              project=str(OUT), name="推理结果", exist_ok=True)

    log("[统计] 按 IoU>=%.2f 贪心匹配" % IOU_MATCH)
    total_gt = sum(len(v) for v in gts.values())

    # 自检: 真值框必须落在图像范围内, 否则说明坐标系搞错了
    bad = 0
    for name in names:
        W, H = Image.open(IMG_OUT / name).size
        for x1, y1, x2, y2 in gts[name]:
            if not (0 <= x1 < x2 <= W + 1 and 0 <= y1 < y2 <= H + 1):
                bad += 1
    log(f"[自检] 真值框越界数量 = {bad}  (应为 0)")
    if bad:
        log("[X] 真值坐标异常, 终止以免给出错误统计")
        sys.exit(1)

    rows = []
    for name in names:
        for c in CONF_SWEEP:
            px, pc = raw.get(name, ([], []))
            preds = [b for b, cf in zip(px, pc) if cf >= c]
            hit, fn, fp = match(gts[name], preds, IOU_MATCH)
            rows.append([name, c, len(gts[name]), len(preds), hit, fn, fp])

    with (OUT / "结果汇总.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["图片", "conf阈值", "真实人脸数", "检出框数", "正确检出", "漏检", "错检"])
        w.writerows(rows)

    log("")
    log("=" * 74)
    log(f"  真实人脸总数（100 张图的真值标注合计）: {total_gt}")
    log(f"  匹配标准: IoU >= {IOU_MATCH} 且置信度 >= 阈值     推理尺寸: {IMGSZ}")
    log("=" * 74)
    log(f"  {'conf':>6} {'检出框':>8} {'正确检出':>9} {'漏检':>7} {'错检':>7} {'召回率':>8} {'精确率':>8}")
    for c in CONF_SWEEP:
        rr = [r for r in rows if r[1] == c]
        tp = sum(r[4] for r in rr); fn = sum(r[5] for r in rr); fp = sum(r[6] for r in rr)
        rec = tp / (tp + fn) * 100 if tp + fn else 0
        pre = tp / (tp + fp) * 100 if tp + fp else 0
        log(f"  {c:>6.2f} {tp+fp:>8} {tp:>9} {fn:>7} {fp:>7} {rec:>7.1f}% {pre:>7.1f}%")
    log("=" * 74)
    log(f"  原图   : {IMG_OUT}")
    log(f"  结果图 : {RES_OUT}")
    log(f"  标注   : {GT_OUT}")
    log(f"  明细   : {OUT / '结果汇总.csv'}")


if __name__ == "__main__":
    main()
