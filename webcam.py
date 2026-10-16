# -*- coding: utf-8 -*-
"""
YOLOv8 人脸检测 —— 摄像头实时检测

本脚本只负责「摄像头实时检测」; 跑图片/文件夹/视频请用 predict.py。

用法示例:
    # 默认摄像头 + 默认模型 (第三轮 face_4000 的 best.pt)
    python webcam.py

    # 指定模型和尺寸
    python webcam.py --model runs/detect/face_4000/weights/best.pt --imgsz 640

    # 换第二个摄像头 (外接 USB 相机), 并指定分辨率
    python webcam.py --source 2 --width 1280 --height 720

    # 只测帧率不弹窗 (用于确认速度和摄像头是否正常)
    python webcam.py --no-show --frames 60

运行中的按键:
    q 或 ESC   退出
    s          保存当前这一帧 (存到 --save-dir)
    m          在 640 / 960 之间切换推理尺寸 (实时看速度与精度的差别)

注意:
    - 弹窗需要本机桌面会话; 无桌面时请加 --no-show
    - 摄像头被其它程序 (微信/钉钉/相机app) 占用时会打不开, 先关掉它们
"""
import argparse
import sys
import time
from pathlib import Path

# 控制台默认编码可能是 GBK, 避免打不出的字符让脚本崩掉
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent


def _abs(p):
    """相对路径 -> 基于项目根目录的绝对路径"""
    p = Path(p)
    return p if p.is_absolute() else (BASE_DIR / p)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 人脸检测 - 摄像头实时检测")
    p.add_argument("--model", default="runs/detect/face_4000/weights/best.pt",
                   help="权重文件, 默认用第三轮(4000张)训练出的 best.pt")
    p.add_argument("--source", type=int, default=0,
                   help="摄像头编号(只支持摄像头)。0=内置, 2=外接USB(本机实测1280x720)。"
                        "跑图片/文件夹/视频请用 predict.py")
    p.add_argument("--imgsz", type=int, default=640,
                   help="推理尺寸。实时建议 640(快) / 960(准但慢一倍)。用 m 键可运行时切换")
    p.add_argument("--conf", type=float, default=0.40, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.60, help="NMS IoU 阈值(人脸重叠多可调高)")
    p.add_argument("--max-det", type=int, default=1000, help="单帧最多检测多少张脸")
    p.add_argument("--device", default="0", help="GPU 编号 0, 或 cpu")
    p.add_argument("--width", type=int, default=0, help="摄像头采集宽度, 0=用默认")
    p.add_argument("--height", type=int, default=0, help="摄像头采集高度, 0=用默认")
    p.add_argument("--no-show", action="store_true", help="不弹窗(无桌面环境/只测速时用)")
    p.add_argument("--frames", type=int, default=0, help="只处理 N 帧后退出, 0=一直跑")
    p.add_argument("--warmup", type=int, default=10, help="前 N 帧不计入帧率统计")
    p.add_argument("--save-dir", default="runs/webcam_shots", help="按 s 键保存的截图目录")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[X] 依赖缺失: {e}\n    请先激活环境: D:\\yolov8\\激活yolo8环境.cmd")
        return 1

    model_path = str(_abs(args.model))
    if not Path(model_path).exists():
        print(f"[X] 找不到权重: {model_path}")
        print("    训练完的话路径形如 runs/detect/<实验名>/weights/best.pt")
        return 1

    print("=" * 62)
    print("  YOLOv8 人脸检测 - 摄像头实时检测")
    print("=" * 62)
    print(f"  模型     : {model_path}")
    print(f"  摄像头   : index {args.source}  (backend=DSHOW)")
    print(f"  推理尺寸 : {args.imgsz}    conf={args.conf}  iou={args.iou}")
    print(f"  显示窗口 : {'关(只测速)' if args.no_show else '开'}")
    if not args.no_show:
        print("  按键     : q/ESC=退出   s=存图   m=切换640/960   +/-=调置信度")
    print("-" * 62)

    # ---- 打开摄像头 (Windows 上 DSHOW 比默认后端快很多) ----
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.source, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[X] 打不开摄像头 index={args.source}")
        print("    可能原因: 被其它程序占用 / 隐私设置禁止 / 换个 --source 编号试试")
        return 1

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[X] 摄像头能打开但读不到画面, 换 --source 或检查设备")
        cap.release()
        return 1
    h, w = frame.shape[:2]
    print(f"  实际采集 : {w}x{h}")
    print("=" * 62)

    model = YOLO(model_path)
    save_dir = _abs(args.save_dir)
    imgsz = args.imgsz
    conf_thr = args.conf          # 置信度阈值, 运行中可用 +/- 键实时调整

    times = []
    n = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[!] 读帧失败, 结束")
            break

        t0 = time.time()
        res = model.predict(frame, imgsz=imgsz, conf=conf_thr, iou=args.iou,
                            max_det=args.max_det, device=args.device,
                            verbose=False)[0]
        dt = time.time() - t0

        n += 1
        if n > args.warmup:
            times.append(dt)

        nface = len(res.boxes)
        fps = (1.0 / (sum(times) / len(times))) if times else 0.0

        if not args.no_show:
            vis = res.plot()  # 已画好框的图 (BGR)
        else:
            vis = frame

        # 左上角叠加实时信息
        txt1 = f"faces: {nface}"
        txt2 = f"FPS: {fps:.1f}  ({dt * 1000:.0f} ms/frame)  imgsz={imgsz}"
        txt3 = f"conf: {conf_thr:.2f}   (+/- 调整, 越高越严)"
        cv2.rectangle(vis, (0, 0), (500, 84), (0, 0, 0), -1)
        cv2.putText(vis, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, txt2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, txt3, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if not args.no_show:
            cv2.imshow("YOLOv8 Face Detection  (q=quit  s=save  m=size  +/-=conf)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # q 或 ESC
                break
            elif key == ord("s"):
                save_dir.mkdir(parents=True, exist_ok=True)
                f = save_dir / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(f), vis)
                print(f"  已保存: {f}")
            elif key == ord("m"):
                imgsz = 960 if imgsz == 640 else 640
                times.clear()
                print(f"  切换推理尺寸 -> imgsz={imgsz}")
            elif key in (ord("+"), ord("=")):   # 调高置信度阈值
                conf_thr = min(0.95, round(conf_thr + 0.05, 2))
                print(f"  conf -> {conf_thr:.2f}")
            elif key in (ord("-"), ord("_")):   # 调低置信度阈值
                conf_thr = max(0.05, round(conf_thr - 0.05, 2))
                print(f"  conf -> {conf_thr:.2f}")

        if args.frames and n >= args.frames:
            break

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    wall = time.time() - t_start
    if times:
        avg = sum(times) / len(times)
        print("-" * 62)
        print(f"  共处理 {n} 帧, 总耗时 {wall:.1f}s")
        print(f"  平均单帧 {avg * 1000:.1f} ms  →  约 {1 / avg:.1f} FPS  (imgsz={imgsz})")
        print(f"  最快 {min(times) * 1000:.1f} ms / 最慢 {max(times) * 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
