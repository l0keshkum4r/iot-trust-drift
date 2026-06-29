"""
camera_monitor_agent.py — Monitor a DroidCam (or any webcam) as an IoT CCTV
==========================================================================
This runs on the laptop that has DroidCam installed (Windows).

It opens the webcam device (DroidCam exposes itself as a normal webcam),
measures health metrics, maps them into this project's feature-vector schema,
and POSTs to:
  POST /api/iot/telemetry

Metrics we derive (per interval):
  - fps, frame_size, bytes_per_sec (approx)
  - dropped_frames (capture read failures)

Install:
  pip install opencv-python requests

Run:
  python camera_monitor_agent.py --server http://<LAPTOP_IP>:8002 --device-id DroidCam_01 --camera-index 0
"""

import argparse
import time
from datetime import datetime

import requests
import subprocess
import re


def ping_stats(host: str, count: int = 4, timeout_ms: int = 900) -> dict | None:
    """
    Windows ping sampler.
    Returns: {"sent": int, "lost": int, "loss_pct": float, "rtt_min_ms": float|None, "rtt_avg_ms": float|None, "rtt_max_ms": float|None}
    """
    if not host:
        return None
    try:
        # -n count, -w timeout in ms (Windows)
        proc = subprocess.run(
            ["ping", "-n", str(count), "-w", str(timeout_ms), host],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")

        # Packets: Sent = 4, Received = 4, Lost = 0 (0% loss),
        m = re.search(r"Sent\s*=\s*(\d+),\s*Received\s*=\s*(\d+),\s*Lost\s*=\s*(\d+)\s*\((\d+)%\s*loss\)", out, re.IGNORECASE)
        sent = recv = lost = None
        loss_pct = None
        if m:
            sent = int(m.group(1))
            recv = int(m.group(2))
            lost = int(m.group(3))
            loss_pct = float(m.group(4))
        else:
            # Fallback: count "Reply from" lines
            recv = len(re.findall(r"Reply from", out, re.IGNORECASE))
            sent = count
            lost = max(0, sent - recv)
            loss_pct = round((lost / max(1, sent)) * 100.0, 1)

        # Minimum = 10ms, Maximum = 15ms, Average = 12ms
        m2 = re.search(r"Minimum\s*=\s*(\d+)ms,\s*Maximum\s*=\s*(\d+)ms,\s*Average\s*=\s*(\d+)ms", out, re.IGNORECASE)
        rtt_min = rtt_max = rtt_avg = None
        if m2:
            rtt_min = float(m2.group(1))
            rtt_max = float(m2.group(2))
            rtt_avg = float(m2.group(3))

        return {
            "sent": int(sent),
            "lost": int(lost),
            "loss_pct": float(loss_pct),
            "rtt_min_ms": rtt_min,
            "rtt_avg_ms": rtt_avg,
            "rtt_max_ms": rtt_max,
        }
    except Exception:
        return None


def http_stream_probe(url: str, sample_secs: float = 2.0, timeout: float = 6.0) -> dict:
    """
    Lightweight HTTP probe for MJPEG/HTTP camera streams.
    Measures:
      - status_code
      - ttfb_ms (time to first byte)
      - bytes_read and bytes_per_sec_est over a short sample
    """
    started = time.time()
    ttfb_ms = None
    bytes_read = 0
    status_code = None
    content_type = None
    error = None

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            status_code = r.status_code
            content_type = r.headers.get("content-type")

            # read a small sample window
            for chunk in r.iter_content(chunk_size=16 * 1024):
                now = time.time()
                if chunk:
                    if ttfb_ms is None:
                        ttfb_ms = round((now - started) * 1000.0, 1)
                    bytes_read += len(chunk)
                if (now - started) >= sample_secs:
                    break
    except Exception as e:
        error = str(e)

    elapsed = max(0.001, time.time() - started)
    bps = int(bytes_read / elapsed) if bytes_read > 0 else 0

    return {
        "url": url,
        "status_code": status_code,
        "content_type": content_type,
        "ttfb_ms": ttfb_ms,
        "sample_secs": round(elapsed, 2),
        "bytes_read": int(bytes_read),
        "bytes_per_sec_est": bps,
        "ok": bool(status_code and 200 <= int(status_code) < 400 and bytes_read > 0),
        "error": error,
    }


def main():
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(
            "OpenCV is not available. Install it in your active environment:\n"
            "  pip install opencv-python\n"
            "Then ensure your IDE is using the same Python interpreter (the project's .venv).\n"
            f"Original error: {e}"
        )

    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="http://<LAPTOP_IP>:8002")
    ap.add_argument("--device-id", default="DroidCam_01")
    ap.add_argument("--device-type", default="CCTV")
    ap.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (0/1/2...)")
    ap.add_argument(
        "--camera-url",
        default=None,
        help=(
            "Network camera URL (use this for DroidCam Wi-Fi). "
            "Examples: http://PHONE_IP:4747/video or http://PHONE_IP:4747/mjpegfeed"
        ),
    )
    ap.add_argument("--interval", type=int, default=8, help="Seconds per telemetry window")
    ap.add_argument("--jpeg-quality", type=int, default=70, help="0-100; used to estimate encoded bytes/sec")
    ap.add_argument("--ping-host", default=None, help="Phone IP/hostname to ping (Wi-Fi health). Example: 192.168.1.123")
    ap.add_argument("--ping-count", type=int, default=4)
    ap.add_argument("--ping-timeout-ms", type=int, default=900)
    ap.add_argument(
        "--probe-url",
        default=None,
        help="HTTP URL to probe for stream availability/performance. Default: http://<ping-host>:4747/ when --ping-host is set.",
    )
    ap.add_argument("--probe-sample-secs", type=float, default=2.0)
    ap.add_argument("--probe-timeout-secs", type=float, default=6.0)
    ap.add_argument(
        "--mode",
        choices=["normal", "malicious"],
        default="normal",
        help="normal=report real camera health, malicious=inject attack-like telemetry patterns",
    )
    ap.add_argument(
        "--attack",
        choices=["exfiltration", "port_scan", "mixed"],
        default="mixed",
        help="What pattern to simulate when --mode malicious",
    )
    ap.add_argument(
        "--attack-windows",
        type=int,
        default=6,
        help="How many telemetry windows to simulate attack for (each window is --interval seconds)",
    )
    args = ap.parse_args()

    server = args.server.rstrip("/")
    endpoint = f"{server}/api/iot/telemetry"

    if args.camera_url:
        cap = cv2.VideoCapture(args.camera_url)
        if not cap.isOpened():
            raise SystemExit(
                "Could not open camera URL.\n"
                f"URL: {args.camera_url}\n"
                "For DroidCam Wi-Fi, try one of these (depending on your DroidCam mode):\n"
                "  http://PHONE_IP:4747/video\n"
                "  http://PHONE_IP:4747/mjpegfeed\n"
                "Also ensure Windows firewall allows DroidCam traffic.\n"
            )
    else:
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise SystemExit(
                f"Could not open camera index {args.camera_index}. Try 0,1,2,...\n"
                "If you're using DroidCam over Wi-Fi (port 4747), prefer --camera-url instead of --camera-index.\n"
            )

    # Warm up
    time.sleep(0.5)

    print(f"[cam] device={args.device_id} type={args.device_type} cam_index={args.camera_index}")
    print(f"[cam] posting to {endpoint} every {args.interval}s")
    if args.ping_host:
        print(f"[cam] pinging {args.ping_host} each window (wifi health)")
    if args.mode == "malicious":
        print(f"[cam] MODE=MALICIOUS attack={args.attack} windows={args.attack_windows}")

    probe_url = args.probe_url
    if not probe_url and args.ping_host:
        probe_url = f"http://{args.ping_host}:4747/"
    if probe_url:
        print(f"[cam] probing stream URL each window: {probe_url}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(5, min(95, args.jpeg_quality)))]
    attack_windows_left = int(args.attack_windows) if args.mode == "malicious" else 0

    while True:
        start = time.time()
        frames = 0
        dropped = 0
        total_encoded_bytes = 0

        # sample frames for `interval` seconds
        while (time.time() - start) < args.interval:
            ok, frame = cap.read()
            if not ok or frame is None:
                dropped += 1
                time.sleep(0.02)
                continue

            frames += 1

            # Estimate "bytes out" by JPEG-encoding one frame (fast enough for a simple monitor)
            try:
                ok2, buf = cv2.imencode(".jpg", frame, encode_params)
                if ok2:
                    total_encoded_bytes += int(buf.size)
            except Exception:
                pass

            # small sleep so we don't max CPU; still captures enough frames for health
            time.sleep(0.01)

        elapsed = max(0.1, time.time() - start)
        fps = frames / elapsed
        bytes_per_sec = total_encoded_bytes / elapsed

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        net = ping_stats(args.ping_host, count=args.ping_count, timeout_ms=args.ping_timeout_ms) if args.ping_host else None
        probe = http_stream_probe(
            probe_url,
            sample_secs=float(args.probe_sample_secs),
            timeout=float(args.probe_timeout_secs),
        ) if probe_url else None

        # Map camera health -> project feature schema
        # - total_bytes_out: approximate encoded bytes/sec * interval
        # - total_packets_out: number of frames (treat frames as "packets")
        # - flows/dsts: fixed small values (single stream)
        # - external_ratio: 0 (local capture)
        total_bytes_out = float(bytes_per_sec * args.interval)
        total_packets_out = float(frames)
        num_flows = 1.0

        # ── Attack simulation (telemetry-only) ──
        # This does NOT hack anything; it just makes the device "look compromised" to the scorer.
        # Designed to trigger anomaly + drift + policy penalties.
        ext_ratio = 0.0
        uniq_ips = 1.0
        uniq_ports = 1.0
        uniq_protos = 1.0

        if attack_windows_left > 0:
            # Exfiltration: bandwidth spike + external destinations
            if args.attack in ("exfiltration", "mixed"):
                total_bytes_out *= 12.0
                ext_ratio = 0.85
                uniq_ips = 25.0
                uniq_ports = 6.0
                uniq_protos = 2.0

            # Port scan: many ports/ips/flows, low bytes per flow
            if args.attack in ("port_scan", "mixed"):
                num_flows = 120.0
                total_packets_out *= 6.0
                uniq_ips = max(uniq_ips, 60.0)
                uniq_ports = max(uniq_ports, 40.0)
                uniq_protos = max(uniq_protos, 2.0)

            attack_windows_left -= 1
        else:
            # normal mode (or attack finished)
            ext_ratio = 0.0

        payload = {
            "device_id": args.device_id,
            "device_type": args.device_type,
            "window_start": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "total_bytes_out": total_bytes_out,
            "total_packets_out": total_packets_out,
            "avg_bytes_per_flow": float(total_bytes_out / max(1.0, num_flows)),
            "num_flows": num_flows,
            "unique_dst_ips": uniq_ips,
            "unique_dst_ports": uniq_ports,
            "unique_protocols": uniq_protos,
            "external_ratio": ext_ratio,
            "avg_duration": float(args.interval),
            "mode": "malicious" if args.mode == "malicious" else "normal",
            # extra fields (ignored by backend model, but useful if you extend schema later)
            "camera": {
                "fps": round(fps, 2),
                "dropped_frames": int(dropped),
                "resolution": f"{width}x{height}",
                "bytes_per_sec_est": int(bytes_per_sec),
            },
            "wifi": net,
            "stream_probe": probe,
        }

        try:
            r = requests.post(endpoint, json=payload, timeout=8)
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"status": r.status_code}
            rtt = None
            loss = None
            if net:
                rtt = net.get("rtt_avg_ms")
                loss = net.get("loss_pct")
            print(
                f"[cam] fps={fps:>5.1f} dropped={dropped:<3} res={width}x{height} "
                f"trust={data.get('trust_score')} risk={data.get('risk_level')}"
                + (f" rtt={rtt}ms loss={loss}%" if net else "")
            )
        except Exception as e:
            print(f"[cam] post error: {e}")


if __name__ == "__main__":
    main()

