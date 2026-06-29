"""
iot_device_agent.py — Lightweight live telemetry sender for a real IoT device
===========================================================================
Runs on any Linux box (Raspberry Pi, router-like device, VM) with Python 3.

It samples:
  - Network I/O deltas (bytes/packets per interval)
  - Active inet connections (unique remote IPs/ports, protocol count)

Then converts them to the project's "feature vector" schema and POSTs to:
  POST /api/iot/telemetry

Install:
  pip install requests psutil

Run:
  python iot_device_agent.py --device-id Pi_01 --device-type Router --server http://<LAPTOP_IP>:8002
"""

import argparse
import time
from datetime import datetime

import psutil
import requests


def is_internal_ip(ip: str) -> bool:
    return (
        ip.startswith("10.")
        or ip.startswith("192.168.")
        or ip.startswith("172.16.") or ip.startswith("172.17.")
        or ip.startswith("172.18.") or ip.startswith("172.19.")
        or ip.startswith("172.20.") or ip.startswith("172.21.")
        or ip.startswith("172.22.") or ip.startswith("172.23.")
        or ip.startswith("172.24.") or ip.startswith("172.25.")
        or ip.startswith("172.26.") or ip.startswith("172.27.")
        or ip.startswith("172.28.") or ip.startswith("172.29.")
        or ip.startswith("172.30.") or ip.startswith("172.31.")
        or ip == "127.0.0.1"
    )


def sample_features(prev_io, interval_s: int):
    io = psutil.net_io_counters()
    d_sent = max(0, io.bytes_sent - prev_io.bytes_sent)
    d_pkts_sent = max(0, io.packets_sent - prev_io.packets_sent)

    # Approximate "flows" from active connections
    remote_ips = set()
    remote_ports = set()
    protos = set()
    ext = 0
    total = 0

    try:
        for conn in psutil.net_connections(kind="inet"):
            if not conn.raddr:
                continue
            total += 1
            rip = conn.raddr.ip
            rport = conn.raddr.port
            remote_ips.add(rip)
            remote_ports.add(rport)
            protos.add("TCP" if conn.type == 1 else "UDP")  # SOCK_STREAM=1, SOCK_DGRAM=2 (good enough)
            if not is_internal_ip(str(rip)):
                ext += 1
    except Exception:
        pass

    num_flows = max(1, total)
    external_ratio = ext / max(1, total)

    total_bytes_out = float(d_sent)
    total_packets_out = float(d_pkts_sent)
    avg_bytes_per_flow = float(total_bytes_out / max(1, num_flows))

    return io, {
        "total_bytes_out": total_bytes_out,
        "total_packets_out": total_packets_out,
        "avg_bytes_per_flow": avg_bytes_per_flow,
        "num_flows": float(num_flows),
        "unique_dst_ips": float(len(remote_ips)),
        "unique_dst_ports": float(len(remote_ports)),
        "unique_protocols": float(len(protos)),
        "external_ratio": float(external_ratio),
        "avg_duration": float(interval_s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", required=True)
    ap.add_argument("--device-type", default="Router", help="CCTV|Router|AccessController|custom")
    ap.add_argument("--server", required=True, help="http://<LAPTOP_IP>:8002")
    ap.add_argument("--interval", type=int, default=8)
    args = ap.parse_args()

    server = args.server.rstrip("/")
    endpoint = f"{server}/api/iot/telemetry"

    prev = psutil.net_io_counters()
    print(f"[agent] sending to {endpoint} every {args.interval}s as {args.device_id} ({args.device_type})")

    while True:
        time.sleep(args.interval)
        prev, feats = sample_features(prev, args.interval)

        payload = {
            "device_id": args.device_id,
            "device_type": args.device_type,
            "window_start": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            **feats,
            "mode": "normal",
        }

        try:
            r = requests.post(endpoint, json=payload, timeout=6)
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"status": r.status_code}
            print(f"[agent] trust={data.get('trust_score')} risk={data.get('risk_level')} drift={data.get('drift')} policy={data.get('policy')}")
        except Exception as e:
            print(f"[agent] error: {e}")


if __name__ == "__main__":
    main()

