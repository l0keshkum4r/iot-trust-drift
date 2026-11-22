"""
api/network_monitor.py
======================
Monitors real IoT devices connected to the local network.
Performs concurrent ping sweeps and port checks, resolves MAC addresses
from the ARP table, and evaluates trust scores by integrating with the
existing LiveEngine, Policy Engine, and Drift Detector.

Enhancement: Per-device rolling baseline tracking so drift is measured
relative to each device's own learned history, not static defaults.
"""

import os
import json
import socket
import threading
import subprocess
import re
import time
import math
from datetime import datetime
import collections
import concurrent.futures
from typing import List, Dict, Any, Optional

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEVICES_FILE = os.path.join(DATA_DIR, "monitored_devices.json")

COMMON_PORTS = [22, 23, 53, 80, 443, 554, 1883, 8080, 8443]

PORT_PROTOCOL_MAP = {
    80: "HTTP", 443: "HTTPS", 53: "DNS", 22: "SSH",
    21: "FTP", 25: "SMTP", 110: "POP3", 143: "IMAP",
    3389: "RDP", 554: "RTSP", 1883: "MQTT", 8883: "MQTTS",
    8080: "HTTP-ALT", 8443: "HTTPS-ALT",
}

# Feature names tracked for drift
FEATURE_NAMES = [
    "total_bytes_out", "total_packets_out", "unique_dst_ips",
    "external_ratio", "num_flows", "avg_bytes_per_flow",
    "unique_dst_ports", "unique_protocols", "avg_duration",
]

def get_protocol_name(port: int) -> str:
    return PORT_PROTOCOL_MAP.get(port, "TCP")

# Helper to get local IP
def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class RollingBaseline:
    """
    Maintains a per-device rolling mean and variance of feature vectors
    using Welford's online algorithm (numerically stable).
    After `min_samples` observations the baseline is considered 'learned'
    and z-score drift can be computed.
    """
    def __init__(self, feature_names: List[str], min_samples: int = 5):
        self.feature_names = feature_names
        self.min_samples = min_samples
        self.n: int = 0
        self.mean: Dict[str, float] = {f: 0.0 for f in feature_names}
        self.M2: Dict[str, float] = {f: 0.0 for f in feature_names}  # sum of squared deviations

    def update(self, features: Dict[str, float]):
        """Add a new observation using Welford's algorithm."""
        self.n += 1
        for f in self.feature_names:
            x = features.get(f, 0.0)
            delta = x - self.mean[f]
            self.mean[f] += delta / self.n
            delta2 = x - self.mean[f]
            self.M2[f] += delta * delta2

    @property
    def variance(self) -> Dict[str, float]:
        if self.n < 2:
            return {f: 1.0 for f in self.feature_names}
        return {f: self.M2[f] / (self.n - 1) for f in self.feature_names}

    @property
    def std(self) -> Dict[str, float]:
        return {f: math.sqrt(max(v, 1e-8)) for f, v in self.variance.items()}

    def is_ready(self) -> bool:
        return self.n >= self.min_samples

    def compute_drift(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute z-scores for each feature and return drift class / magnitude.
        Returns a dict compatible with the LiveEngine drift output.
        """
        if not self.is_ready():
            return {
                "drift_class": "DRIFT_NONE",
                "drift_magnitude": 0.0,
                "top_drifters": [],
                "penalty": 0,
                "baseline_samples": self.n,
            }

        stds = self.std
        z_scores: Dict[str, float] = {}
        for f in self.feature_names:
            x = features.get(f, 0.0)
            z = abs(x - self.mean[f]) / stds[f]
            z_scores[f] = round(z, 3)

        # Max z-score across features
        max_z = max(z_scores.values()) if z_scores else 0.0

        # Sort top drifting features
        top_drifters = sorted(z_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_drifters_list = [
            {"feature": f, "z_score": round(z, 2)}
            for f, z in top_drifters if z > 0.5
        ]

        # Classify drift
        if max_z > 3.0:
            drift_class = "DRIFT_STRONG"
            penalty = 20
        elif max_z > 1.5:
            drift_class = "DRIFT_MILD"
            penalty = 10
        else:
            drift_class = "DRIFT_NONE"
            penalty = 0

        return {
            "drift_class": drift_class,
            "drift_magnitude": round(max_z, 2),
            "top_drifters": top_drifters_list,
            "penalty": penalty,
            "baseline_samples": self.n,
        }

    def to_dict(self) -> Dict:
        return {"n": self.n, "mean": self.mean, "M2": self.M2}

    @classmethod
    def from_dict(cls, data: Dict, feature_names: List[str]) -> "RollingBaseline":
        b = cls(feature_names)
        b.n = data.get("n", 0)
        b.mean = data.get("mean", {f: 0.0 for f in feature_names})
        b.M2 = data.get("M2", {f: 0.0 for f in feature_names})
        return b


class NetworkIoTMonitor:
    """
    Manages discovery and continuous background monitoring of local network IoT devices.
    """
    def __init__(self, live_engine=None):
        self.live_engine = live_engine
        self.monitored_devices: Dict[str, Dict[str, Any]] = {}  # ip -> device metadata
        self.latest_status: Dict[str, Dict[str, Any]] = {}    # ip -> latest scored state
        self.history: Dict[str, collections.deque] = {}       # ip -> history queue
        self.baselines: Dict[str, RollingBaseline] = {}       # ip -> rolling baseline
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
        self.scan_in_progress = False

        # Ensure DATA_DIR exists
        os.makedirs(DATA_DIR, exist_ok=True)
        self.load_devices()
        self._load_baselines()

    # ── Persistence ──────────────────────────────────────────────────

    def load_devices(self):
        with self.lock:
            if os.path.exists(DEVICES_FILE):
                try:
                    with open(DEVICES_FILE, "r") as f:
                        self.monitored_devices = json.load(f)
                    print(f"[NetMonitor] Loaded {len(self.monitored_devices)} devices from config.")
                except Exception as e:
                    print(f"[NetMonitor] Error loading monitored devices: {e}")
                    self.monitored_devices = {}
            else:
                self.monitored_devices = {}

    def save_devices(self):
        try:
            with open(DEVICES_FILE, "w") as f:
                json.dump(self.monitored_devices, f, indent=2)
        except Exception as e:
            print(f"[NetMonitor] Error saving monitored devices: {e}")

    def _baselines_path(self) -> str:
        return os.path.join(DATA_DIR, "monitored_baselines.json")

    def _load_baselines(self):
        """Load persisted rolling baselines from disk."""
        path = self._baselines_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for ip, bdata in data.items():
                    self.baselines[ip] = RollingBaseline.from_dict(bdata, FEATURE_NAMES)
                print(f"[NetMonitor] Loaded baselines for {len(self.baselines)} devices.")
            except Exception as e:
                print(f"[NetMonitor] Error loading baselines: {e}")

    def _save_baselines(self):
        """Persist rolling baselines to disk."""
        try:
            data = {ip: b.to_dict() for ip, b in self.baselines.items()}
            with open(self._baselines_path(), "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[NetMonitor] Error saving baselines: {e}")

    # ── Device Registration ───────────────────────────────────────────

    def add_device(self, ip: str, custom_name: str, device_type: str, mac: str = "") -> bool:
        """Register a device for active monitoring."""
        if not ip:
            return False
        with self.lock:
            self.monitored_devices[ip] = {
                "ip": ip,
                "custom_name": custom_name,
                "device_type": device_type,
                "mac": mac or self.get_mac_for_ip(ip),
                "added_at": datetime.now().isoformat(),
            }
            self.save_devices()
            # Initialize history and baseline
            if ip not in self.history:
                self.history[ip] = collections.deque(maxlen=120)
            if ip not in self.baselines:
                self.baselines[ip] = RollingBaseline(FEATURE_NAMES, min_samples=5)
        print(f"[NetMonitor] Registered device: {custom_name} ({ip}) [{device_type}]")
        return True

    def delete_device(self, ip: str) -> bool:
        """Unregister a device from active monitoring."""
        with self.lock:
            if ip in self.monitored_devices:
                del self.monitored_devices[ip]
                self.save_devices()
                if ip in self.latest_status:
                    del self.latest_status[ip]
                if ip in self.history:
                    del self.history[ip]
                if ip in self.baselines:
                    del self.baselines[ip]
                    self._save_baselines()
                print(f"[NetMonitor] Unregistered device: {ip}")
                return True
        return False

    # ── Network Probing ───────────────────────────────────────────────

    def get_mac_for_ip(self, ip: str) -> str:
        """Get the MAC address corresponding to an IP from /proc/net/arp."""
        try:
            if os.path.exists("/proc/net/arp"):
                with open("/proc/net/arp", "r") as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 4 and parts[0] == ip:
                            mac = parts[3]
                            if mac != "00:00:00:00:00:00":
                                return mac
        except Exception as e:
            print(f"[NetMonitor] Error reading ARP cache: {e}")
        return ""

    @staticmethod
    def ping_ip(ip: str, count: int = 1, timeout: float = 1.0) -> Dict[str, Any]:
        """Pings an IP address and returns latency (RTT) and packet loss."""
        try:
            proc = subprocess.run(
                ["ping", "-c", str(count), "-W", str(timeout), ip],
                capture_output=True,
                text=True,
                timeout=timeout * count + 1
            )
            stdout = proc.stdout or ""

            # Parse packet loss
            loss_match = re.search(r"(\d+)%\s+packet\s+loss", stdout)
            loss = float(loss_match.group(1)) if loss_match else 100.0

            # Parse RTT min/avg/max/mdev
            rtt_match = re.search(r"rtt\s+min/avg/max/mdev\s*=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)", stdout)
            avg_rtt = float(rtt_match.group(2)) if rtt_match else None

            return {
                "alive": proc.returncode == 0 or loss < 100.0,
                "rtt": avg_rtt,
                "loss": loss,
            }
        except Exception:
            return {"alive": False, "rtt": None, "loss": 100.0}

    @staticmethod
    def check_port(ip: str, port: int, timeout: float = 0.15) -> bool:
        """Check if a port is open by attempting connection."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex((ip, port)) == 0
        except Exception:
            return False

    def scan_ports(self, ip: str, ports: List[int] = COMMON_PORTS) -> List[int]:
        """Scan a list of ports concurrently for a single IP."""
        open_ports = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as executor:
            futures = {executor.submit(self.check_port, ip, port): port for port in ports}
            for future in concurrent.futures.as_completed(futures):
                port = futures[future]
                if future.result():
                    open_ports.append(port)
        return sorted(open_ports)

    def discover_subnet(self, subnet_prefix: str = "") -> List[Dict[str, Any]]:
        """
        Scan the subnet /24 concurrently using ping sweeps and port checks.
        Returns a list of discovered active hosts.
        """
        if self.scan_in_progress:
            return []

        self.scan_in_progress = True
        print(f"[NetMonitor] Initiating subnet scan on prefix: {subnet_prefix}...")

        if not subnet_prefix:
            local_ip = _get_local_ip()
            if local_ip == "127.0.0.1":
                self.scan_in_progress = False
                return []
            parts = local_ip.split(".")
            subnet_prefix = f"{parts[0]}.{parts[1]}.{parts[2]}."

        discovered_devices = []
        ips_to_scan = [f"{subnet_prefix}{i}" for i in range(1, 255)]

        # Find host IP to skip scanning ourselves
        local_ip = _get_local_ip()

        # Step 1: Concurrent ping sweep
        alive_ips = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
            futures = {executor.submit(self.ping_ip, ip, count=1, timeout=0.8): ip for ip in ips_to_scan if ip != local_ip}
            for future in concurrent.futures.as_completed(futures):
                ip = futures[future]
                res = future.result()
                if res["alive"]:
                    alive_ips[ip] = res

        # Step 2: For alive hosts, fetch details concurrently
        def fetch_device_details(ip, ping_res):
            open_ports = self.scan_ports(ip)

            # Resolve hostname
            hostname = ""
            try:
                hostname = socket.gethostbyaddr(ip)[0]
            except Exception:
                pass

            mac = self.get_mac_for_ip(ip)

            # Estimate device type
            h_lower = hostname.lower()
            if 554 in open_ports or "camera" in h_lower or "cctv" in h_lower:
                estimated_type = "CCTV"
            elif 53 in open_ports or "router" in h_lower or "gateway" in h_lower or ip.endswith(".1"):
                estimated_type = "Router"
            elif 443 in open_ports and (22 in open_ports or 80 in open_ports or "access" in h_lower):
                estimated_type = "AccessController"
            else:
                estimated_type = "Generic"

            return {
                "ip": ip,
                "hostname": hostname,
                "mac": mac,
                "alive": True,
                "rtt_ms": round(ping_res["rtt"], 1) if ping_res["rtt"] is not None else 0.0,
                "loss": ping_res["loss"],
                "open_ports": open_ports,
                "estimated_type": estimated_type,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            details_futures = [executor.submit(fetch_device_details, ip, ping_res) for ip, ping_res in alive_ips.items()]
            for future in concurrent.futures.as_completed(details_futures):
                try:
                    dev = future.result()
                    discovered_devices.append(dev)
                except Exception as e:
                    print(f"[NetMonitor] Error fetching device details: {e}")

        # Sort by IP address
        def ip_sort_key(dev):
            try:
                return [int(x) for x in dev["ip"].split(".")]
            except Exception:
                return [0, 0, 0, 0]

        discovered_devices.sort(key=ip_sort_key)
        self.scan_in_progress = False
        print(f"[NetMonitor] Subnet scan complete. Found {len(discovered_devices)} active devices.")
        return discovered_devices

    # ── Feature Synthesis & Drift Computation ────────────────────────

    def build_virtual_features_and_flows(
        self, device: Dict[str, Any], alive: bool, rtt: Optional[float], loss: float, open_ports: List[int]
    ) -> tuple:
        """
        Synthesize virtual feature vectors and raw flows matching the device profile
        based on active network measurements (latency, packet loss, open ports).
        """
        device_type = device["device_type"]
        device_ip = device["ip"]

        # Standard profile normal baselines
        if device_type == "CCTV":
            features = {
                "total_bytes_out": 9_145_000.0,
                "total_packets_out": 9126.0,
                "unique_dst_ips": 1.0,
                "external_ratio": 0.0,
                "num_flows": 1.0,
                "avg_bytes_per_flow": 9_145_000.0,
                "unique_dst_ports": 1.0,
                "unique_protocols": 1.0,
                "avg_duration": 2.0,
            }
        elif device_type == "Router":
            features = {
                "total_bytes_out": 478_000.0,
                "total_packets_out": 867.0,
                "unique_dst_ips": 2.0,
                "external_ratio": 1.0,
                "num_flows": 30.0,
                "avg_bytes_per_flow": 15_933.0,
                "unique_dst_ports": 1.0,
                "unique_protocols": 2.0,
                "avg_duration": 1.5,
            }
        else:  # AccessController & Generic
            features = {
                "total_bytes_out": 24_000.0,
                "total_packets_out": 55.0,
                "unique_dst_ips": 1.0,
                "external_ratio": 0.0,
                "num_flows": 12.0,
                "avg_bytes_per_flow": 2_000.0,
                "unique_dst_ports": 1.0,
                "unique_protocols": 1.0,
                "avg_duration": 2.0,
            }

        if not alive:
            for k in features:
                features[k] = 0.0
            return features, []

        # Modify features based on actual ping performance
        if rtt is not None:
            if rtt > 100.0:
                features["avg_duration"] *= 2.5
            elif rtt > 40.0:
                features["avg_duration"] *= 1.5

        # If packet loss is present, reduce throughput features
        if loss > 0.0:
            multiplier = max(0.1, 1.0 - (loss / 100.0))
            features["total_bytes_out"] *= multiplier
            features["total_packets_out"] *= multiplier
            features["avg_bytes_per_flow"] *= multiplier

        # Map open ports to flows
        raw_flows = []
        for port in open_ports:
            proto = get_protocol_name(port)
            raw_flows.append({
                "protocol": proto,
                "dst_port": int(port),
                "dst_ip": device_ip,
                "direction": "outbound",
            })

        # Update feature ports/protocols based on scan
        if open_ports:
            features["unique_dst_ports"] = float(len(open_ports))
            unique_protos = set(get_protocol_name(p) for p in open_ports)
            features["unique_protocols"] = float(len(unique_protos))
            features["num_flows"] = float(max(features["num_flows"], len(open_ports)))
            features["avg_bytes_per_flow"] = features["total_bytes_out"] / max(features["num_flows"], 1)

        return features, raw_flows

    def _compute_device_drift(self, ip: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Update the per-device rolling baseline and compute drift metrics.
        Returns drift result dict with drift_class, drift_magnitude, top_drifters, penalty.
        """
        with self.lock:
            if ip not in self.baselines:
                self.baselines[ip] = RollingBaseline(FEATURE_NAMES, min_samples=5)
            baseline = self.baselines[ip]

        # Compute drift BEFORE updating baseline (so current observation is measured against history)
        drift_result = baseline.compute_drift(features)

        # Update baseline with current observation
        baseline.update(features)

        # Periodically persist baselines (every 10th update)
        if baseline.n % 10 == 0:
            self._save_baselines()

        return drift_result

    # ── Background Monitor Loop ───────────────────────────────────────

    def _monitor_loop(self):
        """Active background monitoring loop — runs every ~8 seconds."""
        print("[NetMonitor] Background pinger started.")
        while self.running:
            with self.lock:
                devices = list(self.monitored_devices.values())

            for dev in devices:
                if not self.running:
                    break
                ip = dev["ip"]
                device_type = dev["device_type"]

                # Active checks
                ping_res = self.ping_ip(ip, count=3, timeout=0.6)
                open_ports = []
                # If ping succeeds, scan ports normally
                if ping_res["alive"]:
                    open_ports = self.scan_ports(ip)
                else:
                    # Attempt to infer liveness from LiveEngine recent window
                    if self.live_engine:
                        recent = self.live_engine.get_latest_scored_window()
                        if recent and recent.get("device_id") == ip:
                            # Treat as alive based on recent live data
                            ping_res["alive"] = True
                            ping_res["rtt"] = None
                            ping_res["loss"] = 0.0
                # Build features using whatever alive status we have
                features, raw_flows = self.build_virtual_features_and_flows(
                    dev, ping_res["alive"], ping_res["rtt"], ping_res["loss"], open_ports
                )

                # Compute per-device rolling baseline drift
                if ping_res["alive"]:
                    drift_result = self._compute_device_drift(ip, features)
                else:
                    drift_result = {
                        "drift_class": "DRIFT_STRONG",
                        "drift_magnitude": 10.0,
                        "top_drifters": [{"feature": "total_bytes_out", "z_score": 10.0}],
                        "penalty": 20,
                        "baseline_samples": self.baselines.get(ip, RollingBaseline(FEATURE_NAMES)).n,
                    }

                # Score with LiveEngine (uses its own anomaly + policy models)
                if self.live_engine and ping_res["alive"]:
                    scored = self.live_engine._score_single_window(
                        device_id=ip,
                        device_type=device_type,
                        features=features,
                        raw_flows=raw_flows,
                        window_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                elif not ping_res["alive"]:
                    tick = self.live_engine.tick_count if self.live_engine else 0
                    scored = {
                        "device_id": ip,
                        "device_type": device_type,
                        "window": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tick": tick,
                        "trust_score_raw": 0.0,
                        "trust_score_smoothed": 0.0,
                        "severity": "Critical",
                        "anomaly_score": 1.0,
                        "is_anomaly": 1,
                        "anomaly_deduction": 40.0,
                        "drift_class": "DRIFT_STRONG",
                        "drift_magnitude": 10.0,
                        "drift_deduction": 20,
                        "policy_status": "HARD_VIOLATION",
                        "policy_violations": "Device Offline / Network Unreachable",
                        "policy_deduction": 40,
                        "top_drifters": json.dumps([]),
                        "baseline_updated": False,
                        "total_bytes_out": 0.0,
                        "total_packets_out": 0.0,
                        "unique_dst_ips": 0.0,
                        "external_ratio": 0.0,
                        "num_flows": 0.0,
                    }
                else:
                    scored = {
                        "device_id": ip,
                        "device_type": device_type,
                        "window": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tick": 0,
                        "trust_score_smoothed": 100.0,
                        "severity": "Low",
                        "anomaly_score": 0.0,
                        "is_anomaly": 0,
                        "anomaly_deduction": 0.0,
                        "drift_class": drift_result["drift_class"],
                        "drift_magnitude": drift_result["drift_magnitude"],
                        "drift_deduction": drift_result["penalty"],
                        "policy_status": "COMPLIANT",
                        "policy_violations": "none",
                        "policy_deduction": 0,
                        "top_drifters": json.dumps(drift_result["top_drifters"]),
                    }

                # Override drift fields from our rolling baseline (more accurate for real devices)
                if ping_res["alive"] and self.live_engine:
                    scored["drift_class"] = drift_result["drift_class"]
                    scored["drift_magnitude"] = drift_result["drift_magnitude"]
                    scored["drift_deduction"] = drift_result["penalty"]
                    scored["top_drifters"] = json.dumps(drift_result["top_drifters"])
                    scored["baseline_samples"] = drift_result.get("baseline_samples", 0)

                # Combine device metadata, scored analytics, ping data
                record = {
                    **dev,
                    **scored,
                    "alive": ping_res["alive"],
                    "ping_rtt_ms": round(ping_res["rtt"], 1) if ping_res["rtt"] is not None else 0.0,
                    "packet_loss_pct": ping_res["loss"],
                    "open_ports": open_ports,
                    "checked_at": datetime.now().isoformat(),
                    # Ensure top_drifters is always a parsed list in history records
                    "top_drifters_parsed": drift_result.get("top_drifters", []),
                }

                with self.lock:
                    self.latest_status[ip] = record
                    if ip not in self.history:
                        self.history[ip] = collections.deque(maxlen=120)
                    self.history[ip].append(record)

                # Small stagger between pings/scans
                time.sleep(0.5)

            # Sleep 8 seconds before scanning registered devices again
            for _ in range(80):
                if not self.running:
                    break
                time.sleep(0.1)

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
        print("[NetMonitor] Monitoring system started.")

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)
        self._save_baselines()
        print("[NetMonitor] Monitoring system stopped.")

    # ── Data Access ───────────────────────────────────────────────────

    def get_monitored_devices(self) -> List[Dict[str, Any]]:
        with self.lock:
            results = []
            for ip, dev in self.monitored_devices.items():
                status = self.latest_status.get(ip, {
                    "alive": False,
                    "ping_rtt_ms": 0.0,
                    "packet_loss_pct": 100.0,
                    "open_ports": [],
                    "trust_score_smoothed": 100.0,
                    "severity": "Low",
                    "drift_class": "DRIFT_NONE",
                    "drift_magnitude": 0.0,
                    "drift_deduction": 0,
                    "anomaly_score": 0.0,
                    "anomaly_deduction": 0.0,
                    "policy_status": "COMPLIANT",
                    "policy_violations": "Pending check",
                    "policy_deduction": 0,
                    "top_drifters_parsed": [],
                    "baseline_samples": 0,
                    "checked_at": None,
                })
                results.append({**dev, **status})
            return results

    def get_device_history(self, ip: str, last_n: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            if ip in self.history:
                return list(self.history[ip])[-last_n:]
            return []

    def get_fleet_summary(self) -> Dict[str, Any]:
        """
        Returns aggregated fleet health metrics across all monitored devices.
        Used by the /api/network/fleet-summary endpoint.
        """
        with self.lock:
            statuses = list(self.latest_status.values())
            all_devices = list(self.monitored_devices.values())

        total = len(all_devices)
        online = sum(1 for s in statuses if s.get("alive", False))
        offline = total - online

        trust_scores = [s.get("trust_score_smoothed", 100.0) for s in statuses if s.get("alive", False)]
        avg_trust = round(sum(trust_scores) / len(trust_scores), 1) if trust_scores else 0.0

        in_drift = sum(
            1 for s in statuses
            if s.get("drift_class", "DRIFT_NONE") != "DRIFT_NONE" and s.get("alive", False)
        )
        critical_count = sum(
            1 for s in statuses
            if s.get("severity", "Low") in ("High", "Critical")
        )
        policy_violations = sum(
            1 for s in statuses
            if s.get("policy_status", "COMPLIANT") != "COMPLIANT" and s.get("alive", False)
        )

        return {
            "total_devices": total,
            "online": online,
            "offline": offline,
            "avg_trust": avg_trust,
            "in_drift": in_drift,
            "critical_count": critical_count,
            "policy_violations": policy_violations,
        }

    def get_drift_alerts(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """
        Returns recent drift/violation events sorted by time descending,
        across all monitored devices.
        """
        with self.lock:
            all_history: List[Dict] = []
            for ip, dq in self.history.items():
                for record in dq:
                    drift_class = record.get("drift_class", "DRIFT_NONE")
                    policy_status = record.get("policy_status", "COMPLIANT")
                    if drift_class != "DRIFT_NONE" or policy_status != "COMPLIANT":
                        all_history.append({
                            "ip": ip,
                            "custom_name": record.get("custom_name", ip),
                            "device_type": record.get("device_type", "Generic"),
                            "checked_at": record.get("checked_at", ""),
                            "severity": record.get("severity", "Low"),
                            "trust_score_smoothed": record.get("trust_score_smoothed", 100.0),
                            "drift_class": drift_class,
                            "drift_magnitude": record.get("drift_magnitude", 0.0),
                            "policy_status": policy_status,
                            "policy_violations": record.get("policy_violations", "none"),
                            "top_drifters_parsed": record.get("top_drifters_parsed", []),
                        })

        # Sort by time descending
        all_history.sort(key=lambda x: x.get("checked_at", ""), reverse=True)
        return all_history[:last_n]
