import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell, ReferenceLine,
} from 'recharts';

const API = 'http://localhost:8002/api';

const DEVICE_ICONS = {
  CCTV: '📷',
  Router: '🌐',
  AccessController: '🔐',
  Generic: '🖥️',
};

const DRIFT_COLORS = {
  DRIFT_NONE: '#22c55e',
  DRIFT_MILD: '#f59e0b',
  DRIFT_STRONG: '#ef4444',
};

const DRIFT_LABELS = {
  DRIFT_NONE: 'Normal',
  DRIFT_MILD: 'Mild Drift',
  DRIFT_STRONG: 'Strong Drift',
};

const SEV_COLORS = { Low: '#22c55e', Medium: '#f59e0b', High: '#ef4444', Critical: '#991b1b' };

function formatTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`;
}

function formatTimestamp(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ── SVG Trust Gauge ────────────────────────────────────────────────
function TrustGauge({ score, size = 100 }) {
  const r = size * 0.4;
  const cx = size / 2;
  const cy = size / 2;
  const circ = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score)) / 100;
  const dash = pct * circ;

  let color = '#22c55e';
  if (score < 40) color = '#ef4444';
  else if (score < 70) color = '#f59e0b';

  return (
    <svg width={size} height={size} style={{ display: 'block', margin: '0 auto' }}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#f1f5f9" strokeWidth={8} />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={8}
        strokeDasharray={`${dash} ${circ - dash}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${cx} ${cy})`}
        style={{ transition: 'stroke-dasharray 0.6s ease' }}
      />
      <text x={cx} y={cy - 2} textAnchor="middle" fontSize={size * 0.16} fontWeight="700" fill={color}>
        {Math.round(score)}
      </text>
      <text x={cx} y={cy + size * 0.14} textAnchor="middle" fontSize={size * 0.09} fill="#94a3b8">TRUST</text>
    </svg>
  );
}

// ── Fleet Stat Card ─────────────────────────────────────────────────
function FleetStatCard({ label, value, icon, color, subtitle }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: '16px 20px',
      border: '1px solid #e2e8f0', boxShadow: '0 2px 6px rgba(0,0,0,0.04)',
      display: 'flex', flexDirection: 'column', gap: 6,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 20 }}>{icon}</span>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: 32, fontWeight: 800, color: color || '#1a1a2e', letterSpacing: -1 }}>
        {value}
      </div>
      {subtitle && <div style={{ fontSize: 11, color: '#94a3b8' }}>{subtitle}</div>}
    </div>
  );
}

// ── Custom Tooltip ──────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div style={{
      background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8,
      padding: '8px 12px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)', fontSize: 12,
    }}>
      <div style={{ color: '#64748b', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: p.color, display: 'inline-block' }} />
          <span style={{ fontWeight: 500 }}>{p.name}:</span>
          <span>{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

// ── Drift Badge ─────────────────────────────────────────────────────
function DriftBadge({ driftClass }) {
  const color = DRIFT_COLORS[driftClass] || '#22c55e';
  const label = DRIFT_LABELS[driftClass] || 'Normal';
  const bg = driftClass === 'DRIFT_STRONG' ? '#fef2f2' : driftClass === 'DRIFT_MILD' ? '#fffbeb' : '#f0fdf4';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      background: bg, color, fontSize: 10, fontWeight: 700,
      padding: '2px 8px', borderRadius: 12, border: `1px solid ${color}33`,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: '50%', background: color,
        animation: driftClass !== 'DRIFT_NONE' ? 'pulse-dot 1.5s infinite' : 'none',
      }} />
      {label}
    </span>
  );
}

// ── Top Drifters Mini Table ─────────────────────────────────────────
function TopDriftersList({ drifters }) {
  if (!drifters || drifters.length === 0) {
    return <div style={{ fontSize: 12, color: '#94a3b8', fontStyle: 'italic' }}>No significant feature drift detected</div>;
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {drifters.map((d, i) => {
        const z = d.z_score;
        const pct = Math.min(100, (z / 5) * 100);
        const color = z > 3 ? '#ef4444' : z > 1.5 ? '#f59e0b' : '#22c55e';
        return (
          <div key={i}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, fontSize: 12 }}>
              <span style={{ fontWeight: 500, color: '#334155', fontFamily: 'monospace' }}>{d.feature}</span>
              <span style={{ fontWeight: 700, color, fontSize: 11 }}>z = {d.z_score.toFixed(2)}</span>
            </div>
            <div style={{ height: 4, background: '#f1f5f9', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2, transition: 'width 0.5s ease' }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── 3-Tier Score Waterfall Bar ──────────────────────────────────────
function ScoreWaterfall({ anomalyDed, driftDed, policyDed, trustScore }) {
  const tiers = [
    { name: 'Anomaly', value: Number(anomalyDed) || 0, color: '#3b82f6' },
    { name: 'Drift', value: Number(driftDed) || 0, color: '#f59e0b' },
    { name: 'Policy', value: Number(policyDed) || 0, color: '#ef4444' },
  ];
  const data = [{ name: 'Deductions', ...Object.fromEntries(tiers.map(t => [t.name, t.value])) }];
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <span style={{ background: '#f0fdf4', color: '#15803d', padding: '4px 10px', borderRadius: 6, fontSize: 13, fontWeight: 700 }}>100</span>
        <span style={{ color: '#94a3b8' }}>→</span>
        {tiers.map((t, i) => t.value > 0 && (
          <React.Fragment key={i}>
            <span style={{ background: '#fef2f2', color: t.color, padding: '4px 10px', borderRadius: 6, fontSize: 12, fontWeight: 600 }}>
              −{t.value.toFixed(1)} {t.name}
            </span>
            <span style={{ color: '#94a3b8' }}>→</span>
          </React.Fragment>
        ))}
        <span style={{
          padding: '4px 12px', borderRadius: 6, fontSize: 14, fontWeight: 800,
          color: trustScore > 80 ? '#15803d' : trustScore > 60 ? '#b45309' : '#dc2626',
          background: trustScore > 80 ? '#f0fdf4' : trustScore > 60 ? '#fffbeb' : '#fef2f2',
        }}>
          = {Number(trustScore).toFixed(1)}
        </span>
      </div>
      <div style={{ height: 40 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis type="category" dataKey="name" hide />
            {tiers.map(t => (
              <Bar key={t.name} dataKey={t.name} stackId="a" fill={t.color} radius={t.name === 'Policy' ? [0, 4, 4, 0] : [0, 0, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
export default function LocalNetworkMonitor() {
  const [monitoredDevices, setMonitoredDevices] = useState([]);
  const [discoveredDevices, setDiscoveredDevices] = useState([]);
  const [selectedDeviceIp, setSelectedDeviceIp] = useState(null);
  const [deviceHistory, setDeviceHistory] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [subnetInput, setSubnetInput] = useState('');
  const [scanMessage, setScanMessage] = useState('');
  const [fleetSummary, setFleetSummary] = useState(null);
  const [driftAlerts, setDriftAlerts] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [modalDevice, setModalDevice] = useState({
    ip: '', hostname: '', mac: '', device_type: 'Generic', custom_name: ''
  });

  const pollIntervalRef = useRef(null);

  // ── Data Fetching ─────────────────────────────────────────────────

  const fetchMonitored = useCallback(async () => {
    try {
      const [mRes, fRes, dRes] = await Promise.all([
        axios.get(`${API}/network/monitored`),
        axios.get(`${API}/network/fleet-summary`),
        axios.get(`${API}/network/drift-alerts?last_n=30`),
      ]);
      setMonitoredDevices(mRes.data);
      setFleetSummary(fRes.data);
      setDriftAlerts(dRes.data);
    } catch (err) {
      console.error('Error fetching monitor data:', err);
    }
  }, []);

  const fetchHistory = useCallback(async (ip) => {
    try {
      const res = await axios.get(`${API}/network/monitored/history?ip=${ip}&last_n=60`);
      setDeviceHistory(res.data);
    } catch (err) {
      console.error('Error fetching device history:', err);
    }
  }, []);

  useEffect(() => {
    fetchMonitored();
    pollIntervalRef.current = setInterval(fetchMonitored, 6000);
    return () => clearInterval(pollIntervalRef.current);
  }, [fetchMonitored]);

  useEffect(() => {
    if (selectedDeviceIp) {
      fetchHistory(selectedDeviceIp);
      const timer = setInterval(() => fetchHistory(selectedDeviceIp), 6000);
      return () => clearInterval(timer);
    }
  }, [selectedDeviceIp, fetchHistory]);

  // ── Subnet Scan ───────────────────────────────────────────────────

  const handleScan = async () => {
    setScanning(true);
    setScanMessage('Scanning network subnet (ping sweep & port banner scan). Please wait...');
    setDiscoveredDevices([]);
    try {
      const url = subnetInput
        ? `${API}/network/discover?subnet=${subnetInput}`
        : `${API}/network/discover`;
      const res = await axios.get(url);
      setDiscoveredDevices(res.data);
      setScanMessage(
        res.data.length === 0
          ? 'Scan completed. No active devices found.'
          : `Scan completed. Found ${res.data.length} active devices.`
      );
    } catch (err) {
      console.error(err);
      setScanMessage('Error performing network scan.');
    }
    setScanning(false);
  };

  // ── Device Registration ───────────────────────────────────────────

  const openAddModal = (dev) => {
    setModalDevice({
      ip: dev.ip,
      hostname: dev.hostname || '',
      mac: dev.mac || '',
      device_type: dev.estimated_type || 'Generic',
      custom_name: dev.hostname || dev.ip,
    });
    setShowAddModal(true);
  };

  const submitAddDevice = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${API}/network/monitored/add`, {
        ip: modalDevice.ip,
        custom_name: modalDevice.custom_name,
        device_type: modalDevice.device_type,
        mac: modalDevice.mac,
      });
      setShowAddModal(false);
      fetchMonitored();
    } catch (err) {
      console.error('Error registering device:', err);
    }
  };

  const handleDeleteDevice = async (ip, e) => {
    e.stopPropagation();
    if (window.confirm(`Stop monitoring device ${ip}?`)) {
      try {
        await axios.post(`${API}/network/monitored/delete`, { ip });
        if (selectedDeviceIp === ip) {
          setSelectedDeviceIp(null);
          setDeviceHistory([]);
        }
        fetchMonitored();
      } catch (err) {
        console.error('Error deleting device:', err);
      }
    }
  };

  const selectedDevice = monitoredDevices.find(d => d.ip === selectedDeviceIp);

  // ── Render ────────────────────────────────────────────────────────

  return (
    <>
      <style>{`
        @keyframes pulse-dot { 0%,100%{opacity:0.3; transform:scale(1)} 50%{opacity:1; transform:scale(1.2)} }
        @keyframes drift-ring { 0%,100%{ box-shadow: 0 0 0 0 rgba(245,158,11,0); } 50%{ box-shadow: 0 0 0 6px rgba(245,158,11,0.25); } }
        @keyframes drift-ring-strong { 0%,100%{ box-shadow: 0 0 0 0 rgba(239,68,68,0); } 50%{ box-shadow: 0 0 0 8px rgba(239,68,68,0.3); } }

        .net-card {
          background: rgba(255,255,255,0.85);
          backdrop-filter: blur(12px);
          border: 1px solid var(--border-light);
          border-radius: var(--radius-lg);
          padding: 20px;
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
        }
        .net-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); border-color: #3b82f6aa; }
        .net-card.selected { border: 2px solid #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
        .net-card.drift-mild { border-color: #f59e0b99; animation: drift-ring 2s ease-in-out infinite; }
        .net-card.drift-strong { border-color: #ef444499; animation: drift-ring-strong 1.5s ease-in-out infinite; }

        .status-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px; }
        .status-pill.online { background: #dcfce7; color: #15803d; }
        .status-pill.offline { background: #fee2e2; color: #dc2626; }
        .pulse-circle { width: 8px; height: 8px; border-radius: 50%; animation: pulse-dot 1.5s infinite; }
        .pulse-circle.green { background: #22c55e; }
        .pulse-circle.red { background: #ef4444; }

        .modal-overlay { position: fixed; top:0; left:0; right:0; bottom:0; background: rgba(15,23,42,0.55); backdrop-filter: blur(4px); display:flex; align-items:center; justify-content:center; z-index:1000; }
        .modal-container { background: #fff; border-radius: var(--radius-lg); border: 1px solid var(--border-light); width: 440px; max-width: 90%; box-shadow: var(--shadow-lg); overflow: hidden; }
        .modal-header { padding: 16px 20px; border-bottom: 1px solid var(--border-light); display: flex; justify-content: space-between; align-items: center; }
        .modal-body { padding: 20px; }
        .form-group { margin-bottom: 14px; }
        .form-group label { display: block; font-size: 12px; font-weight: 600; color: #475569; margin-bottom: 6px; }
        .form-group input, .form-group select { width: 100%; padding: 8px 12px; border: 1px solid var(--border-light); border-radius: var(--radius-sm); font-size: 13px; outline: none; font-family: inherit; }
        .form-group input:focus, .form-group select:focus { border-color: #3b82f6; }
        .modal-footer { padding: 12px 20px; border-top: 1px solid var(--border-light); display: flex; justify-content: flex-end; gap: 10px; background: #f8fafc; }
        .btn-primary { background: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: var(--radius-sm); font-size: 13px; font-weight: 600; cursor: pointer; font-family: inherit; transition: background 0.2s; }
        .btn-primary:hover { background: #2563eb; }
        .btn-secondary { background: white; border: 1px solid var(--border-light); padding: 8px 16px; border-radius: var(--radius-sm); font-size: 13px; font-weight: 500; cursor: pointer; color: #475569; font-family: inherit; }
        .btn-secondary:hover { background: #f1f5f9; }

        .alert-row { transition: background 0.2s; }
        .alert-row:hover { background: #f8fafc !important; }
      `}</style>

      {/* ── Page Header ── */}
      <div className="page-title-section">
        <h1 className="page-title">📡 Real IoT Network Monitor</h1>
        <p className="page-subtitle">Scan, register, and monitor physical IoT devices — with live drift detection and trust scoring</p>
      </div>

      {/* ── Fleet Summary Header ── */}
      {fleetSummary && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 14, marginBottom: 24 }}>
          <FleetStatCard
            label="Devices Monitored" icon="📡"
            value={fleetSummary.total_devices}
            subtitle={`${fleetSummary.online} online · ${fleetSummary.offline} offline`}
            color="#1a1a2e"
          />
          <FleetStatCard
            label="Avg Trust Score" icon="🛡️"
            value={fleetSummary.avg_trust}
            color={fleetSummary.avg_trust > 80 ? '#15803d' : fleetSummary.avg_trust > 60 ? '#b45309' : '#dc2626'}
            subtitle="Across online devices"
          />
          <FleetStatCard
            label="In Drift" icon="📈"
            value={fleetSummary.in_drift}
            color={fleetSummary.in_drift > 0 ? '#f59e0b' : '#22c55e'}
            subtitle="Behavioral deviation detected"
          />
          <FleetStatCard
            label="Critical / High" icon="🚨"
            value={fleetSummary.critical_count}
            color={fleetSummary.critical_count > 0 ? '#dc2626' : '#22c55e'}
            subtitle="Severity alerts"
          />
          <FleetStatCard
            label="Policy Violations" icon="⚠️"
            value={fleetSummary.policy_violations}
            color={fleetSummary.policy_violations > 0 ? '#ef4444' : '#22c55e'}
            subtitle="Active rule breaches"
          />
        </div>
      )}

      {/* ── Subnet Scan Control Panel ── */}
      <div className="panel" style={{ marginBottom: 24 }}>
        <div className="panel-header">
          <div>
            <div className="panel-title">Subnet Discovery Scanner</div>
            <div className="panel-subtitle">Perform concurrent ping sweeps to locate responsive network endpoints</div>
          </div>
        </div>
        <div className="panel-body">
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
            <input
              type="text"
              value={subnetInput}
              onChange={(e) => setSubnetInput(e.target.value)}
              placeholder="e.g. 192.168.1. (leave empty to auto-detect)"
              style={{
                flex: 1, minWidth: 240, background: '#fff', border: '1px solid var(--border-light)',
                padding: '10px 14px', borderRadius: 8, fontSize: 13, fontFamily: 'inherit', outline: 'none',
              }}
              disabled={scanning}
            />
            <button
              onClick={handleScan}
              className="btn-primary"
              style={{ padding: '10px 20px', borderRadius: 8, whiteSpace: 'nowrap' }}
              disabled={scanning}
            >
              {scanning ? '🔍 Scanning...' : '🔍 Scan Subnet'}
            </button>
          </div>

          {scanMessage && (
            <div style={{ marginTop: 14, fontSize: 12, color: '#475569', display: 'flex', alignItems: 'center', gap: 8 }}>
              {scanning && <div className="loading-spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />}
              <span>{scanMessage}</span>
            </div>
          )}

          {discoveredDevices.length > 0 && (
            <div style={{ marginTop: 20, borderTop: '1px solid var(--border-light)', paddingTop: 16 }}>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: '#334155' }}>
                Active Hosts Found ({discoveredDevices.length}):
              </div>
              <div style={{ maxHeight: 260, overflowY: 'auto', border: '1px solid var(--border-light)', borderRadius: 8 }}>
                <table className="data-table" style={{ margin: 0 }}>
                  <thead>
                    <tr>
                      <th>IP Address</th>
                      <th>Hostname</th>
                      <th>MAC Address</th>
                      <th>Latency</th>
                      <th>Open Ports</th>
                      <th>Est. Type</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {discoveredDevices.map((dev) => {
                      const isMonitored = monitoredDevices.some(d => d.ip === dev.ip);
                      return (
                        <tr key={dev.ip}>
                          <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{dev.ip}</td>
                          <td>{dev.hostname || '—'}</td>
                          <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{dev.mac || 'unknown'}</td>
                          <td style={{ fontVariantNumeric: 'tabular-nums' }}>{dev.rtt_ms} ms</td>
                          <td>
                            {dev.open_ports && dev.open_ports.length > 0 ? (
                              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                                {dev.open_ports.map(p => (
                                  <span key={p} style={{ background: '#eff6ff', color: '#1d4ed8', padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600 }}>
                                    {p}
                                  </span>
                                ))}
                              </div>
                            ) : 'None'}
                          </td>
                          <td style={{ fontWeight: 500 }}>{DEVICE_ICONS[dev.estimated_type]} {dev.estimated_type}</td>
                          <td>
                            {isMonitored ? (
                              <span style={{ color: '#22c55e', fontSize: 12, fontWeight: 600 }}>✓ Monitored</span>
                            ) : (
                              <button
                                onClick={() => openAddModal(dev)}
                                className="btn-secondary"
                                style={{ padding: '4px 10px', fontSize: 11, borderRadius: 6 }}
                              >
                                + Monitor
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Monitored Devices Grid ── */}
      <div style={{ marginBottom: 14 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
          📡 Monitored Network Nodes
        </h2>
        <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
          Click any card to inspect drift timelines, score breakdowns, and top drifting features
        </p>
      </div>

      {monitoredDevices.length === 0 ? (
        <div className="panel" style={{ textAlign: 'center', padding: 48, marginBottom: 24 }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>🔍</div>
          <div style={{ fontWeight: 700, fontSize: 16, color: '#334155', marginBottom: 6 }}>
            No network devices registered
          </div>
          <div style={{ color: '#64748b', fontSize: 13, maxWidth: 440, margin: '0 auto' }}>
            Trigger a subnet scan above, then add a device to monitor. The system will start periodic
            latency tracking, port compliance audits, and dynamic trust + drift computation.
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 16, marginBottom: 24 }}>
          {monitoredDevices.map((dev) => {
            const isSelected = selectedDeviceIp === dev.ip;
            const driftClass = dev.drift_class || 'DRIFT_NONE';
            const driftCardClass = driftClass === 'DRIFT_STRONG' ? 'drift-strong'
              : driftClass === 'DRIFT_MILD' ? 'drift-mild' : '';

            return (
              <div
                key={dev.ip}
                className={`net-card${isSelected ? ' selected' : ''} ${driftCardClass}`}
                onClick={() => setSelectedDeviceIp(isSelected ? null : dev.ip)}
              >
                <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
                  <TrustGauge score={dev.trust_score_smoothed ?? 100} size={90} />

                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
                      <div>
                        <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)' }}>
                          {DEVICE_ICONS[dev.device_type] || '🖥️'} {dev.custom_name}
                        </div>
                        <div style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-secondary)', marginTop: 1 }}>
                          {dev.ip}
                        </div>
                      </div>
                      <span className={`status-pill ${dev.alive ? 'online' : 'offline'}`}>
                        <span className={`pulse-circle ${dev.alive ? 'green' : 'red'}`} />
                        {dev.alive ? 'ONLINE' : 'OFFLINE'}
                      </span>
                    </div>

                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                      <span className={`severity-badge ${(dev.severity || 'low').toLowerCase()}`} style={{ fontSize: 10 }}>
                        {dev.severity || 'Low'}
                      </span>
                      <DriftBadge driftClass={driftClass} />
                      {dev.policy_status && dev.policy_status !== 'COMPLIANT' && (
                        <span style={{ background: '#fef2f2', color: '#dc2626', fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 10, border: '1px solid #fecaca' }}>
                          {dev.policy_status.replace('_', ' ')}
                        </span>
                      )}
                    </div>

                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
                      MAC: <span style={{ fontFamily: 'monospace' }}>{dev.mac || 'unknown'}</span>
                      {dev.baseline_samples > 0 && (
                        <span style={{ marginLeft: 8, color: dev.baseline_samples >= 5 ? '#22c55e' : '#f59e0b' }}>
                          · {dev.baseline_samples >= 5 ? `Baseline: ${dev.baseline_samples} samples` : `Learning (${dev.baseline_samples}/5)`}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Sub-panel metrics */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginTop: 16, borderTop: '1px solid #f1f5f9', paddingTop: 12 }}>
                  {[
                    { label: 'Latency', value: dev.alive ? `${dev.ping_rtt_ms} ms` : '—' },
                    { label: 'Pkt Loss', value: `${dev.packet_loss_pct || 0}%`, color: dev.packet_loss_pct > 0 ? '#ef4444' : undefined },
                    { label: 'Drift Z', value: dev.drift_magnitude ? dev.drift_magnitude.toFixed(2) : '0.00', color: (dev.drift_magnitude || 0) > 3 ? '#ef4444' : (dev.drift_magnitude || 0) > 1.5 ? '#f59e0b' : undefined },
                    { label: 'Ports', value: dev.open_ports && dev.open_ports.length > 0 ? dev.open_ports.join(', ') : 'None' },
                  ].map(m => (
                    <div key={m.label} style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 10, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>{m.label}</div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: m.color || '#334155', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {m.value}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Policy alert snippet */}
                {dev.policy_status !== 'COMPLIANT' && dev.alive && (
                  <div style={{ marginTop: 10, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 6, padding: '6px 10px', fontSize: 11, color: '#b45309' }}>
                    ⚠️ {dev.policy_violations}
                  </div>
                )}

                {/* Card footer */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
                  <span style={{ fontSize: 10, color: '#94a3b8' }}>
                    {dev.checked_at ? `Updated ${formatTime(dev.checked_at)}` : 'Pending first check…'}
                  </span>
                  <button
                    onClick={(e) => handleDeleteDevice(dev.ip, e)}
                    style={{ background: 'transparent', border: 'none', color: '#94a3b8', fontSize: 11, cursor: 'pointer' }}
                  >
                    ✕ Stop Monitoring
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Selected Device Analytics Detail Panel ── */}
      {selectedDevice && (
        <div className="panel" style={{ marginBottom: 24, borderTop: '4px solid #3b82f6' }}>
          <div className="panel-header">
            <div>
              <div className="panel-title">
                {DEVICE_ICONS[selectedDevice.device_type]} {selectedDevice.custom_name} — Analytics
              </div>
              <div className="panel-subtitle">
                Drift trends, score breakdown, and top drifting features for {selectedDevice.ip}
              </div>
            </div>
            <button
              onClick={() => { setSelectedDeviceIp(null); setDeviceHistory([]); }}
              style={{ background: '#f1f5f9', border: 'none', borderRadius: 6, padding: '4px 10px', cursor: 'pointer', fontSize: 11, color: '#64748b' }}
            >
              ✕ Close
            </button>
          </div>
          <div className="panel-body">

            {/* Score Waterfall */}
            <div style={{ background: '#f8fafc', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0', marginBottom: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#1a1a2e', marginBottom: 12 }}>
                Score Waterfall — 3-Tier Deduction
              </div>
              <ScoreWaterfall
                anomalyDed={selectedDevice.anomaly_deduction}
                driftDed={selectedDevice.drift_deduction}
                policyDed={selectedDevice.policy_deduction}
                trustScore={selectedDevice.trust_score_smoothed}
              />
            </div>

            {/* Charts Row */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
              {/* Trust Score Timeline */}
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 8 }}>
                  Trust Score Over Time
                </div>
                <div style={{ height: 180 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={deviceHistory} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="checked_at" tickFormatter={formatTime} tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <Tooltip content={<ChartTooltip />} labelFormatter={formatTime} />
                      <ReferenceLine y={80} stroke="#22c55e" strokeDasharray="4 2" strokeWidth={1} />
                      <ReferenceLine y={60} stroke="#f59e0b" strokeDasharray="4 2" strokeWidth={1} />
                      <Area type="monotone" dataKey="trust_score_smoothed" name="Trust Score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} strokeWidth={2} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Drift Magnitude Timeline */}
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 8 }}>
                  Behavioral Drift Magnitude (Z-Score)
                </div>
                <div style={{ height: 180 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={deviceHistory} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="checked_at" tickFormatter={formatTime} tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <Tooltip content={<ChartTooltip />} labelFormatter={formatTime} />
                      <ReferenceLine y={1.5} stroke="#f59e0b" strokeDasharray="4 2" strokeWidth={1} label={{ value: 'Mild', fill: '#f59e0b', fontSize: 9, position: 'right' }} />
                      <ReferenceLine y={3.0} stroke="#ef4444" strokeDasharray="4 2" strokeWidth={1} label={{ value: 'Strong', fill: '#ef4444', fontSize: 9, position: 'right' }} />
                      <Area
                        type="monotone"
                        dataKey="drift_magnitude"
                        name="Drift (Z)"
                        stroke="#f59e0b"
                        fill="#f59e0b"
                        fillOpacity={0.15}
                        strokeWidth={2}
                        dot={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Latency / Loss Timeline */}
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 8 }}>
                  Network Latency & Packet Loss
                </div>
                <div style={{ height: 180 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={deviceHistory} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="checked_at" tickFormatter={formatTime} tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <YAxis yAxisId="left" tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <Tooltip content={<ChartTooltip />} labelFormatter={formatTime} />
                      <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 10 }} />
                      <Line yAxisId="left" type="monotone" dataKey="ping_rtt_ms" name="RTT (ms)" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
                      <Line yAxisId="right" type="monotone" dataKey="packet_loss_pct" name="Loss (%)" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Anomaly Deduction Timeline */}
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 8 }}>
                  Deduction Components Over Time
                </div>
                <div style={{ height: 180 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={deviceHistory} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="checked_at" tickFormatter={formatTime} tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} />
                      <Tooltip content={<ChartTooltip />} labelFormatter={formatTime} />
                      <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 10 }} />
                      <Area type="monotone" dataKey="anomaly_deduction" name="Anomaly" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} strokeWidth={1.5} dot={false} />
                      <Area type="monotone" dataKey="drift_deduction" name="Drift" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.6} strokeWidth={1.5} dot={false} />
                      <Area type="monotone" dataKey="policy_deduction" name="Policy" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} strokeWidth={1.5} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Top Drifters + Current Status Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              {/* Top Drifting Features */}
              <div style={{ background: '#fff', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1a1a2e', marginBottom: 4 }}>
                  Top Drifting Features
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 12 }}>
                  Behavioral features with highest z-score deviation from learned baseline
                  {selectedDevice.baseline_samples > 0 && (
                    <span style={{
                      marginLeft: 8, padding: '1px 6px', borderRadius: 10, fontSize: 10, fontWeight: 600,
                      background: selectedDevice.baseline_samples >= 5 ? '#f0fdf4' : '#fffbeb',
                      color: selectedDevice.baseline_samples >= 5 ? '#15803d' : '#b45309',
                    }}>
                      {selectedDevice.baseline_samples >= 5
                        ? `Baseline: ${selectedDevice.baseline_samples} samples`
                        : `Learning baseline (${selectedDevice.baseline_samples}/5 samples)`}
                    </span>
                  )}
                </div>
                <TopDriftersList drifters={selectedDevice.top_drifters_parsed || []} />
              </div>

              {/* Current Status Summary */}
              <div style={{ background: '#fff', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1a1a2e', marginBottom: 12 }}>
                  Current Status Snapshot
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                  {[
                    { label: 'Trust Score', value: `${(selectedDevice.trust_score_smoothed || 0).toFixed(1)}`, unit: '/ 100' },
                    { label: 'Anomaly Score', value: `${(selectedDevice.anomaly_score || 0).toFixed(3)}`, unit: '(0–1)' },
                    { label: 'Drift Class', value: DRIFT_LABELS[selectedDevice.drift_class] || 'Normal' },
                    { label: 'Drift Magnitude', value: `${(selectedDevice.drift_magnitude || 0).toFixed(2)}`, unit: 'Z' },
                    { label: 'Policy Status', value: (selectedDevice.policy_status || 'COMPLIANT').replace('_', ' ') },
                    { label: 'Severity', value: selectedDevice.severity || 'Low' },
                  ].map(m => (
                    <div key={m.label} style={{ background: '#f8fafc', borderRadius: 8, padding: '8px 10px' }}>
                      <div style={{ fontSize: 10, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600, marginBottom: 2 }}>{m.label}</div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: '#334155' }}>
                        {m.value}
                        {m.unit && <span style={{ fontSize: 10, color: '#94a3b8', marginLeft: 3 }}>{m.unit}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

          </div>
        </div>
      )}

      {/* ── Drift Alert Feed ── */}
      {driftAlerts.length > 0 && (
        <div className="panel" style={{ marginBottom: 24 }}>
          <div className="panel-header">
            <div>
              <div className="panel-title">🔔 Drift & Violation Alert Feed</div>
              <div className="panel-subtitle">
                Recent behavioral drift and policy violation events across all monitored devices
              </div>
            </div>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>{driftAlerts.length} events</span>
          </div>
          <div className="panel-body no-padding">
            <table className="data-table" style={{ margin: 0 }}>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Device</th>
                  <th>IP</th>
                  <th>Drift</th>
                  <th>Z-Score</th>
                  <th>Policy</th>
                  <th>Top Feature</th>
                  <th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {driftAlerts.map((alert, i) => (
                  <tr
                    key={i}
                    className="alert-row"
                    style={{
                      background: alert.drift_class === 'DRIFT_STRONG' || alert.policy_status === 'HARD_VIOLATION'
                        ? '#fef9f9' : i % 2 === 0 ? '#fff' : '#fafafa'
                    }}
                  >
                    <td style={{ fontSize: 10, fontFamily: 'monospace', color: '#64748b', whiteSpace: 'nowrap' }}>
                      {formatTimestamp(alert.checked_at)}
                    </td>
                    <td style={{ fontWeight: 600, fontSize: 13 }}>
                      {DEVICE_ICONS[alert.device_type] || '🖥️'} {alert.custom_name}
                    </td>
                    <td style={{ fontFamily: 'monospace', fontSize: 12, color: '#64748b' }}>{alert.ip}</td>
                    <td>
                      <DriftBadge driftClass={alert.drift_class} />
                    </td>
                    <td style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 700, color: (alert.drift_magnitude || 0) > 3 ? '#ef4444' : '#f59e0b' }}>
                      {(alert.drift_magnitude || 0).toFixed(2)}
                    </td>
                    <td>
                      {alert.policy_status !== 'COMPLIANT' ? (
                        <span style={{ fontSize: 11, fontWeight: 600, color: alert.policy_status === 'HARD_VIOLATION' ? '#dc2626' : '#b45309' }}>
                          {alert.policy_status.replace('_', ' ')}
                        </span>
                      ) : (
                        <span style={{ fontSize: 11, color: '#22c55e' }}>✓ Compliant</span>
                      )}
                    </td>
                    <td style={{ fontSize: 11, fontFamily: 'monospace', color: '#64748b' }}>
                      {alert.top_drifters_parsed && alert.top_drifters_parsed.length > 0
                        ? `${alert.top_drifters_parsed[0].feature} (z=${alert.top_drifters_parsed[0].z_score})`
                        : '—'}
                    </td>
                    <td>
                      <span className={`severity-badge ${(alert.severity || 'low').toLowerCase()}`} style={{ fontSize: 10 }}>
                        {alert.severity || 'Low'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Registration Modal ── */}
      {showAddModal && (
        <div className="modal-overlay">
          <div className="modal-container">
            <div className="modal-header">
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#1e293b', margin: 0 }}>
                Register Device for Monitoring
              </h3>
              <button
                onClick={() => setShowAddModal(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 16, color: '#94a3b8' }}
              >✕</button>
            </div>
            <form onSubmit={submitAddDevice}>
              <div className="modal-body">
                <div className="form-group">
                  <label>IP Address</label>
                  <input type="text" value={modalDevice.ip} disabled style={{ background: '#f1f5f9', color: '#64748b' }} />
                </div>
                <div className="form-group">
                  <label>MAC Address</label>
                  <input type="text" value={modalDevice.mac || 'unknown'} disabled style={{ background: '#f1f5f9', color: '#64748b' }} />
                </div>
                <div className="form-group">
                  <label>Custom Name</label>
                  <input
                    type="text"
                    required
                    value={modalDevice.custom_name}
                    onChange={(e) => setModalDevice({ ...modalDevice, custom_name: e.target.value })}
                    placeholder="Enter custom device label"
                  />
                </div>
                <div className="form-group">
                  <label>Device Profile Template</label>
                  <select
                    value={modalDevice.device_type}
                    onChange={(e) => setModalDevice({ ...modalDevice, device_type: e.target.value })}
                  >
                    <option value="CCTV">CCTV IP Camera (Expects RTSP 554)</option>
                    <option value="Router">Network Router (Expects DNS 53)</option>
                    <option value="AccessController">Access Controller (Expects HTTPS 443)</option>
                    <option value="Generic">Generic Node (Open / No Port Enforcement)</option>
                  </select>
                </div>
                <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, padding: '10px 14px', fontSize: 12, color: '#1e40af' }}>
                  <strong>How trust is scored:</strong> The system will ping this device every ~8 seconds,
                  scan its ports, and build a rolling behavioral baseline. After 5 checks, it starts
                  computing drift z-scores relative to that learned baseline.
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn-secondary" onClick={() => setShowAddModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary">Start Monitoring</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
