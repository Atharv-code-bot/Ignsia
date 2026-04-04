import './ZoneDetail.css'

export default function ZoneDetail({ zone, onClose }) {
  if (!zone) return null
  const p = zone.properties

  const metrics = [
    { label: 'TPIS Score', value: p.tpis?.toFixed(3), color: '#f59e0b', icon: '🎯' },
    { label: 'Final Rank', value: `#${p.final_rank}`, color: '#10b981', icon: '🏆' },
    { label: 'Mean NDVI', value: p.mean_ndvi?.toFixed(3), color: '#22c55e', icon: '🌿' },
    { label: 'Mean LST', value: `${p.mean_lst}°C`, color: '#ef4444', icon: '🌡️' },
    { label: 'Canopy %', value: `${p.canopy_pct}%`, color: '#22c55e', icon: '🌳' },
    { label: 'Bare Land %', value: `${p.bare_pct}%`, color: '#a78bfa', icon: '🏜️' },
    { label: 'Vulnerability', value: p.vuln_score?.toFixed(3), color: '#8b5cf6', icon: '👥' },
    { label: 'Trees Possible', value: p.trees_possible?.toLocaleString(), color: '#06b6d4', icon: '🌲' },
    { label: 'Plantable Area', value: `${(p.plantable_area / 1000).toFixed(1)}k m²`, color: '#06b6d4', icon: '📐' },
  ]

  const roi = [
    { label: 'Cooling', value: p.roi_cooling, icon: '❄️' },
    { label: 'Carbon', value: p.roi_carbon, icon: '🏭' },
    { label: 'Air Quality', value: p.roi_air_quality, icon: '💨' },
    { label: 'Stormwater', value: p.roi_stormwater, icon: '🌧️' },
  ]

  return (
    <div className="zone-detail glass-card animate-slideIn">
      <div className="zd-header">
        <div>
          <h2 className="zd-name">{p.name}</h2>
          <span className="zd-id">{p.zone_id}</span>
        </div>
        <button className="zd-close" onClick={onClose}>✕</button>
      </div>

      {/* Status & Tags */}
      <div className="zd-tags">
        {p.selected && <span className="badge badge-success">✅ Selected</span>}
        {!p.selected && <span className="badge badge-info">Not Selected</span>}
        {p.anomaly_tag && <span className="badge badge-danger">⚠ {p.anomaly_tag}</span>}
        {!p.water_feasible && <span className="badge badge-warning">⚠️ No Water</span>}
      </div>

      {/* Metrics Grid */}
      <div className="zd-metrics">
        {metrics.map(m => (
          <div key={m.label} className="zd-metric">
            <span className="zd-metric-icon">{m.icon}</span>
            <div className="zd-metric-data">
              <span className="zd-metric-value" style={{ color: m.color }}>{m.value}</span>
              <span className="zd-metric-label">{m.label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* ROI Breakdown */}
      <div className="zd-section">
        <h3 className="zd-section-title">Ecosystem Service ROI</h3>
        <div className="roi-total">
          <span className="roi-total-label">Total Annual ROI</span>
          <span className="roi-total-value">${(p.roi_total || 0).toLocaleString()}</span>
        </div>
        <div className="roi-bars">
          {roi.map(r => {
            const pct = p.roi_total ? (r.value / p.roi_total * 100) : 0
            return (
              <div key={r.label} className="roi-bar-item">
                <div className="roi-bar-header">
                  <span>{r.icon} {r.label}</span>
                  <span className="mono-sm">${(r.value || 0).toLocaleString()}</span>
                </div>
                <div className="roi-bar-track">
                  <div className="roi-bar-fill" style={{ width: `${pct}%` }} />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Vulnerability Breakdown */}
      <div className="zd-section">
        <h3 className="zd-section-title">Vulnerability Components</h3>
        <div className="vuln-components">
          <div className="vuln-item">
            <span>Poverty Proxy</span>
            <span className="mono-sm" style={{ color: '#8b5cf6' }}>{(p.poverty_proxy || 0).toFixed(2)}</span>
          </div>
          <div className="vuln-item">
            <span>Health Burden</span>
            <span className="mono-sm" style={{ color: '#ef4444' }}>{(p.health_burden || 0).toFixed(2)}</span>
          </div>
          <div className="vuln-item">
            <span>Thermal Stress</span>
            <span className="mono-sm" style={{ color: '#f59e0b' }}>{(p.thermal_stress || 0).toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
