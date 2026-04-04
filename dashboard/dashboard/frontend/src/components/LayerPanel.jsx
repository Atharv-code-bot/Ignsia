import './LayerPanel.css'

const LAYERS = [
  { id: 'tpis', label: 'TPIS Score', color: '#f59e0b', desc: 'Tree Planting Impact Score' },
  { id: 'canopy', label: 'Canopy Coverage', color: '#22c55e', desc: 'Tree canopy percentage' },
  { id: 'lst', label: 'Surface Temperature', color: '#ef4444', desc: 'Land Surface Temperature °C' },
  { id: 'vulnerability', label: 'Vulnerability', color: '#8b5cf6', desc: 'Social vulnerability index' },
  { id: 'selected', label: 'Selected Zones', color: '#10b981', desc: 'Budget-optimized zones' },
  { id: 'water', label: 'Water Buffer', color: '#3b82f6', desc: '150m water infrastructure' },
  { id: 'anomaly', label: 'Anomaly Flags', color: '#ef4444', desc: 'IF-detected outliers' },
  { id: 'segmentation', label: 'Land Cover', color: '#06b6d4', desc: 'Canopy/Built/Bare mask' },
]

export default function LayerPanel({ activeLayers, onToggleLayer }) {
  return (
    <div className="layer-panel">
      <h3 className="panel-title">Map Layers</h3>
      <div className="layer-list">
        {LAYERS.map(layer => (
          <div
            key={layer.id}
            className={`layer-item ${activeLayers[layer.id] ? 'active' : ''}`}
            onClick={() => onToggleLayer(layer.id)}
          >
            <div className="layer-color" style={{ background: layer.color, opacity: activeLayers[layer.id] ? 1 : 0.3 }} />
            <div className="layer-info">
              <span className="layer-name">{layer.label}</span>
              <span className="layer-desc">{layer.desc}</span>
            </div>
            <div className={`toggle ${activeLayers[layer.id] ? 'active' : ''}`}>
              <div className="toggle-knob" />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
