import { useState } from 'react'
import LayerPanel from './LayerPanel'
import WeightSliders from './WeightSliders'
import RankedTable from './RankedTable'
import ParetoCurve from './ParetoCurve'
import './Sidebar.css'

const TABS = [
  { id: 'controls', label: 'Controls', icon: '⚙️' },
  { id: 'table', label: 'Rankings', icon: '📊' },
  { id: 'pareto', label: 'Pareto', icon: '📈' },
]

export default function Sidebar({
  zones, weights, budget, activeLayers,
  onToggleLayer, onWeightChange, onBudgetChange,
  onSelectZone, selectedZone,
}) {
  const [activeTab, setActiveTab] = useState('controls')
  const [collapsed, setCollapsed] = useState(false)

  const selected = zones.filter(z => z.properties.selected)
  const totalTrees = selected.reduce((s, z) => s + z.properties.trees_possible, 0)
  const totalROI = selected.reduce((s, z) => s + (z.properties.roi_total || 0), 0)
  const anomalies = zones.filter(z => z.properties.anomaly_tag?.length > 0)

  if (collapsed) {
    return (
      <div className="sidebar-collapsed" onClick={() => setCollapsed(false)}>
        <span className="expand-icon">›</span>
        <div className="collapsed-stats">
          <span className="mini-stat">{selected.length} zones</span>
          <span className="mini-stat">{totalTrees} 🌳</span>
        </div>
      </div>
    )
  }

  return (
    <aside className="sidebar">
      {/* Collapse button */}
      <button className="collapse-btn" onClick={() => setCollapsed(true)} title="Collapse sidebar">‹</button>

      {/* Summary cards */}
      <div className="sidebar-summary">
        <div className="summary-card summary-card--green">
          <div className="sc-icon">✅</div>
          <div className="sc-data">
            <div className="sc-value">{selected.length}</div>
            <div className="sc-label">Selected</div>
          </div>
        </div>
        <div className="summary-card summary-card--cyan">
          <div className="sc-icon">🌳</div>
          <div className="sc-data">
            <div className="sc-value">{totalTrees.toLocaleString()}</div>
            <div className="sc-label">Trees</div>
          </div>
        </div>
        <div className="summary-card summary-card--amber">
          <div className="sc-icon">💰</div>
          <div className="sc-data">
            <div className="sc-value">${(totalROI / 1000).toFixed(0)}k</div>
            <div className="sc-label">ROI/yr</div>
          </div>
        </div>
        <div className="summary-card summary-card--red">
          <div className="sc-icon">⚠️</div>
          <div className="sc-data">
            <div className="sc-value">{anomalies.length}</div>
            <div className="sc-label">Anomalies</div>
          </div>
        </div>
      </div>

      {/* Tab nav */}
      <div className="sidebar-tabs">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="sidebar-content">
        {activeTab === 'controls' && (
          <div className="animate-fadeIn">
            <LayerPanel activeLayers={activeLayers} onToggleLayer={onToggleLayer} />

            <div className="section-divider" />

            <WeightSliders
              weights={weights}
              budget={budget}
              onWeightChange={onWeightChange}
              onBudgetChange={onBudgetChange}
            />
          </div>
        )}

        {activeTab === 'table' && (
          <div className="animate-fadeIn">
            <RankedTable
              zones={zones}
              selectedZone={selectedZone}
              onSelectZone={onSelectZone}
            />
          </div>
        )}

        {activeTab === 'pareto' && (
          <div className="animate-fadeIn">
            <ParetoCurve budget={budget} />
          </div>
        )}
      </div>
    </aside>
  )
}
