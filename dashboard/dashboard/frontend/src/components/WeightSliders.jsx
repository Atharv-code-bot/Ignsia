import { useState, useCallback } from 'react'
import './WeightSliders.css'

const WEIGHT_CONFIG = [
  { key: 'canopy_deficit', label: 'Canopy Deficit', icon: '🌿', color: '#22c55e' },
  { key: 'thermal_stress', label: 'Thermal Stress', icon: '🌡️', color: '#ef4444' },
  { key: 'vulnerability', label: 'Vulnerability', icon: '👥', color: '#8b5cf6' },
  { key: 'plantability', label: 'Plantability', icon: '🌱', color: '#06b6d4' },
  { key: 'roi_norm', label: 'Ecosystem ROI', icon: '💰', color: '#f59e0b' },
]

export default function WeightSliders({ weights, budget, onWeightChange, onBudgetChange }) {
  const [localWeights, setLocalWeights] = useState({ ...weights })
  const [localBudget, setLocalBudget] = useState(budget)

  const handleWeight = useCallback((key, value) => {
    const newW = { ...localWeights, [key]: parseFloat(value) }
    setLocalWeights(newW)
  }, [localWeights])

  const handleApply = useCallback(() => {
    // Normalize weights to sum to 1
    const sum = Object.values(localWeights).reduce((a, b) => a + b, 0)
    const normalized = {}
    for (const [k, v] of Object.entries(localWeights)) {
      normalized[k] = +(v / sum).toFixed(3)
    }
    onWeightChange(normalized, localBudget)
  }, [localWeights, localBudget, onWeightChange])

  const total = Object.values(localWeights).reduce((a, b) => a + b, 0)

  return (
    <div className="weight-sliders">
      <h3 className="panel-title">TPIS Weights</h3>
      <p className="weight-hint">
        Adjust how the impact score is calculated. Weights auto-normalize on apply.
      </p>

      {WEIGHT_CONFIG.map(w => (
        <div key={w.key} className="slider-container">
          <div className="slider-label">
            <span><span className="slider-icon">{w.icon}</span> {w.label}</span>
            <span className="slider-value" style={{ color: w.color }}>
              {(localWeights[w.key] / total * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={localWeights[w.key]}
            onChange={(e) => handleWeight(w.key, e.target.value)}
            style={{ accentColor: w.color }}
          />
        </div>
      ))}

      <div className="section-divider" />

      <h3 className="panel-title">Budget Constraint</h3>
      <div className="slider-container">
        <div className="slider-label">
          <span>🌳 Max Trees</span>
          <span className="slider-value">{localBudget.toLocaleString()}</span>
        </div>
        <input
          type="range"
          min="100"
          max="5000"
          step="50"
          value={localBudget}
          onChange={(e) => setLocalBudget(parseInt(e.target.value))}
        />
      </div>

      <button className="btn btn-primary apply-btn" onClick={handleApply}>
        ⚡ Apply & Re-optimize
      </button>
    </div>
  )
}
