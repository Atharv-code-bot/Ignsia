import { useState, useEffect, useMemo } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { fetchPareto } from '../utils/api'
import './ParetoCurve.css'

export default function ParetoCurve({ budget }) {
  const [data, setData] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const result = await fetchPareto()
        setData(result.pareto)
      } catch {
        // Generate demo pareto curve
        const curve = []
        for (let b = 100; b <= 2100; b += 100) {
          const impact = b * 0.8 * (1 - Math.exp(-b / 800))
          curve.push({
            budget: b,
            total_impact: +impact.toFixed(2),
            zones_selected: Math.min(Math.floor(b / 80), 25),
            total_trees: Math.min(b, 2000),
          })
        }
        setData(curve)
      }
    }
    load()
  }, [])

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
      <div className="pareto-tooltip glass-card-solid">
        <div className="pt-row">
          <span>Budget</span>
          <span className="pt-value">{d.budget} trees</span>
        </div>
        <div className="pt-row">
          <span>Impact</span>
          <span className="pt-value" style={{ color: 'var(--green-400)' }}>{d.total_impact.toFixed(1)}</span>
        </div>
        <div className="pt-row">
          <span>Zones</span>
          <span className="pt-value">{d.zones_selected}</span>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="pareto-loading">
        <p className="text-muted">Loading Pareto curve...</p>
      </div>
    )
  }

  return (
    <div className="pareto-curve">
      <h3 className="panel-title">Budget Efficiency Frontier</h3>
      <p className="pareto-hint">
        Shows impact vs. budget trade-off. Current budget marked.
      </p>

      <div className="pareto-chart">
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="impactGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="budget"
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              tickLine={false}
              label={{ value: 'Budget (trees)', position: 'insideBottom', fill: '#64748b', fontSize: 10, offset: -4 }}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              tickLine={false}
              label={{ value: 'Impact', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              x={budget}
              stroke="#f59e0b"
              strokeDasharray="6 3"
              strokeWidth={2}
              label={{ value: 'Current', position: 'top', fill: '#f59e0b', fontSize: 10 }}
            />
            <Area
              type="monotone"
              dataKey="total_impact"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#impactGrad)"
              dot={false}
              activeDot={{ r: 4, fill: '#10b981', stroke: '#0a0e17', strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Efficiency stats */}
      <div className="pareto-stats">
        {data.length > 0 && (() => {
          const current = data.find(d => d.budget >= budget) || data[data.length - 1]
          const marginal = data.length > 1
            ? (data[data.length - 1].total_impact - data[0].total_impact) / (data[data.length - 1].budget - data[0].budget)
            : 0
          return (
            <>
              <div className="ps-item">
                <span className="ps-label">At current budget</span>
                <span className="ps-value">{current.total_impact.toFixed(1)} impact</span>
              </div>
              <div className="ps-item">
                <span className="ps-label">Zones at budget</span>
                <span className="ps-value">{current.zones_selected}</span>
              </div>
              <div className="ps-item">
                <span className="ps-label">Avg marginal return</span>
                <span className="ps-value">{marginal.toFixed(3)}/tree</span>
              </div>
            </>
          )
        })()}
      </div>
    </div>
  )
}
