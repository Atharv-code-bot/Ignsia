import { useEffect, useRef, useMemo } from 'react'
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet'
import './MapView.css'

const LAYER_COLORS = {
  tpis: { low: '#fef3c7', mid: '#f59e0b', high: '#dc2626' },
  canopy: { low: '#f0fdf4', mid: '#22c55e', high: '#14532d' },
  lst: { low: '#3b82f6', mid: '#fbbf24', high: '#ef4444' },
  vulnerability: { low: '#f5f3ff', mid: '#8b5cf6', high: '#4c1d95' },
}

function getColor(value, min, max, palette) {
  const t = max > min ? (value - min) / (max - min) : 0
  if (t < 0.5) {
    const u = t * 2
    return interpolateColor(palette.low, palette.mid, u)
  }
  const u = (t - 0.5) * 2
  return interpolateColor(palette.mid, palette.high, u)
}

function interpolateColor(c1, c2, t) {
  const r1 = parseInt(c1.slice(1, 3), 16), g1 = parseInt(c1.slice(3, 5), 16), b1 = parseInt(c1.slice(5, 7), 16)
  const r2 = parseInt(c2.slice(1, 3), 16), g2 = parseInt(c2.slice(3, 5), 16), b2 = parseInt(c2.slice(5, 7), 16)
  const r = Math.round(r1 + (r2 - r1) * t), g = Math.round(g1 + (g2 - g1) * t), b = Math.round(b1 + (b2 - b1) * t)
  return `rgb(${r},${g},${b})`
}

function FitBounds({ zones }) {
  const map = useMap()
  useEffect(() => {
    if (!zones?.features?.length) return
    const coords = zones.features.flatMap(f =>
      f.geometry.coordinates[0].map(c => [c[1], c[0]])
    )
    if (coords.length) {
      const lats = coords.map(c => c[0]), lngs = coords.map(c => c[1])
      map.fitBounds([
        [Math.min(...lats) - 0.01, Math.min(...lngs) - 0.01],
        [Math.max(...lats) + 0.01, Math.max(...lngs) + 0.01],
      ])
    }
  }, [zones, map])
  return null
}

function getActiveProperty(activeLayers) {
  if (activeLayers.tpis) return 'tpis'
  if (activeLayers.canopy) return 'canopy_pct'
  if (activeLayers.lst) return 'mean_lst'
  if (activeLayers.vulnerability) return 'vuln_score'
  return 'tpis'
}

function getActivePalette(activeLayers) {
  if (activeLayers.tpis) return LAYER_COLORS.tpis
  if (activeLayers.canopy) return LAYER_COLORS.canopy
  if (activeLayers.lst) return LAYER_COLORS.lst
  if (activeLayers.vulnerability) return LAYER_COLORS.vulnerability
  return LAYER_COLORS.tpis
}

export default function MapView({ zones, activeLayers, selectedZone, onSelectZone }) {
  const prop = getActiveProperty(activeLayers)
  const palette = getActivePalette(activeLayers)

  const { min, max } = useMemo(() => {
    if (!zones?.features?.length) return { min: 0, max: 1 }
    const vals = zones.features.map(f => f.properties[prop] || 0)
    return { min: Math.min(...vals), max: Math.max(...vals) }
  }, [zones, prop])

  const styleFeature = (feature) => {
    const p = feature.properties
    const value = p[prop] || 0
    const isSelected = p.selected
    const isActive = selectedZone?.properties?.zone_id === p.zone_id
    const isAnomaly = p.anomaly_tag && p.anomaly_tag.length > 0

    let fillColor = getColor(value, min, max, palette)
    let fillOpacity = 0.55
    let weight = 1
    let color = 'rgba(255,255,255,0.15)'
    let dashArray = null

    if (activeLayers.selected && isSelected) {
      weight = 2.5
      color = '#10b981'
      fillOpacity = 0.65
    }

    if (activeLayers.anomaly && isAnomaly) {
      weight = 2.5
      color = p.anomaly_tag.includes('URGENT') ? '#ef4444'
        : p.anomaly_tag.includes('HEAT') ? '#f59e0b'
          : p.anomaly_tag.includes('LOSS') ? '#f97316' : '#ef4444'
      dashArray = '6 3'
    }

    if (isActive) {
      weight = 3
      color = '#60a5fa'
      fillOpacity = 0.8
    }

    if (!activeLayers.water && !p.water_feasible) {
      fillOpacity *= 0.5
      dashArray = '4 4'
    }

    return { fillColor, fillOpacity, weight, color, dashArray }
  }

  const onEachFeature = (feature, layer) => {
    const p = feature.properties
    layer.on({
      click: () => onSelectZone(feature),
      mouseover: (e) => {
        const l = e.target
        l.setStyle({ weight: 3, fillOpacity: 0.8 })
        l.bringToFront()
        const tooltip = `<div style="font-family:Inter,sans-serif;font-size:12px;line-height:1.5">
          <strong>${p.name}</strong><br/>
          TPIS: <b>${p.tpis}</b> · Rank #${p.final_rank}<br/>
          🌡 ${p.mean_lst}°C · 🌿 ${p.canopy_pct}%<br/>
          🌳 ${p.trees_possible} trees
          ${p.anomaly_tag ? `<br/><span style="color:#ef4444">⚠ ${p.anomaly_tag}</span>` : ''}
        </div>`
        l.bindTooltip(tooltip, { sticky: true, className: 'custom-tooltip' }).openTooltip()
      },
      mouseout: (e) => {
        e.target.setStyle(styleFeature(feature))
        e.target.closeTooltip()
      },
    })
  }

  const geoKey = useMemo(() =>
    JSON.stringify(activeLayers) + prop + (selectedZone?.properties?.zone_id || '') +
    zones?.features?.map(f => f.properties.tpis + (f.properties.selected ? 'S' : '')).join(','),
    [zones, activeLayers, prop, selectedZone]
  )

  return (
    <div className="mapview-wrapper">
      <MapContainer
        center={[28.64, 77.23]}
        zoom={12}
        style={{ height: '100%', width: '100%' }}
        zoomControl={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
        {zones?.features && (
          <GeoJSON
            key={geoKey}
            data={zones}
            style={styleFeature}
            onEachFeature={onEachFeature}
          />
        )}
        <FitBounds zones={zones} />
      </MapContainer>

      {/* Legend */}
      <div className="map-legend glass-card">
        <div className="legend-title">
          {prop === 'tpis' ? 'TPIS Score' :
           prop === 'canopy_pct' ? 'Canopy %' :
           prop === 'mean_lst' ? 'LST °C' : 'Vulnerability'}
        </div>
        <div className="legend-gradient">
          <div className="legend-bar" style={{
            background: `linear-gradient(to right, ${palette.low}, ${palette.mid}, ${palette.high})`
          }} />
          <div className="legend-labels">
            <span>{prop === 'canopy_pct' ? '0%' : min.toFixed(1)}</span>
            <span>{prop === 'canopy_pct' ? '50%' : max.toFixed(1)}</span>
          </div>
        </div>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-swatch" style={{ border: '2px solid #10b981' }} />
            <span>Selected Zone</span>
          </div>
          <div className="legend-item">
            <span className="legend-swatch" style={{ border: '2px dashed #ef4444' }} />
            <span>Anomaly</span>
          </div>
        </div>
      </div>
    </div>
  )
}
