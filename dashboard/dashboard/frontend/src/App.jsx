import { useState, useEffect, useCallback } from 'react'
import MapView from './components/MapView'
import Sidebar from './components/Sidebar'
import ZoneDetail from './components/ZoneDetail'
import DataSourceWarning from './components/DataSourceWarning'
import { fetchZones, reoptimize, healthCheck } from './utils/api'
import './App.css'

// Demo zones for when API is offline
function generateDemoZones() {
  const zones = []
  const tags = ['', '', '', 'ANOMALY — URGENT', 'ACTIVE LOSS', 'HEAT EMERGENCY', '']
  for (let i = 0; i < 25; i++) {
    const lat = 28.60 + Math.random() * 0.08
    const lon = 77.18 + Math.random() * 0.10
    const sz = 0.012 + Math.random() * 0.008
    const canopy = 3 + Math.random() * 32
    const lst = 28 + Math.random() * 14
    const vuln = 0.1 + Math.random() * 0.85
    const trees = Math.floor(20 + Math.random() * 280)
    const tpis = +(0.2 + Math.random() * 0.78).toFixed(3)
    const roi = Math.round(5000 + Math.random() * 75000)
    const sel = Math.random() > 0.4
    zones.push({
      type: 'Feature',
      properties: {
        zone_id: `zone_${String(i).padStart(3, '0')}`,
        name: `Ward ${i + 1}`,
        tpis, final_rank: i + 1,
        mean_ndvi: +(0.05 + Math.random() * 0.5).toFixed(3),
        mean_lst: +lst.toFixed(1),
        canopy_pct: +canopy.toFixed(1),
        bare_pct: +(100 - canopy - 20 - Math.random() * 40).toFixed(1),
        vuln_score: +vuln.toFixed(3),
        trees_possible: trees,
        plantable_area: Math.round(5000 + Math.random() * 45000),
        roi_total: roi,
        roi_cooling: Math.round(roi * 0.3),
        roi_carbon: Math.round(roi * 0.25),
        roi_air_quality: Math.round(roi * 0.25),
        roi_stormwater: Math.round(roi * 0.2),
        anomaly_tag: tags[i % tags.length],
        status: sel ? '✅ GO' : '⚠️ H₂O',
        selected: sel,
        water_feasible: Math.random() > 0.2,
        thermal_stress: +(0.1 + Math.random() * 0.8).toFixed(3),
        canopy_deficit: +Math.max(0, 0.30 - canopy / 100).toFixed(3),
        poverty_proxy: +(0.1 + Math.random() * 0.8).toFixed(3),
        health_burden: +(0.1 + Math.random() * 0.7).toFixed(3),
      },
      geometry: {
        type: 'Polygon',
        coordinates: [[[lon, lat], [lon + sz, lat], [lon + sz, lat + sz], [lon, lat + sz], [lon, lat]]],
      },
    })
  }
  zones.sort((a, b) => b.properties.tpis - a.properties.tpis)
  zones.forEach((z, i) => { z.properties.final_rank = i + 1 })
  return { type: 'FeatureCollection', features: zones }
}

export default function App() {
  const [zones, setZones] = useState(null)
  const [selectedZone, setSelectedZone] = useState(null)
  const [apiOnline, setApiOnline] = useState(false)
  const [loading, setLoading] = useState(true)
  const [isRealData, setIsRealData] = useState(false)
  const [dataWarning, setDataWarning] = useState('')
  const [activeLayers, setActiveLayers] = useState({
    tpis: true, canopy: false, lst: false,
    vulnerability: false, selected: true,
    water: false, anomaly: true, segmentation: false,
  })
  const [weights, setWeights] = useState({
    canopy_deficit: 0.25, thermal_stress: 0.25,
    vulnerability: 0.25, plantability: 0.15, roi_norm: 0.10,
  })
  const [budget, setBudget] = useState(1000)

  useEffect(() => {
    async function init() {
      const online = await healthCheck()
      setApiOnline(online)
      try {
        if (online) {
          const data = await fetchZones()
          setZones(data)
          // Check if API returned real or demo data
          setIsRealData(data.is_real_data === true)
          if (data.warning) {
            setDataWarning(data.warning)
          }
        } else {
          setZones(generateDemoZones())
          setIsRealData(false)
          setDataWarning('API offline - showing demo data')
        }
      } catch {
        setZones(generateDemoZones())
        setIsRealData(false)
        setDataWarning('Failed to connect - showing demo data')
      }
      setLoading(false)
    }
    init()
  }, [])

  const handleReoptimize = useCallback(async (newWeights, newBudget) => {
    const w = newWeights || weights
    const b = newBudget || budget
    setWeights(w)
    setBudget(b)
    try {
      if (apiOnline) {
        const result = await reoptimize(w, b)
        if (result.zones) setZones(result.zones)
      } else {
        // Client-side re-ranking
        const z = { ...zones }
        z.features.forEach(f => {
          const p = f.properties
          p.tpis = +(
            w.canopy_deficit * (p.canopy_deficit / 0.3) +
            w.thermal_stress * p.thermal_stress +
            w.vulnerability * p.vuln_score +
            w.plantability * 0.5 +
            w.roi_norm * 0.5
          ).toFixed(3)
        })
        z.features.sort((a, b) => b.properties.tpis - a.properties.tpis)
        let total = 0
        z.features.forEach((f, i) => {
          f.properties.final_rank = i + 1
          if (total + f.properties.trees_possible <= b && f.properties.water_feasible) {
            f.properties.selected = true
            f.properties.status = '✅ GO'
            total += f.properties.trees_possible
          } else {
            f.properties.selected = false
            f.properties.status = f.properties.water_feasible ? '❌ Budget' : '⚠️ H₂O'
          }
        })
        setZones({ ...z })
      }
    } catch (e) {
      console.error('Reoptimize failed:', e)
    }
  }, [zones, weights, budget, apiOnline])

  const toggleLayer = useCallback((layerId) => {
    setActiveLayers(prev => ({ ...prev, [layerId]: !prev[layerId] }))
  }, [])

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-logo">
          <span className="logo-icon">🌳</span>
          <h1>TEORA <span>v3.0</span></h1>
          <p>Loading geospatial intelligence...</p>
          <div className="loading-bar"><div className="loading-bar-fill" /></div>
        </div>
      </div>
    )
  }

  const features = zones?.features || []
  const selectedFeatures = features.filter(f => f.properties.selected)
  const totalTrees = selectedFeatures.reduce((s, f) => s + f.properties.trees_possible, 0)
  const avgTpis = features.length ? (features.reduce((s, f) => s + f.properties.tpis, 0) / features.length) : 0

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="header-left">
          <span className="logo-icon-sm">🌳</span>
          <h1 className="header-title">TEORA <span className="version">v3.0</span></h1>
          <span className="header-divider" />
          <span className="header-subtitle">Tree Equity Dashboard</span>
        </div>
        <div className="header-stats">
          <div className="stat-chip">
            <span className="stat-label">Zones</span>
            <span className="stat-value">{features.length}</span>
          </div>
          <div className="stat-chip stat-chip--green">
            <span className="stat-label">Selected</span>
            <span className="stat-value">{selectedFeatures.length}</span>
          </div>
          <div className="stat-chip stat-chip--cyan">
            <span className="stat-label">Trees</span>
            <span className="stat-value">{totalTrees.toLocaleString()}</span>
          </div>
          <div className="stat-chip stat-chip--purple">
            <span className="stat-label">Avg TPIS</span>
            <span className="stat-value">{avgTpis.toFixed(3)}</span>
          </div>
          <div className={`api-indicator ${apiOnline ? 'online' : 'offline'}`}>
            <span className="api-dot" />
            {apiOnline ? 'API Online' : 'Demo Mode'}
          </div>
        </div>
      </header>

      <div className="app-body">
        <Sidebar
          zones={features}
          weights={weights}
          budget={budget}
          activeLayers={activeLayers}
          onToggleLayer={toggleLayer}
          onWeightChange={handleReoptimize}
          onBudgetChange={(b) => handleReoptimize(weights, b)}
          onSelectZone={setSelectedZone}
          selectedZone={selectedZone}
        />

        <main className="map-container">
          <DataSourceWarning isRealData={isRealData} warning={dataWarning} />
          <MapView
            zones={zones}
            activeLayers={activeLayers}
            selectedZone={selectedZone}
            onSelectZone={setSelectedZone}
          />
        </main>

        {selectedZone && (
          <ZoneDetail
            zone={selectedZone}
            onClose={() => setSelectedZone(null)}
          />
        )}
      </div>
    </div>
  )
}
