import './RankedTable.css'

function getAnomalyBadge(tag) {
  if (!tag) return null
  if (tag.includes('URGENT')) return <span className="badge badge-danger">⚠ URGENT</span>
  if (tag.includes('HEAT')) return <span className="badge badge-warning">🌡 HEAT</span>
  if (tag.includes('LOSS')) return <span className="badge badge-warning">📉 LOSS</span>
  return <span className="badge badge-info">{tag}</span>
}

function getStatusBadge(status) {
  if (status?.includes('GO')) return <span className="badge badge-success">✅ GO</span>
  if (status?.includes('H₂O')) return <span className="badge badge-warning">⚠️ H₂O</span>
  return <span className="badge badge-info">{status || '—'}</span>
}

export default function RankedTable({ zones, selectedZone, onSelectZone }) {
  const sorted = [...zones].sort((a, b) => a.properties.final_rank - b.properties.final_rank)

  return (
    <div className="ranked-table-wrapper">
      <h3 className="panel-title">Intervention Priority</h3>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Zone</th>
              <th>TPIS</th>
              <th>LST</th>
              <th>Can%</th>
              <th>Trees</th>
              <th>Tag</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(zone => {
              const p = zone.properties
              const isActive = selectedZone?.properties?.zone_id === p.zone_id
              return (
                <tr
                  key={p.zone_id}
                  className={`${p.selected ? 'selected' : ''} ${isActive ? 'active-row' : ''}`}
                  onClick={() => onSelectZone(zone)}
                >
                  <td className="rank-cell">
                    <span className={`rank-badge ${p.final_rank <= 3 ? 'top-3' : ''}`}>
                      {p.final_rank}
                    </span>
                  </td>
                  <td className="zone-name-cell">{p.name}</td>
                  <td className="mono-cell">
                    <span className="tpis-bar" style={{ width: `${p.tpis * 100}%` }} />
                    {p.tpis.toFixed(2)}
                  </td>
                  <td className="mono-cell">{p.mean_lst}°</td>
                  <td className="mono-cell">{p.canopy_pct}%</td>
                  <td className="mono-cell">{p.trees_possible}</td>
                  <td>{getAnomalyBadge(p.anomaly_tag)}</td>
                  <td>{getStatusBadge(p.status)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
