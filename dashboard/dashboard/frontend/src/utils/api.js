const API_BASE = 'http://localhost:8000';

export async function analyzeAOI(aoi, weights, budget) {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ aoi, weights, budget }),
  });
  if (!res.ok) throw new Error(`Analyze failed: ${res.status}`);
  return res.json();
}

export async function fetchZones() {
  const res = await fetch(`${API_BASE}/zones`);
  if (!res.ok) throw new Error(`Fetch zones failed: ${res.status}`);
  return res.json();
}

export async function fetchZoneDetail(zoneId) {
  const res = await fetch(`${API_BASE}/zones/${zoneId}`);
  if (!res.ok) throw new Error(`Zone detail failed: ${res.status}`);
  return res.json();
}

export async function fetchMapLayers() {
  const res = await fetch(`${API_BASE}/map-layers`);
  if (!res.ok) throw new Error(`Map layers failed: ${res.status}`);
  return res.json();
}

export async function reoptimize(weights, budget, anomalyContamination) {
  const res = await fetch(`${API_BASE}/reoptimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      weights,
      budget,
      anomaly_contamination: anomalyContamination,
    }),
  });
  if (!res.ok) throw new Error(`Reoptimize failed: ${res.status}`);
  return res.json();
}

export async function fetchPareto() {
  const res = await fetch(`${API_BASE}/pareto`);
  if (!res.ok) throw new Error(`Pareto failed: ${res.status}`);
  return res.json();
}

export async function exportGeoJSON() {
  const res = await fetch(`${API_BASE}/export/geojson`);
  if (!res.ok) throw new Error(`Export failed: ${res.status}`);
  return res.json();
}

export async function exportCSV() {
  const res = await fetch(`${API_BASE}/export/csv`);
  if (!res.ok) throw new Error(`Export CSV failed: ${res.status}`);
  return res.json();
}

export async function healthCheck() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
