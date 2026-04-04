import './DataSourceWarning.css'

export default function DataSourceWarning({ isRealData, warning }) {
  if (isRealData) return null

  return (
    <div className="data-source-warning">
      <span className="warning-icon">⚠️</span>
      <div className="warning-content">
        <strong>Demo Data Mode</strong>
        <p>{warning || 'Showing synthetic test data. Run the pipeline for real analysis.'}</p>
      </div>
    </div>
  )
}
