import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, LinearScale, PointElement, LineElement, Filler, Tooltip,
} from 'chart.js';
import { Bluetooth, HeartPulse } from 'lucide-react';

ChartJS.register(LinearScale, PointElement, LineElement, Filler, Tooltip);

export default function HeartRateCard({ heart, series, source, restingBpm, ble, phase }) {
  const quality = heart?.quality ?? 0;
  const reliable = heart?.reliable;
  const points = (series || []).filter((p) => p[2]).map(([t, b]) => ({ x: t, y: b }));
  const warming = !heart && phase && phase !== 'done';

  const data = {
    datasets: [{
      data: points,
      borderColor: '#ff6b81',
      backgroundColor: 'rgba(255,107,129,0.12)',
      fill: true,
      pointRadius: 0,
      borderWidth: 2,
      tension: 0.3,
    }],
  };
  const options = {
    animation: false,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { type: 'linear', display: false },
      y: { suggestedMin: 55, suggestedMax: 100, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#777' } },
    },
  };

  return (
    <div className="card">
      <div className="card-title"><HeartPulse size={14} /> Heart rate</div>
      <div className="hr-row">
        <div>
          <div className={`hr-value ${reliable ? '' : 'dim'}`}>
            {heart?.bpm ? Math.round(heart.bpm) : '--'}<span className="unit"> BPM</span>
          </div>
          <div className="muted small">
            {warming && 'Measuring... keep your face still and well lit (about 10 s)'}
            {heart && !reliable && 'Weak signal - hold still, face the light'}
            {heart && reliable && `Source: ${source === 'ble' ? 'Bluetooth strap' : 'webcam (rPPG)'}`}
            {heart?.rmssd != null && ` · HRV ${Math.round(heart.rmssd)} ms`}
          </div>
        </div>
        <div className="quality">
          <div className="muted small">Signal quality</div>
          <div className="bar"><div className="bar-fill" style={{ width: `${Math.round(quality * 100)}%`, background: reliable ? '#2ecc71' : '#e67e22' }} /></div>
          <div className="muted small">Resting: {restingBpm ? Math.round(restingBpm) : '?'} BPM</div>
        </div>
      </div>
      <div style={{ height: 120 }}><Line data={data} options={options} /></div>
      {ble.supported && (
        <button className="btn ghost small" onClick={ble.connect} disabled={!!ble.device}>
          <Bluetooth size={14} /> {ble.device ? `Connected: ${ble.device}` : 'Connect Bluetooth HR strap (optional)'}
        </button>
      )}
      {ble.error && <div className="error small">{ble.error}</div>}
    </div>
  );
}
