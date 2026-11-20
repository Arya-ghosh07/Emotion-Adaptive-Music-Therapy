import React from 'react';
import { Scatter } from 'react-chartjs-2';
import { Chart as ChartJS, LinearScale, PointElement, LineElement, Tooltip } from 'chart.js';
import { Brain } from 'lucide-react';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip);

const SENSOR_NAME = { face: 'Face', text: 'Words', heart: 'Heart rate' };

function WeightBars({ title, weights }) {
  return (
    <div className="weights">
      <div className="muted small">{title}</div>
      {Object.entries(weights || {}).map(([m, w]) => (
        <div key={m} className="weight-row">
          <span>{SENSOR_NAME[m] || m}</span>
          <div className="bar"><div className="bar-fill" style={{ width: `${Math.round(w * 100)}%` }} /></div>
          <span className="small">{Math.round(w * 100)}%</span>
        </div>
      ))}
    </div>
  );
}

export default function EmotionChart({ fused, trail, weights, face }) {
  const data = {
    datasets: [
      { data: trail, backgroundColor: 'rgba(77,148,255,0.25)', pointRadius: 3 },
      { data: fused ? [{ x: fused.valence, y: fused.arousal }] : [], backgroundColor: '#4d94ff', pointRadius: 9 },
    ],
  };
  const axis = (text) => ({
    min: -1, max: 1,
    title: { display: true, text, color: '#888', font: { size: 11 } },
    grid: { color: (c) => (c.tick.value === 0 ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.05)') },
    ticks: { display: false },
  });
  const options = {
    animation: false, maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: { x: axis('Unpleasant ← → Pleasant'), y: axis('Calm ← → Activated') },
  };

  return (
    <div className="card">
      <div className="card-title"><Brain size={14} /> Estimated state</div>
      <div className="emotion-head">
        <span className="tag">{fused?.emotion_label || 'waiting for signals'}</span>
        {face && <span className="muted small">{face.face_found ? `Face: ${face.dominant_emotion}` : 'No face detected'}</span>}
      </div>
      <div style={{ height: 220 }}><Scatter data={data} options={options} /></div>
      <p className="muted small">
        Sensor trust is learned from your own slider answers: a sensor that keeps disagreeing with how
        you say you feel gets less say.
      </p>
      <div className="weights-grid">
        <WeightBars title="Trust for pleasantness" weights={weights?.valence} />
        <WeightBars title="Trust for activation" weights={weights?.arousal} />
      </div>
    </div>
  );
}
