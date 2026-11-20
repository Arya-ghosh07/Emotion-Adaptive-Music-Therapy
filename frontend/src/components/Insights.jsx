import React from 'react';
import { Sparkles } from 'lucide-react';
import { CONTEXT_LABEL } from '../Constants';

// What the system has learned about this user, from real sessions only.
export default function Insights({ insights }) {
  if (!insights) return null;
  const { learning, presets, sessions, hr_change_by_mode: byMode, learned_segments: n } = insights;

  return (
    <div className="card">
      <div className="card-title"><Sparkles size={14} /> What it has learned about you</div>
      {!n && <p className="muted">Nothing yet. Each 90 s adaptive segment with a clean heart-rate signal teaches it something.</p>}
      {!!n && learning && (
        <>
          <p className="muted small">Calming score per style (0.5 = no effect; higher = your heart rate dropped more). {n} scored segments.</p>
          <table className="learn-table">
            <thead>
              <tr><th />{Object.keys(learning).map((c) => <th key={c}>{CONTEXT_LABEL[c]}</th>)}</tr>
            </thead>
            <tbody>
              {Object.entries(presets).map(([id, name]) => (
                <tr key={id}>
                  <td>{name}</td>
                  {Object.entries(learning).map(([ctx, l]) => {
                    const s = l.scores[id];
                    const best = l.best === id;
                    return (
                      <td key={ctx} className={best ? 'best' : ''}>
                        {s.pulls ? `${s.mean.toFixed(2)} (${s.pulls})` : '–'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {byMode && Object.keys(byMode).length > 0 && (
        <div className="by-mode">
          {Object.entries(byMode).map(([m, v]) => (
            <div key={m} className="mode-stat">
              <div className="muted small">{m}</div>
              <b>{v.median > 0 ? '+' : ''}{v.median} BPM</b>
              <div className="muted small">median of {v.n}</div>
            </div>
          ))}
        </div>
      )}
      {sessions?.length > 0 && (
        <details>
          <summary className="small">Past sessions ({sessions.length})</summary>
          <table className="learn-table">
            <thead><tr><th>Date</th><th>Mode</th><th>Baseline</th><th>HR change</th></tr></thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id}>
                  <td>{new Date(s.started_at * 1000).toLocaleString()}</td>
                  <td>{s.mode}</td>
                  <td>{s.baseline_bpm ? `${Math.round(s.baseline_bpm)}` : '–'}</td>
                  <td>{s.hr_change != null ? `${s.hr_change > 0 ? '+' : ''}${s.hr_change}` : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}
