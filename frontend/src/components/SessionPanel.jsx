import React, { useState } from 'react';
import { Mic, Play, Square } from 'lucide-react';
import SelfReport from './SelfReport';
import useSpeech from '../hooks/useSpeech';
import { MODE_INFO } from '../Constants';

const fmt = (s) => (s == null ? '' : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`);

export default function SessionPanel({
  userId, setUserId, status, summary, busy, error,
  onStart, onCheckinText, onBegin, onFinish, onEndEarly, onReset,
}) {
  const [mode, setMode] = useState('adaptive');
  const [minutes, setMinutes] = useState(9);
  const [report, setReport] = useState({ valence: 5, arousal: 5 });
  const [typed, setTyped] = useState('');
  const speech = useSpeech((t) => { setTyped(t); onCheckinText(t); });
  const phase = summary ? 'summary' : status?.phase || 'idle';

  return (
    <div className="card session">
      <div className="card-title">Session</div>
      {error && <div className="error">{error}</div>}

      {phase === 'idle' && (
        <>
          <label className="field">
            <span className="muted small">Your name (used to keep your personal model)</span>
            <input value={userId} onChange={(e) => setUserId(e.target.value.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 32))} />
          </label>
          <div className="modes">
            {Object.entries(MODE_INFO).map(([m, info]) => (
              <button key={m} className={`mode ${mode === m ? 'selected' : ''}`} onClick={() => setMode(m)}>
                <b>{info.label}</b><span className="muted small">{info.text}</span>
              </button>
            ))}
          </div>
          <label className="field inline">
            <span className="muted small">Music length</span>
            <select value={minutes} onChange={(e) => setMinutes(Number(e.target.value))}>
              {[4.5, 6, 9, 12, 15].map((m) => <option key={m} value={m}>{m} min</option>)}
            </select>
          </label>
          <button className="btn primary" disabled={!userId || busy} onClick={() => onStart(mode, minutes)}>
            <Play size={16} /> Start session
          </button>
          <p className="muted small">
            Uses your webcam to estimate heart rate. Video stays in your browser except one small frame
            every 3 s for expression analysis, which is processed in memory and never stored.
            A wellness experiment, not a medical device.
          </p>
        </>
      )}

      {phase === 'checkin' && (
        <>
          <p><b>How are you feeling right now?</b> Say it or type it.</p>
          <div className="checkin">
            {speech.supported && (
              <button className={`btn ghost ${speech.listening ? 'active' : ''}`} onClick={speech.listening ? speech.stop : speech.listen}>
                <Mic size={16} /> {speech.listening ? 'Listening...' : 'Speak'}
              </button>
            )}
            <input placeholder="e.g. a bit stressed about tomorrow" value={typed}
              onChange={(e) => setTyped(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && typed.trim() && onCheckinText(typed.trim())} />
            <button className="btn ghost" disabled={!typed.trim()} onClick={() => onCheckinText(typed.trim())}>Send</button>
          </div>
          {speech.error && <div className="muted small">{speech.error}</div>}
          {status?.text && (
            <div className="muted small">
              Heard: <i>{status.text.dominant_emotion}</i> (via {status.text.method})
            </div>
          )}
          <SelfReport value={report} onChange={setReport} />
          <button className="btn primary" disabled={busy} onClick={() => onBegin(report)}>Continue</button>
        </>
      )}

      {phase === 'baseline' && (
        <div className="center">
          <div className="big-timer">{fmt(status.phase_remaining_s)}</div>
          <p>Sit comfortably, face the camera and breathe normally.</p>
          <p className="muted small">Measuring your resting heart rate in silence.</p>
        </div>
      )}

      {phase === 'music' && (
        <div className="center">
          <div className="big-timer">{fmt(status.phase_remaining_s)}</div>
          <p>Relax and listen. Keep your face in view.</p>
          <p className="muted small">Mode: {MODE_INFO[status.mode].label}</p>
          <button className="btn ghost small" onClick={onEndEarly}><Square size={14} /> End early</button>
        </div>
      )}

      {phase === 'report' && (
        <>
          <p><b>Session complete.</b> How do you feel now?</p>
          <SelfReport value={report} onChange={setReport} />
          <button className="btn primary" disabled={busy} onClick={() => onFinish(report)}>Finish</button>
        </>
      )}

      {phase === 'summary' && (
        <>
          <p><b>Done.</b> {summary.hr_change != null
            ? `Your heart rate changed by ${summary.hr_change > 0 ? '+' : ''}${summary.hr_change} BPM during the music.`
            : 'Not enough clean heart-rate signal to measure a change this time.'}</p>
          <table className="learn-table">
            <thead><tr><th>#</th><th>Style</th><th>HR</th><th>Score</th></tr></thead>
            <tbody>
              {summary.segments.map((s) => (
                <tr key={s.idx}>
                  <td>{s.idx + 1}</td>
                  <td>{s.preset ? s.preset.replace('_', ' ') : 'silence'}</td>
                  <td>{s.hr_start != null ? `${s.hr_start} → ${s.hr_end}` : '–'}</td>
                  <td>{s.valid ? s.reward.toFixed(2) : <span className="muted">{s.basis.replace(/_/g, ' ')}</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <button className="btn primary" onClick={onReset}>New session</button>
        </>
      )}
    </div>
  );
}
