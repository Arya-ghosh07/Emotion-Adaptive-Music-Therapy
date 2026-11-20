import React from 'react';
import { Music, ThumbsDown, ThumbsUp } from 'lucide-react';

const PHASE_TEXT = {
  match: 'Matching your current pace',
  guide: 'Gradually slowing down',
  target: 'Settled, calm texture',
  fixed: 'Fixed style (control)',
};

export default function MusicDashboard({ status, onThumbs }) {
  const music = status?.music;
  const seg = status?.segment;
  const last = status?.last_segment;

  return (
    <div className="card">
      <div className="card-title"><Music size={14} /> Now playing</div>
      {!music && <p className="muted">{seg?.name === 'Silence' ? 'Silence (control session)' : 'Music starts after the baseline.'}</p>}
      {music && (
        <>
          <div className="now-playing">
            <div className="np-name">{seg?.name}</div>
            <div className="muted small">{PHASE_TEXT[music.phase]}</div>
          </div>
          <div className="np-stats">
            <div><span className="muted small">Tempo</span><b>{Math.round(music.tempo_bpm)}</b></div>
            <div><span className="muted small">Mode</span><b>{music.scale.replace('_', ' ')}</b></div>
            <div><span className="muted small">Next style in</span><b>{seg ? `${Math.max(0, Math.round(seg.ends_in))}s` : '-'}</b></div>
          </div>
          <div className="thumbs">
            <span className="muted small">Is this helping?</span>
            <button className={`btn ghost small ${seg?.thumbs === 1 ? 'active' : ''}`} onClick={() => onThumbs(1)} aria-label="Helping"><ThumbsUp size={14} /></button>
            <button className={`btn ghost small ${seg?.thumbs === -1 ? 'active' : ''}`} onClick={() => onThumbs(-1)} aria-label="Not helping"><ThumbsDown size={14} /></button>
          </div>
        </>
      )}
      {last && (
        <div className="last-seg small">
          Last segment ({last.preset ? last.preset.replace('_', ' ') : 'silence'}):{' '}
          {last.valid
            ? <>HR {last.hr_start ?? '?'} → {last.hr_end ?? '?'} BPM · score {last.reward?.toFixed(2)} ({last.basis})</>
            : <span className="muted">not scored ({last.basis.replace(/_/g, ' ')})</span>}
        </div>
      )}
    </div>
  );
}
