import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, Camera } from 'lucide-react';

import { api } from './api';
import { musicEngine } from './audio/musicEngine';
import useCamera from './hooks/useCamera';
import useRppgSampler, { skinRegions } from './hooks/useRppgSampler';
import useHeartRateBLE from './hooks/useHeartRateBLE';

import SessionPanel from './components/SessionPanel';
import HeartRateCard from './components/HeartRateCard';
import EmotionChart from './components/EmotionChart';
import MusicDashboard from './components/MusicDashboard';
import Insights from './components/Insights';
import './App.css';

const loadName = () => { try { return localStorage.getItem('eamta_user') || ''; } catch { return ''; } };
const saveName = (n) => { try { localStorage.setItem('eamta_user', n); } catch { /* private mode */ } };

function App() {
  const [userId, setUserIdState] = useState(loadName);
  const [health, setHealth] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [status, setStatus] = useState(null);
  const [summary, setSummary] = useState(null);
  const [insights, setInsights] = useState(null);
  const [face, setFace] = useState(null);
  const [trail, setTrail] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const camera = useCamera();
  const lastMusicKey = useRef('');

  const setUserId = (n) => { setUserIdState(n); saveName(n); };

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: 'down' }));
  }, []);

  const refreshInsights = useCallback(() => {
    if (userId) api.insights(userId).then(setInsights).catch(() => {});
  }, [userId]);
  useEffect(() => { if (!sessionId) refreshInsights(); }, [sessionId, refreshInsights]);

  const live = !!sessionId && status?.phase !== 'done';
  useRppgSampler({
    videoRef: camera.videoRef,
    sessionId,
    active: live && camera.on && status?.heart_source !== 'ble',
    sendFace: live && camera.on,
    onFace: setFace,
  });

  const ble = useHeartRateBLE(({ bpm, rr }) => {
    if (sessionId) api.hr(sessionId, bpm, rr).catch(() => {});
  });

  // poll session status
  useEffect(() => {
    if (!sessionId) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await api.status(sessionId);
        if (cancelled) return;
        setStatus(s);
        if (s.fused) setTrail((t) => [...t.slice(-30), { x: s.fused.valence, y: s.fused.arousal }]);
      } catch (e) {
        if (!cancelled && e.response?.status === 404) { setSessionId(null); setStatus(null); }
      }
    };
    poll();
    const id = setInterval(poll, 1000);
    return () => { cancelled = true; clearInterval(id); };
  }, [sessionId]);

  // drive the music engine from the backend's parameters
  useEffect(() => {
    const m = status?.music || null;
    const key = m ? `${m.preset}|${m.tempo_bpm}|${m.scale}|${m.velocity}|${m.density}` : 'none';
    if (key === lastMusicKey.current) return;
    lastMusicKey.current = key;
    musicEngine.setParams(m);
  }, [status?.music]);

  const run = async (fn) => {
    setBusy(true); setError(null);
    try { await fn(); } catch (e) {
      setError(e.response?.data?.error || (e.request ? 'Cannot reach the backend on port 5001. Is it running?' : e.message));
    }
    setBusy(false);
  };

  const start = (mode, minutes) => run(async () => {
    await musicEngine.init(); // needs this click to unlock audio
    await camera.start(); // if denied, a Bluetooth strap or thumbs still work
    const s = await api.start(userId, mode, minutes);
    setSummary(null); setTrail([]); setFace(null);
    setStatus(s); setSessionId(s.id);
  });

  const checkinText = (text) => run(async () => {
    await api.text(sessionId, text);
    setStatus(await api.status(sessionId));
  });

  const begin = (report) => run(async () => {
    await api.selfReport(sessionId, report.valence, report.arousal, 'start');
    setStatus(await api.begin(sessionId));
  });

  const finish = (report) => run(async () => {
    await api.selfReport(sessionId, report.valence, report.arousal, 'end');
    const sum = await api.end(sessionId);
    musicEngine.stop(); camera.stop();
    setSummary(sum); setSessionId(null);
    refreshInsights();
  });

  const endEarly = () => run(async () => {
    const sum = await api.end(sessionId);
    musicEngine.stop(); camera.stop();
    setSummary(sum); setSessionId(null);
    refreshInsights();
  });

  const reset = () => { setSummary(null); setStatus(null); };
  const thumbs = (v) => sessionId && api.feedback(sessionId, v).catch(() => {});

  const box = face?.face_found ? face.box : null;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="logo"><Activity size={22} /></div>
          <div>
            <h1>EAMTA</h1>
            <p className="muted small">Music that learns what calms <i>you</i>, from your heart rate</p>
          </div>
        </div>
        <div className={`pill ${health?.status === 'ok' ? 'ok' : 'bad'}`}>
          {health == null ? 'Connecting...' : health.status === 'ok' ? `Backend online${health.face_model ? '' : ' · face model missing'}` : 'Backend offline'}
        </div>
      </header>

      <main className="grid">
        <div className="col">
          <div className="card">
            <div className="card-title"><Camera size={14} /> Camera</div>
            <div className="video-wrap">
              <video ref={camera.videoRef} autoPlay muted playsInline className={camera.on ? '' : 'hidden'} />
              {camera.on && (box ? skinRegions(box) : skinRegions([0.33, 0.18, 0.34, 0.5])).map(([x, y, w, h], i) => (
                <div key={i} className={`roi ${box ? 'locked' : ''}`}
                  style={{ left: `${(1 - x - w) * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` }} />
              ))}
              {!camera.on && <div className="video-placeholder muted">Camera starts with a session</div>}
            </div>
            {camera.on && !box && <div className="muted small">Centre your face in the guide boxes.</div>}
            {camera.error && <div className="error small">{camera.error}</div>}
          </div>
          <HeartRateCard heart={status?.heart} series={status?.heart_series} source={status?.heart_source}
            restingBpm={status?.resting_bpm ?? insights?.resting_bpm} ble={ble} phase={status?.phase} />
        </div>

        <div className="col">
          <SessionPanel userId={userId} setUserId={setUserId} status={sessionId ? status : null} summary={summary}
            busy={busy} error={error} onStart={start} onCheckinText={checkinText} onBegin={begin}
            onFinish={finish} onEndEarly={endEarly} onReset={reset} />
          {sessionId && <MusicDashboard status={status} onThumbs={thumbs} />}
          {sessionId && <EmotionChart fused={status?.fused} trail={trail} weights={status?.weights} face={face} />}
          {!sessionId && <Insights insights={insights} />}
        </div>
      </main>
    </div>
  );
}

export default App;
