import axios from 'axios';
import { API_BASE_URL } from './Constants';

const http = axios.create({ baseURL: API_BASE_URL, timeout: 15000 });
const data = (p) => p.then((r) => r.data);

export const api = {
  health: () => data(http.get('/health')),
  start: (user_id, mode, music_minutes) => data(http.post('/session/start', { user_id, mode, music_minutes })),
  begin: (id) => data(http.post(`/session/${id}/begin`)),
  status: (id) => data(http.get(`/session/${id}/status`)),
  rppg: (id, samples) => data(http.post(`/session/${id}/rppg`, { samples })),
  hr: (id, bpm, rr_ms) => data(http.post(`/session/${id}/hr`, { bpm, rr_ms })),
  face: (id, image) => data(http.post(`/session/${id}/face`, { image })),
  text: (id, text) => data(http.post(`/session/${id}/text`, { text }, { timeout: 120000 })),
  selfReport: (id, valence, arousal, moment) =>
    data(http.post(`/session/${id}/self_report`, { valence, arousal, moment })),
  feedback: (id, thumbs) => data(http.post(`/session/${id}/feedback`, { thumbs })),
  end: (id) => data(http.post(`/session/${id}/end`)),
  insights: (user) => data(http.get(`/users/${encodeURIComponent(user)}/insights`)),
};
