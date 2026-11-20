import { useEffect, useRef } from 'react';
import { api } from '../api';

// Samples the average skin colour of forehead + cheeks on every video frame and
// sends [t_ms, r, g, b] to the backend once per second. Only these numbers leave
// the page for heart-rate estimation - never the video.
//
// It also sends one small JPEG every few seconds for facial expression analysis;
// the face box it returns is used to place the skin regions.

const SAMPLE_W = 160;
const FACE_EVERY_MS = 3000;
const FALLBACK_BOX = [0.33, 0.18, 0.34, 0.5]; // centre of frame: "put your face here"

export function skinRegions(box) {
  const [x, y, w, h] = box;
  return [
    [x + 0.3 * w, y + 0.08 * h, 0.4 * w, 0.17 * h], // forehead
    [x + 0.15 * w, y + 0.5 * h, 0.2 * w, 0.18 * h], // left cheek
    [x + 0.65 * w, y + 0.5 * h, 0.2 * w, 0.18 * h], // right cheek
  ];
}

export default function useRppgSampler({ videoRef, sessionId, active, sendFace, onHeart, onFace }) {
  const boxRef = useRef(null);
  const cbRef = useRef({ onHeart, onFace });
  cbRef.current = { onHeart, onFace };

  useEffect(() => {
    if (!active || !sessionId) return undefined;
    const video = videoRef.current;
    if (!video) return undefined;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const faceCanvas = document.createElement('canvas');
    let buffer = [];
    let stopped = false;
    let frameHandle = null;
    let lastFace = 0;
    let sending = false;

    const sample = () => {
      if (stopped) return;
      if (video.readyState >= 2 && video.videoWidth) {
        const sh = Math.round((SAMPLE_W * video.videoHeight) / video.videoWidth);
        if (canvas.width !== SAMPLE_W) { canvas.width = SAMPLE_W; canvas.height = sh; }
        ctx.drawImage(video, 0, 0, SAMPLE_W, sh);
        let r = 0, g = 0, b = 0, n = 0;
        for (const [rx, ry, rw, rh] of skinRegions(boxRef.current?.box || FALLBACK_BOX)) {
          const px = Math.max(0, Math.round(rx * SAMPLE_W));
          const py = Math.max(0, Math.round(ry * sh));
          const pw = Math.max(1, Math.min(SAMPLE_W - px, Math.round(rw * SAMPLE_W)));
          const ph = Math.max(1, Math.min(sh - py, Math.round(rh * sh)));
          const d = ctx.getImageData(px, py, pw, ph).data;
          for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; n += 1; }
        }
        if (n) buffer.push([performance.timeOrigin + performance.now(), r / n, g / n, b / n]);
      }
      frameHandle = 'requestVideoFrameCallback' in video
        ? video.requestVideoFrameCallback(sample)
        : requestAnimationFrame(sample);
    };
    sample();

    const flush = setInterval(async () => {
      const now = Date.now();
      if (buffer.length && !sending) {
        const batch = buffer;
        buffer = [];
        sending = true;
        try {
          const res = await api.rppg(sessionId, batch);
          if (!stopped) cbRef.current.onHeart?.(res.heart);
        } catch (e) { /* transient network error: next batch continues */ }
        sending = false;
      }
      if (sendFace && now - lastFace > FACE_EVERY_MS && video.videoWidth) {
        lastFace = now;
        faceCanvas.width = 320;
        faceCanvas.height = Math.round((320 * video.videoHeight) / video.videoWidth);
        faceCanvas.getContext('2d').drawImage(video, 0, 0, faceCanvas.width, faceCanvas.height);
        api.face(sessionId, faceCanvas.toDataURL('image/jpeg', 0.7))
          .then((f) => {
            if (stopped) return;
            if (f.face_found) boxRef.current = { box: f.box, t: Date.now() };
            else if (boxRef.current && Date.now() - boxRef.current.t > 10000) boxRef.current = null;
            cbRef.current.onFace?.(f);
          })
          .catch(() => {});
      }
    }, 1000);

    return () => {
      stopped = true;
      clearInterval(flush);
      if (frameHandle && 'cancelVideoFrameCallback' in video) video.cancelVideoFrameCallback(frameHandle);
      else if (frameHandle) cancelAnimationFrame(frameHandle);
    };
  }, [active, sessionId, sendFace, videoRef]);

  return boxRef;
}
