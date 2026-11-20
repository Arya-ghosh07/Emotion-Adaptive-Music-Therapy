import { useCallback, useEffect, useRef, useState } from 'react';

// One shared webcam stream for both face analysis and heart-rate sampling.
export default function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [on, setOn] = useState(false);
  const [error, setError] = useState(null);

  const start = useCallback(async () => {
    if (streamRef.current) return true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setOn(true);
      setError(null);
      return true;
    } catch (e) {
      setError('Camera access was denied. Allow it in the browser to measure heart rate.');
      return false;
    }
  }, []);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setOn(false);
  }, []);

  useEffect(() => stop, [stop]);

  return { videoRef, on, error, start, stop };
}
