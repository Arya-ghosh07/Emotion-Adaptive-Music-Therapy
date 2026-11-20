import { useCallback, useRef, useState } from 'react';

// Browser speech-to-text for the spoken check-in (Chrome, Edge, Safari).
export default function useSpeech(onTranscript) {
  const Recognition = typeof window !== 'undefined'
    && (window.SpeechRecognition || window.webkitSpeechRecognition);
  const [listening, setListening] = useState(false);
  const [error, setError] = useState(null);
  const recRef = useRef(null);
  const cb = useRef(onTranscript);
  cb.current = onTranscript;

  const listen = useCallback(() => {
    if (!Recognition) return;
    setError(null);
    const rec = new Recognition();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = (e) => cb.current?.(e.results[0][0].transcript);
    rec.onerror = (e) => setError(e.error === 'no-speech' ? "Didn't catch that - try again." : `Mic error: ${e.error}`);
    rec.onend = () => setListening(false);
    recRef.current = rec;
    setListening(true);
    rec.start();
  }, [Recognition]);

  const stop = useCallback(() => recRef.current?.stop(), []);

  return { supported: !!Recognition, listening, error, listen, stop };
}
