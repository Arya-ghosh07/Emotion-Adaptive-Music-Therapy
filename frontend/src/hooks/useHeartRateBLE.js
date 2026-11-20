import { useCallback, useRef, useState } from 'react';

// Bluetooth heart-rate strap (any device with the standard 0x180D profile,
// e.g. Polar H10, Garmin HRM, Wahoo TICKR). Works in Chrome / Edge.

export function parseHeartRate(dataView) {
  const flags = dataView.getUint8(0);
  let offset = 1;
  let bpm;
  if (flags & 0x01) { bpm = dataView.getUint16(offset, true); offset += 2; }
  else { bpm = dataView.getUint8(offset); offset += 1; }
  if (flags & 0x08) offset += 2; // energy expended
  const rr = [];
  if (flags & 0x10) {
    for (; offset + 1 < dataView.byteLength; offset += 2) {
      rr.push((dataView.getUint16(offset, true) / 1024) * 1000); // 1/1024 s -> ms
    }
  }
  return { bpm, rr };
}

export default function useHeartRateBLE(onReading) {
  const [device, setDevice] = useState(null);
  const [error, setError] = useState(null);
  const cb = useRef(onReading);
  cb.current = onReading;
  const supported = typeof navigator !== 'undefined' && !!navigator.bluetooth;

  const connect = useCallback(async () => {
    setError(null);
    try {
      const dev = await navigator.bluetooth.requestDevice({ filters: [{ services: ['heart_rate'] }] });
      const server = await dev.gatt.connect();
      const service = await server.getPrimaryService('heart_rate');
      const ch = await service.getCharacteristic('heart_rate_measurement');
      ch.addEventListener('characteristicvaluechanged', (e) => cb.current?.(parseHeartRate(e.target.value)));
      await ch.startNotifications();
      dev.addEventListener('gattserverdisconnected', () => setDevice(null));
      setDevice(dev.name || 'Heart-rate strap');
    } catch (e) {
      if (e?.name !== 'NotFoundError') setError(e?.message || 'Could not connect');
    }
  }, []);

  return { supported, device, error, connect };
}
