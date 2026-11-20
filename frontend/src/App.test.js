import { render, screen } from '@testing-library/react';
import { parseHeartRate } from './hooks/useHeartRateBLE';
import { skinRegions } from './hooks/useRppgSampler';
import App from './App';

// Audio and canvas charts need real browser APIs that jsdom doesn't have.
jest.mock('tone', () => ({}));
jest.mock('./audio/musicEngine', () => ({ musicEngine: { init: jest.fn(), setParams: jest.fn(), stop: jest.fn() } }));
jest.mock('react-chartjs-2', () => ({ Line: () => null, Scatter: () => null }));
jest.mock('./api', () => ({
  api: {
    health: () => Promise.resolve({ status: 'ok', face_model: true }),
    insights: () => Promise.resolve(null),
  },
}));

test('renders the start screen with the three session modes', async () => {
  render(<App />);
  expect(await screen.findByText(/Backend online/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /start session/i })).toBeInTheDocument();
  expect(screen.getByText('Adaptive')).toBeInTheDocument();
  expect(screen.getByText(/Fixed \(control\)/)).toBeInTheDocument();
  expect(screen.getByText(/Silence \(control\)/)).toBeInTheDocument();
});

test('parses Bluetooth heart-rate packets with RR intervals', () => {
  // flags 0x10 (8-bit HR, RR present), HR 72, RR 1024 (=1000 ms), 819 (~800 ms)
  const bytes = new Uint8Array([0x10, 72, 0x00, 0x04, 0x33, 0x03]);
  const { bpm, rr } = parseHeartRate(new DataView(bytes.buffer));
  expect(bpm).toBe(72);
  expect(rr[0]).toBeCloseTo(1000);
  expect(rr[1]).toBeCloseTo(799.8, 0);
});

test('skin regions sit inside the face box', () => {
  const regions = skinRegions([0.2, 0.1, 0.4, 0.6]);
  expect(regions).toHaveLength(3);
  regions.forEach(([x, y, w, h]) => {
    expect(x).toBeGreaterThanOrEqual(0.2);
    expect(x + w).toBeLessThanOrEqual(0.6);
    expect(y + h).toBeLessThanOrEqual(0.7);
  });
});
