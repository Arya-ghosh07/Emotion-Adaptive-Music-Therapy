import React from 'react';

// Self-Assessment Manikin style sliders (1-9). These are the ground truth the
// system learns from: each sensor's trust is updated against them.
const Slider = ({ label, left, right, value, onChange }) => (
  <div className="sam-row">
    <div className="sam-label">{label}</div>
    <input type="range" min="1" max="9" step="1" value={value}
      aria-label={label} onChange={(e) => onChange(parseInt(e.target.value, 10))} />
    <div className="sam-ends"><span>{left}</span><span>{right}</span></div>
  </div>
);

export default function SelfReport({ value, onChange, prompt }) {
  return (
    <div className="self-report">
      {prompt && <p className="muted">{prompt}</p>}
      <Slider label="How pleasant do you feel?" left="Very unpleasant" right="Very pleasant"
        value={value.valence} onChange={(v) => onChange({ ...value, valence: v })} />
      <Slider label="How calm or activated?" left="Very calm" right="Very activated"
        value={value.arousal} onChange={(a) => onChange({ ...value, arousal: a })} />
    </div>
  );
}
