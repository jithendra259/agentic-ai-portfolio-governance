import React from 'react';
import { PUTER_MODELS } from '../../lib/puterModels';

export default function PuterModelSelector({ value, onChange }) {
  return (
    <label style={{ display: 'block', marginBottom: 8 }}>
      Model:
      <select
        value={value}
        onChange={(e) => onChange && onChange(e.target.value)}
        style={{ marginLeft: 8 }}
      >
        {PUTER_MODELS.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    </label>
  );
}
