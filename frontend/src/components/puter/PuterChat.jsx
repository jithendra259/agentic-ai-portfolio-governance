import React, { useState } from 'react';
import PuterModelSelector from './PuterModelSelector';
import { chat } from '../../lib/puterClient';

export default function PuterChat({ defaultModel = 'qwen/qwen3.7-plus' }) {
  const [model, setModel] = useState(defaultModel);
  const [prompt, setPrompt] = useState('Hello, say something friendly');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  async function handleSend() {
    setLoading(true);
    setResponse(null);
    setError(null);
    try {
      const res = await chat(prompt, { model });
      // puter.ai.chat may return text or an object; handle both
      const text = typeof res === 'string' ? res : res?.text || JSON.stringify(res);
      setResponse(text);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ border: '1px solid #ddd', padding: 12, borderRadius: 6, maxWidth: 700 }}>
      <PuterModelSelector value={model} onChange={setModel} />

      <div style={{ marginBottom: 8 }}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          style={{ width: '100%', fontFamily: 'inherit' }}
        />
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <button onClick={handleSend} disabled={loading}>
          {loading ? 'Sending…' : 'Send'}
        </button>
        <div style={{ color: '#666' }}>{loading ? 'Waiting for model…' : ''}</div>
      </div>

      <div>
        <strong>Response</strong>
        <div style={{ whiteSpace: 'pre-wrap', padding: 8, background: '#fafafa', borderRadius: 4, minHeight: 48 }}>
          {error ? <span style={{ color: 'crimson' }}>{error}</span> : response}
        </div>
      </div>
    </div>
  );
}
