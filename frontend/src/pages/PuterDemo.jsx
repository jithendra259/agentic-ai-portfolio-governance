import React from 'react';
import PuterChat from '../components/puter/PuterChat';

export default function PuterDemo() {
  return (
    <div style={{ padding: 20 }}>
      <h2>Puter Demo</h2>
      <p>Uses Puter.js (CDN) to call Qwen models from the browser.</p>
      <PuterChat />
    </div>
  );
}
