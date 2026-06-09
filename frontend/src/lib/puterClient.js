// Lightweight Puter client loader using the CDN script
// Usage:
// import { chat } from '../lib/puterClient'
// const res = await chat('Hello', { model: 'qwen/qwen3.7-plus' })

const PUTER_CDN = 'https://unpkg.com/@heyputer/puter.js/dist/puter.umd.js';
let _puter = null;
let _loading = null;

function _loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[data-puter]`)) {
      resolve();
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.setAttribute('data-puter', '1');
    s.onload = () => resolve();
    s.onerror = (e) => reject(new Error('Failed to load Puter script'));
    document.head.appendChild(s);
  });
}

export async function initPuter() {
  if (_puter) return _puter;
  if (_loading) return _loading;
  _loading = (async () => {
    if (typeof window === 'undefined') throw new Error('Puter client only available in browser');
    await _loadScript(PUTER_CDN);
    if (!window.puter) throw new Error('Puter script loaded but `window.puter` not found');
    _puter = window.puter;
    return _puter;
  })();
  return _loading;
}

export async function chat(prompt, opts = {}) {
  const put = await initPuter();
  // puter.ai.chat returns a Promise resolving to the model response
  return put.ai.chat(prompt, opts);
}

export async function txt2img(prompt, opts = {}) {
  const put = await initPuter();
  return put.ai.txt2img(prompt, opts);
}
