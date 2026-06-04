const DEFAULT_BACKEND_BASE = 'http://127.0.0.1:8000';

export const BACKEND_BASE = (
  import.meta.env?.VITE_BACKEND_BASE || DEFAULT_BACKEND_BASE
).replace(/\/$/, '');
