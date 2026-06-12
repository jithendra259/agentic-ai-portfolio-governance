const DEFAULT_BACKEND_BASE = 'http://127.0.0.1:8000';
const VERCEL_BACKEND_BASE = '/_/backend';

export const BACKEND_BASE = (
  import.meta.env?.DEV ? DEFAULT_BACKEND_BASE : VERCEL_BACKEND_BASE
).replace(/\/$/, '');
