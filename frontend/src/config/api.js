const LOCAL_BACKEND_BASE = 'http://127.0.0.1:8000';
const VERCEL_BACKEND_BASE = '/_/backend';

export const BACKEND_BASE = (
  import.meta.env?.PROD === true ? VERCEL_BACKEND_BASE : LOCAL_BACKEND_BASE
).replace(/\/$/, '');
