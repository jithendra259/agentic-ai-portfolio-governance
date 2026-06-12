const DEFAULT_BACKEND_BASE = 'https://agentic-ai-portfolio-governance.onrender.com';
const VERCEL_BACKEND_BASE = '/_/backend';

export const BACKEND_BASE = (
  import.meta.env?.DEV ? DEFAULT_BACKEND_BASE : VERCEL_BACKEND_BASE
).replace(/\/$/, '');
