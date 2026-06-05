export const CHART_TIER = {
  FREE: 'free',
  PRO: 'pro',
  PREMIUM: 'premium',
};

export function getMuiPremiumChartsEnabled(options = {}) {
  if (options.premiumEnabled != null) return Boolean(options.premiumEnabled);
  const env = options.env || safeImportMetaEnv();
  const raw = env?.VITE_ENABLE_MUI_PREMIUM_CHARTS ?? env?.ENABLE_MUI_PREMIUM_CHARTS;
  if (raw == null || raw === '') return false;
  return ['1', 'true', 'yes', 'on'].includes(String(raw).trim().toLowerCase());
}

export function premiumUnavailableMessage(fallbackChart) {
  return fallbackChart
    ? `This chart requires MUI X Premium. Rendering fallback chart ${fallbackChart} instead.`
    : 'This chart requires MUI X Premium and no fallback chart is configured.';
}

function safeImportMetaEnv() {
  try {
    return import.meta?.env || {};
  } catch {
    return {};
  }
}
