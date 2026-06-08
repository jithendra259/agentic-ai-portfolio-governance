export const PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];

export function getPieShare(value, total) {
  if (!total) return 0;
  return (value / total) * 100;
}
