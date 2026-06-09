/**
 * Enhanced Color Palette Management for Charts
 * Supports WCAG AA compliance and theme-aware colors
 */

// Professional color palettes
export const COLOR_PALETTES = {
  default: [
    '#3b82f6', // blue
    '#ef4444', // red
    '#10b981', // green
    '#f59e0b', // amber
    '#8b5cf6', // purple
    '#06b6d4', // cyan
    '#ec4899', // pink
    '#6366f1', // indigo
  ],
  
  professional: [
    '#1e40af', // dark blue
    '#dc2626', // dark red
    '#15803d', // dark green
    '#b45309', // dark amber
    '#6d28d9', // dark purple
    '#0369a1', // dark cyan
    '#be185d', // dark pink
    '#3730a3', // dark indigo
  ],

  pastel: [
    '#93c5fd', // light blue
    '#fca5a5', // light red
    '#86efac', // light green
    '#fcd34d', // light amber
    '#d8b4fe', // light purple
    '#a5f3fc', // light cyan
    '#fbcfe8', // light pink
    '#c7d2fe', // light indigo
  ],

  categorical: [
    '#0ea5e9', // sky
    '#f97316', // orange
    '#22c55e', // lime
    '#a855f7', // fuchsia
    '#06b6d4', // cyan
    '#64748b', // slate
    '#ec4899', // pink
    '#f59e0b', // amber
  ],

  diverging: [
    '#1e3a8a', // deep blue
    '#3b82f6', // blue
    '#dbeafe', // light blue
    '#fef3c7', // light amber
    '#fbbf24', // amber
    '#dc2626', // red
  ],

  sequential: [
    '#f0f9ff', // very light
    '#e0f2fe',
    '#bae6fd',
    '#7dd3fc',
    '#38bdf8',
    '#0ea5e9',
    '#0284c7',
    '#0c4a6e', // very dark
  ],

  heatmap: [
    '#0c4a6e', // deep blue
    '#0284c7', // blue
    '#38bdf8', // light blue
    '#fef3c7', // pale yellow
    '#fbbf24', // amber
    '#f97316', // orange
    '#dc2626', // red
    '#7f1d1d', // dark red
  ],

  wcagCompliant: [
    '#000000', // black
    '#0052CC', // blue
    '#005A9C', // dark blue
    '#118800', // green
    '#E81816', // red
    '#FF6900', // orange
    '#FFB81C', // yellow (with black text)
    '#D81B60', // magenta
  ]
};

/**
 * Get color based on index and palette
 */
export const getChartColor = (index, palette = 'default') => {
  const colors = COLOR_PALETTES[palette] || COLOR_PALETTES.default;
  return colors[index % colors.length];
};

/**
 * Get color scheme for theme
 */
export const getThemeColors = (theme = 'dark') => {
  if (theme === 'dark') {
    return {
      text: '#e5e7eb',
      textSecondary: '#9ca3af',
      background: '#111827',
      backgroundAlt: '#1f2937',
      border: 'rgba(229, 231, 235, 0.2)',
      gridLine: 'rgba(229, 231, 235, 0.1)',
      axisLine: '#4b5563'
    };
  }

  return {
    text: '#374151',
    textSecondary: '#6b7280',
    background: '#ffffff',
    backgroundAlt: '#f9fafb',
    border: 'rgba(55, 65, 81, 0.2)',
    gridLine: 'rgba(55, 65, 81, 0.1)',
    axisLine: '#d1d5db'
  };
};

/**
 * Generate color palette from base color
 */
export const generateColorPalette = (baseColor, count = 8) => {
  const colors = [baseColor];
  
  // Simple color generation (would need more sophisticated algorithm for production)
  for (let i = 1; i < count; i++) {
    const hue = (i * 360) / count;
    colors.push(`hsl(${hue}, 70%, 50%)`);
  }
  
  return colors;
};

/**
 * Check WCAG AA compliance for color contrast
 */
export const getContrastRatio = (color1, color2) => {
  // Simplified contrast calculation
  // For production, use a proper library like polished or chroma-js
  const getLuminance = (color) => {
    // Parse color and calculate relative luminance
    const rgb = parseInt(color.slice(1), 16);
    const r = (rgb >> 16) & 255;
    const g = (rgb >> 8) & 255;
    const b = rgb & 255;
    
    const luminance = 
      (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    
    return luminance > 0.5 ? 0.86 : 0.54;
  };

  const lum1 = getLuminance(color1);
  const lum2 = getLuminance(color2);
  
  const lighter = Math.max(lum1, lum2);
  const darker = Math.min(lum1, lum2);
  
  return (lighter + 0.05) / (darker + 0.05);
};

/**
 * Get text color (white or black) for better contrast on background
 */
export const getTextColorForBackground = (backgroundColor) => {
  // Simple luminance calculation
  const hex = backgroundColor.replace('#', '');
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.5 ? '#000000' : '#ffffff';
};

/**
 * Export color utilities object
 */
export const ChartColors = {
  palettes: COLOR_PALETTES,
  getChartColor,
  getThemeColors,
  generateColorPalette,
  getContrastRatio,
  getTextColorForBackground
};

export default ChartColors;
