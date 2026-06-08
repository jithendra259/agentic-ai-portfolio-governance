import React, { useRef, useState, useEffect } from 'react';

export function useResponsiveChartWidth(fallback = 360) {
  const ref = useRef(null);
  const [width, setWidth] = useState(fallback);

  useEffect(() => {
    if (!ref.current) return;
    const updateWidth = (value) => setWidth(Math.max(280, Math.floor(value || fallback)));
    updateWidth(ref.current.getBoundingClientRect().width);
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries?.[0]) updateWidth(entries[0].contentRect.width);
    });
    resizeObserver.observe(ref.current);
    return () => resizeObserver.disconnect();
  }, [fallback]);

  return [ref, width];
}
