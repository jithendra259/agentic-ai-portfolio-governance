import React from 'react';
import { useDrawingArea, useXScale } from '@mui/x-charts/hooks';
import { useTheme, alpha } from '@mui/material/styles';

export default function RecessionBands({ periods }) {
  const { top, left, width, height } = useDrawingArea();
  const xScale = useXScale();
  const theme = useTheme();
  const labelFill = alpha(theme.palette.text.primary, 0.7);

  if (!periods || !Array.isArray(periods)) return null;

  return (
    <g>
      {periods.map((p, index) => {
        if (!p.start || !p.end) return null;
        const startDate = new Date(p.start);
        const endDate = new Date(p.end);
        const xStart = xScale(startDate);
        const xEnd = xScale(endDate);
        if (xStart === undefined || xEnd === undefined || isNaN(xStart) || isNaN(xEnd)) return null;
        
        let startX = xStart;
        let endX = xEnd;
        if (startX < left) startX = left;
        if (endX > left + width) endX = left + width;
        if (startX >= endX) return null;
        
        const textX = xStart >= left ? xStart : left;
        return (
          <React.Fragment key={index}>
            <rect 
              x={startX} 
              y={top} 
              width={endX - startX} 
              height={height} 
              fill="grey" 
              opacity={0.15} 
            />
            <text 
              x={textX + 4} 
              y={top - 5} 
              textAnchor="start" 
              dominantBaseline="auto" 
              fill={labelFill} 
              fontSize="0.75rem" 
              fontWeight={500} 
              pointerEvents="none"
            >
              {p.label}
            </text>
          </React.Fragment>
        );
      })}
    </g>
  );
}
export { RecessionBands };
