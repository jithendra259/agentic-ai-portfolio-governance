import React from 'react';
import { useXScale, useYScale } from '@mui/x-charts/hooks';
import { ChartsClipPath } from '@mui/x-charts/ChartsClipPath';

export default function ScatterRegressionLine({ regression }) {
  const xScale = useXScale();
  const yScale = useYScale();
  const clipPathId = `scatter-regression-${React.useId()}`;
  
  if (!regression || !xScale || !yScale) return null;
  
  const xMin = Number(regression.x_min);
  const xMax = Number(regression.x_max);
  const slope = Number(regression.slope);
  const intercept = Number(regression.intercept);
  
  const yMin = Number.isFinite(Number(regression.y_min)) ? Number(regression.y_min) : slope * xMin + intercept;
  const yMax = Number.isFinite(Number(regression.y_max)) ? Number(regression.y_max) : slope * xMax + intercept;
  
  if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) return null;
  
  const x1 = xScale(xMin); 
  const x2 = xScale(xMax); 
  const y1 = yScale(yMin); 
  const y2 = yScale(yMax);
  
  if (![x1, x2, y1, y2].every(Number.isFinite)) return null;
  
  return (
    <React.Fragment>
      <ChartsClipPath id={clipPathId} />
      <g clipPath={`url(#${clipPathId})`}>
        <line 
          x1={x1} 
          y1={y1} 
          x2={x2} 
          y2={y2} 
          stroke={regression.color || '#f25467'} 
          strokeWidth={2} 
          strokeDasharray={regression.strokeDasharray || '6 4'} 
          pointerEvents="none" 
        />
      </g>
    </React.Fragment>
  );
}
export { ScatterRegressionLine };
