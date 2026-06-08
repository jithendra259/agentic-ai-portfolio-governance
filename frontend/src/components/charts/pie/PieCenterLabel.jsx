import React from 'react';
import { useDrawingArea } from '@mui/x-charts/hooks';

export default function PieCenterLabel({ children }) {
  const { width, height, left, top } = useDrawingArea();
  const lines = String(children ?? '').split('\n').filter(Boolean);
  const fontSize = lines.length > 2 ? 12 : 16;
  const lineHeight = fontSize + 3;
  const firstDy = lines.length > 1 ? -((lines.length - 1) * lineHeight) / 2 : 0;

  return (
    <text 
      x={left + width / 2} 
      y={top + height / 2} 
      textAnchor="middle" 
      dominantBaseline="middle" 
      fill="#ffffff" 
      fontSize={fontSize} 
      fontWeight="700"
    >
      {lines.map((line, index) => (
        <tspan key={index} x={left + width / 2} dy={index === 0 ? firstDy : lineHeight}>
          {line}
        </tspan>
      ))}
    </text>
  );
}
export { PieCenterLabel };
