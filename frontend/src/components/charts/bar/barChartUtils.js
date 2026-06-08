export const AXIS_STYLE = { fill: '#e5e7eb', fontSize: 12, fontWeight: 600 };
export const GRID_STYLE = { stroke: '#2b3138', strokeWidth: 1 };

export function decorateAxes(axes) {
  return (axes || []).map((axis) => ({
    ...axis,
    tickLabelStyle: axis.tickLabelStyle || AXIS_STYLE,
  }));
}
