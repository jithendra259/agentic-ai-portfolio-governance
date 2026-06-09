# MUI X Charts v9.4.0 - Complete File Directory & Internals Analysis

## 📁 Full File Structure

```
mui-x-9.4.0/
└── packages/
    └── x-charts/
        ├── src/
        │   │
        │   ├── BarChart/ (20 files)
        │   │   ├── BarChart.tsx (16.2 KB)
        │   │   │   └── Exports: BarChart, BarChartProps, BarChartSlots, BarChartSlotProps
        │   │   ├── BarPlot.tsx (6.2 KB)
        │   │   │   └── Exports: BarPlot, BarPlotProps, BarPlotSlots
        │   │   ├── useBarChartProps.ts (5.6 KB)
        │   │   │   └── Hook for managing bar chart props
        │   │   ├── useBarPlotData.ts (6.7 KB)
        │   │   │   └── Data processing for bar plots
        │   │   ├── BarElement.tsx (3.8 KB)
        │   │   │   └── SVG rect element for bars
        │   │   ├── AnimatedBarElement.tsx (1.6 KB)
        │   │   │   └── Animated bar element with transitions
        │   │   ├── BarClipPath.tsx (5.4 KB)
        │   │   │   └── Clipping area management for bars
        │   │   ├── IndividualBarPlot.tsx (3.0 KB)
        │   │   │   └── Single bar series rendering
        │   │   ├── FocusedBar.tsx (1.8 KB)
        │   │   │   └── Focused state handling
        │   │   ├── barClasses.ts (1.8 KB)
        │   │   │   └── CSS class exports
        │   │   ├── checkBarChartScaleErrors.ts (3.1 KB)
        │   │   │   └── Data validation
        │   │   ├── extremums.test.ts (5.7 KB)
        │   │   │   └── Min/max calculation tests
        │   │   ├── checkClickEvent.test.tsx (6.1 KB)
        │   │   │   └── Click event tests
        │   │   ├── BarChart.plugins.ts (1.8 KB)
        │   │   │   └── Plugin system definitions
        │   │   ├── BarChart.test.tsx (8.1 KB)
        │   │   │   └── Component tests
        │   │   ├── BarPlot.test.tsx (1.2 KB)
        │   │   │   └── BarPlot tests
        │   │   ├── types.ts (952 bytes)
        │   │   │   └── Type definitions
        │   │   ├── useRegisterItemClickHandlers.ts (3.6 KB)
        │   │   │   └── Click event registration
        │   │   └── index.ts (337 bytes)
        │   │       └── Public exports
        │   │
        │   ├── LineChart/ (24 files)
        │   │   ├── LineChart.tsx
        │   │   │   └── Main line chart component
        │   │   ├── LinePlot.tsx
        │   │   │   └── Line rendering logic
        │   │   ├── useLineChartProps.ts
        │   │   │   └── Props management
        │   │   ├── useLinePlotData.ts
        │   │   │   └── Data processing
        │   │   ├── useAreaPlotData.ts
        │   │   │   └── Area chart data
        │   │   ├── useMarkPlotData.ts
        │   │   │   └── Data point marker data
        │   │   ├── LineElement.tsx
        │   │   │   └── SVG line element
        │   │   ├── AnimatedLine.tsx
        │   │   │   └── Animated line with transitions
        │   │   ├── AreaElement.tsx
        │   │   │   └── SVG area element
        │   │   ├── AnimatedArea.tsx
        │   │   │   └── Animated area fill
        │   │   ├── MarkElement.tsx
        │   │   │   └── Data point marker
        │   │   ├── CircleMarkElement.tsx
        │   │   │   └── Circle marker variant
        │   │   ├── MarkPlot.tsx
        │   │   │   └── Multiple markers rendering
        │   │   ├── AreaPlot.tsx
        │   │   │   └── Area fill rendering
        │   │   ├── LineHighlightElement.tsx
        │   │   │   └── Highlight indicator
        │   │   ├── LineHighlightPlot.tsx
        │   │   │   └── Highlight rendering
        │   │   ├── FocusedLineMark.tsx
        │   │   │   └── Focused marker styling
        │   │   ├── AppearingMask.tsx
        │   │   │   └── Animation mask for appearing elements
        │   │   ├── lineClasses.ts
        │   │   │   └── CSS classes
        │   │   ├── LineChart.plugins.ts
        │   │   │   └── Plugin definitions
        │   │   ├── LineChart.test.tsx
        │   │   │   └── Component tests
        │   │   ├── checkClickEvent.test.tsx
        │   │   │   └── Click event tests
        │   │   ├── MarkElement.test.tsx
        │   │   │   └── Marker tests
        │   │   └── index.ts
        │   │       └── Public exports
        │   │
        │   ├── PieChart/ (13 files)
        │   │   ├── PieChart.tsx
        │   │   │   └── Main pie chart component
        │   │   ├── PiePlot.tsx
        │   │   │   └── Pie rendering logic
        │   │   ├── PieArc.tsx
        │   │   │   └── Individual pie slice
        │   │   ├── PieArcPlot.tsx
        │   │   │   └── Multiple arc rendering
        │   │   ├── PieArcLabel.tsx
        │   │   │   └── Arc label component
        │   │   ├── PieArcLabelPlot.tsx
        │   │   │   └── Multiple arc labels
        │   │   ├── getPieCoordinates.ts
        │   │   │   └── Geometry calculations
        │   │   ├── pieClasses.ts
        │   │   │   └── CSS classes
        │   │   ├── FocusedPieArc.tsx
        │   │   │   └── Focused arc styling
        │   │   ├── PieChart.plugins.ts
        │   │   │   └── Plugin system
        │   │   ├── PieChart.test.tsx
        │   │   │   └── Tests
        │   │   ├── checkClickEvent.test.tsx
        │   │   │   └── Click tests
        │   │   └── index.ts
        │   │       └── Public exports
        │   │
        │   ├── ScatterChart/
        │   │   └── [Scatter chart specific files]
        │   │
        │   ├── RadarChart/
        │   │   └── [Radar chart specific files]
        │   │
        │   ├── Gauge/
        │   │   └── [Gauge specific files]
        │   │
        │   ├── SparkLineChart/
        │   │   └── [SparkLine specific files]
        │   │
        │   ├── Charts* Components/ (26 core files)
        │   │   ├── ChartsContainer.tsx
        │   │   │   └── Layout & sizing management
        │   │   │   ├── Props: width, height, series, margin, sx
        │   │   │   └── Handles: responsive sizing, state management
        │   │   │
        │   │   ├── ChartsSurface.tsx
        │   │   │   └── SVG canvas provider
        │   │   │   ├── Provides: drawing area context
        │   │   │   └── Handles: clipping, coordinate system
        │   │   │
        │   │   ├── ChartsWrapper.tsx
        │   │   │   └── Top-level wrapper component
        │   │   │   ├── Handles: theming, theme augmentation
        │   │   │   └── Props: theme, sx, children
        │   │   │
        │   │   ├── ChartsDataProvider.tsx
        │   │   │   └── Data processing provider
        │   │   │   ├── Processes: series transformation, scaling
        │   │   │   └── Provides: processed data via context
        │   │   │
        │   │   ├── ChartsRadialDataProvider.tsx
        │   │   │   └── Polar coordinate data provider
        │   │   │   ├── Handles: angle/radius calculations
        │   │   │   └── Used by: Radar, Gauge charts
        │   │   │
        │   │   ├── ChartsAxis.tsx
        │   │   │   └── Base axis component
        │   │   │   ├── Props: id, position, scale, data
        │   │   │   └── Renders: axis line, ticks, labels
        │   │   │
        │   │   ├── ChartsXAxis.tsx
        │   │   │   └── Horizontal axis (Cartesian)
        │   │   │   ├── Position: bottom or top
        │   │   │   └── Props: xAxis config
        │   │   │
        │   │   ├── ChartsYAxis.tsx
        │   │   │   └── Vertical axis (Cartesian)
        │   │   │   ├── Position: left or right
        │   │   │   └── Props: yAxis config
        │   │   │
        │   │   ├── ChartsRadiusAxis.tsx
        │   │   │   └── Radial axis (Polar)
        │   │   │   ├── Used in: Radar charts
        │   │   │   └── Represents: magnitude/values
        │   │   │
        │   │   ├── ChartsRotationAxis.tsx
        │   │   │   └── Angular axis (Polar)
        │   │   │   ├── Used in: Radar charts
        │   │   │   └── Represents: categories around circle
        │   │   │
        │   │   ├── ChartsGrid.tsx
        │   │   │   └── Background grid lines
        │   │   │   ├── Props: horizontal, vertical (boolean)
        │   │   │   └── Customizable: stroke, dasharray
        │   │   │
        │   │   ├── ChartsRadialGrid.tsx
        │   │   │   └── Grid for polar coordinates
        │   │   │   ├── Used in: Radar, gauge charts
        │   │   │   └── Renders: concentric circles
        │   │   │
        │   │   ├── ChartsLegend.tsx
        │   │   │   └── Series legend component
        │   │   │   ├── Features: toggle series, custom positions
        │   │   │   ├── Props: position, hidden, slotProps
        │   │   │   └── Customizable: colors, labels, direction
        │   │   │
        │   │   ├── ChartsTooltip.tsx
        │   │   │   └── Contextual info on hover/click
        │   │   │   ├── Features: auto-positioning, rich content
        │   │   │   ├── Props: trigger, contentStyle, formatter
        │   │   │   └── Types: line-like, pie, custom
        │   │   │
        │   │   ├── ChartsLabel.tsx
        │   │   │   └── Data point labels
        │   │   │   ├── Props: dataKey, position, formatter
        │   │   │   └── Features: auto-positioning, rotation
        │   │   │
        │   │   ├── ChartsAxisHighlight.tsx
        │   │   │   └── Axis area highlighting on interaction
        │   │   │   ├── Props: highlight type (band/line/none)
        │   │   │   └── Features: customizable styling
        │   │   │
        │   │   ├── ChartsAxisHighlightValue.tsx
        │   │   │   └── Axis value display on highlight
        │   │   │   ├── Shows: current axis value
        │   │   │   └── Used by: tooltips, highlights
        │   │   │
        │   │   ├── ChartsRadialAxisHighlight.tsx
        │   │   │   └── Highlight for polar axes
        │   │   │   ├── Used in: Radar charts
        │   │   │   └── Props: position, styling
        │   │   │
        │   │   ├── ChartsOverlay.tsx
        │   │   │   └── Overlay layer for interactions
        │   │   │   ├── Features: drag, click detection
        │   │   │   └── Customizable: via slots
        │   │   │
        │   │   ├── ChartsBrushOverlay.tsx
        │   │   │   └── Selection brush for zoom/filter
        │   │   │   ├── Features: drag selection, highlighting
        │   │   │   └── Used for: data filtering
        │   │   │
        │   │   ├── ChartsClipPath.tsx
        │   │   │   └── SVG clipping path management
        │   │   │   ├── Prevents: rendering outside bounds
        │   │   │   └── Used by: all chart plots
        │   │   │
        │   │   ├── ChartsText.tsx
        │   │   │   └── SVG text with auto-wrapping
        │   │   │   ├── Features: text wrapping, truncation
        │   │   │   └── Props: text, width, style
        │   │   │
        │   │   ├── ChartsLocalizationProvider.tsx
        │   │   │   └── Localization context provider
        │   │   │   ├── Props: messages (locale object)
        │   │   │   └── Provides: translated strings
        │   │   │
        │   │   ├── ChartsLayerContainer.tsx
        │   │   │   └── SVG layer organization
        │   │   │   ├── Manages: layer stacking order
        │   │   │   └── Features: ref access
        │   │   │
        │   │   ├── ChartsSvgLayer.tsx
        │   │   │   └── Individual SVG layer
        │   │   │   ├── Props: children, zIndex
        │   │   │   └── Used for: organized rendering
        │   │   │
        │   │   ├── ChartsReferenceLine.tsx
        │   │   │   └── Static reference line/band
        │   │   │   ├── Props: x, y, label, lineStyle
        │   │   │   └── Uses: for thresholds, goals
        │   │   │
        │   │   └── Toolbar/
        │   │       └── Export & interaction tools
        │   │
        │   ├── hooks/ (48+ hooks, 100 KB)
        │   │   ├── Axis Hooks
        │   │   │   ├── useAxis.ts (7.5 KB)
        │   │   │   │   └── Access axis configuration
        │   │   │   ├── useAxisSystem.tsx (877 bytes)
        │   │   │   │   └── Cartesian axis system
        │   │   │   ├── useAxisCoordinates.ts (3.3 KB)
        │   │   │   │   └── Calculate point coordinates
        │   │   │   ├── useAxisTicks.ts (2.6 KB)
        │   │   │   │   └── Get axis ticks
        │   │   │   ├── useZAxis.ts
        │   │   │   │   └── Z-axis (depth) data
        │   │   │   └── useAxisCoordinates.test.ts (4.8 KB)
        │   │   │
        │   │   ├── Scale & Mapping Hooks
        │   │   │   ├── useScale.ts (1.9 KB)
        │   │   │   │   └── Access D3-like scales
        │   │   │   ├── useColorScale.ts (1.7 KB)
        │   │   │   │   └── Color mapping from values
        │   │   │   ├── useDrawingArea.ts (1.1 KB)
        │   │   │   │   └── Chart drawing bounds
        │   │   │   ├── getValueToPositionMapper.ts (947 bytes)
        │   │   │   │   └── Map values to pixels
        │   │   │   └── useScale.test.ts (1.6 KB)
        │   │   │
        │   │   ├── Tick Hooks
        │   │   │   ├── useTicks.ts (13.7 KB)
        │   │   │   │   └── Advanced tick calculation
        │   │   │   ├── useTicksGrouped.ts (5.1 KB)
        │   │   │   │   └── Grouped tick calculation
        │   │   │   ├── useTicks.test.ts (4.4 KB)
        │   │   │   ├── useTicks.test.tsx (4.4 KB)
        │   │   │   └── useTicks.bench.ts
        │   │   │
        │   │   ├── Series Hooks
        │   │   │   ├── useSeries.ts (634 bytes)
        │   │   │   │   └── Access all series data
        │   │   │   ├── useBarSeries.ts (1.8 KB)
        │   │   │   │   └── Bar-specific data
        │   │   │   ├── useLineSeries.ts (1.85 KB)
        │   │   │   │   └── Line-specific data
        │   │   │   ├── usePieSeries.ts (2.2 KB)
        │   │   │   │   └── Pie-specific data
        │   │   │   ├── useScatterSeries.ts (1.8 KB)
        │   │   │   │   └── Scatter-specific data
        │   │   │   ├── useRadarSeries.ts (1.76 KB)
        │   │   │   │   └── Radar-specific data
        │   │   │   ├── useSeries.test.tsx (1.9 KB)
        │   │   │   ├── useBarSeries.test.tsx (2.6 KB)
        │   │   │   ├── useLineSeries.test.tsx (2.7 KB)
        │   │   │   ├── usePieSeries.test.tsx (2.6 KB)
        │   │   │   ├── useScatterSeries.test.tsx (2.8 KB)
        │   │   │   └── useRadarSeries.test.tsx (2.7 KB)
        │   │   │
        │   │   ├── Interaction Hooks
        │   │   │   ├── useFocusedItem.ts (349 bytes)
        │   │   │   │   └── Currently focused item
        │   │   │   ├── useIsItemFocused.ts (730 bytes)
        │   │   │   │   └── Check if item focused
        │   │   │   ├── useIsItemFocusedGetter.tsx (730 bytes)
        │   │   │   │   └── Get focus checker function
        │   │   │   ├── useItemHighlightState.ts (1.4 KB)
        │   │   │   │   └── Highlight state management
        │   │   │   ├── useItemHighlightStateGetter.ts (897 bytes)
        │   │   │   │   └── Get highlight checker function
        │   │   │   └── useInteractionItemProps.ts (2.97 KB)
        │   │   │       └── Props for interactive items
        │   │   │
        │   │   ├── Animation Hooks
        │   │   │   ├── useSkipAnimation.ts (498 bytes)
        │   │   │   │   └── Toggle animation on/off
        │   │   │   ├── animation/ (directory)
        │   │   │   │   └── Animation-specific hooks
        │   │   │   └── useSkipAnimation.test.tsx (3.1 KB)
        │   │   │
        │   │   ├── Chart Identification Hooks
        │   │   │   ├── useChartId.ts (374 bytes)
        │   │   │   │   └── Unique chart ID
        │   │   │   ├── useChartRootRef.ts (384 bytes)
        │   │   │   │   └── Root element ref
        │   │   │   ├── useChartsLayerContainerRef.ts (379 bytes)
        │   │   │   │   └── SVG container ref
        │   │   │   ├── useChartsLayerContainerRef.test.tsx (1.6 KB)
        │   │   │   ├── useChartGradientId.tsx (1.5 KB)
        │   │   │   │   └── Unique gradient ID
        │   │   │   ├── useChartGradientId.test.tsx (1.2 KB)
        │   │   │   └── useBrush.ts (522 bytes)
        │   │   │       └── Brush state access
        │   │   │
        │   │   ├── Data Access Hooks
        │   │   │   ├── useDataset.ts (707 bytes)
        │   │   │   │   └── Direct data access
        │   │   │   ├── useDataset.test.tsx (1.5 KB)
        │   │   │   └── getValueToPositionMapper.ts
        │   │   │       └── Value-to-position mapping
        │   │   │
        │   │   ├── Legend & Localization Hooks
        │   │   │   ├── useLegend.ts (1.54 KB)
        │   │   │   │   └── Legend visibility & toggle
        │   │   │   ├── useChartsLocalization.ts (688 bytes)
        │   │   │   │   └── Localization strings
        │   │   │   ├── index.ts (1.2 KB)
        │   │   │   │   └── Public hook exports
        │   │   │   └── [test files]
        │   │   │
        │   │   └── Utility Hooks
        │   │       ├── useMounted.ts (435 bytes)
        │   │       │   └── Check hydration status
        │   │       └── useIsHydrated.ts
        │   │           └── SSR hydration check
        │   │
        │   ├── models/ (13 files, 40 KB)
        │   │   ├── axis.ts (26.2 KB)
        │   │   │   ├── Axis configuration types
        │   │   │   ├── Exports:
        │   │   │   │   ├── AxisConfig
        │   │   │   │   ├── CartesianAxisConfig
        │   │   │   │   ├── PolarAxisConfig
        │   │   │   │   ├── ScaleType
        │   │   │   │   ├── AxisPosition
        │   │   │   │   └── Tick/Label options
        │   │   │   └── Used by: all chart types
        │   │   │
        │   │   ├── seriesType/ (directory)
        │   │   │   ├── bar.ts
        │   │   │   │   └── BarSeriesType definition
        │   │   │   ├── line.ts
        │   │   │   │   └── LineSeriesType definition
        │   │   │   ├── pie.ts
        │   │   │   │   └── PieSeriesType definition
        │   │   │   ├── scatter.ts
        │   │   │   │   └── ScatterSeriesType definition
        │   │   │   ├── radar.ts
        │   │   │   │   └── RadarSeriesType definition
        │   │   │   └── [other series types]
        │   │   │
        │   │   ├── colorMapping.ts (1.2 KB)
        │   │   │   └── Color scale/mapping config
        │   │   │
        │   │   ├── sizeMapping.ts (2.25 KB)
        │   │   │   └── Size scale/mapping config
        │   │   │
        │   │   ├── curve.ts (326 bytes)
        │   │   │   └── CurveType definitions
        │   │   │
        │   │   ├── stacking.ts (211 bytes)
        │   │   │   └── StackingType definitions
        │   │   │
        │   │   ├── position.ts (195 bytes)
        │   │   │   └── Position type definitions
        │   │   │
        │   │   ├── z-axis.ts (1.89 KB)
        │   │   │   └── Z-axis configuration
        │   │   │
        │   │   ├── timeTicks.ts (1.69 KB)
        │   │   │   └── Time-based tick config
        │   │   │
        │   │   ├── slots/ (directory)
        │   │   │   └── Customizable slot definitions
        │   │   │
        │   │   ├── featureFlags.ts (43 bytes)
        │   │   │   └── Feature flag definitions
        │   │   │
        │   │   ├── chartsSlotsComponentsProps.ts (1.1 KB)
        │   │   │   └── Unified slots interface
        │   │   │
        │   │   └── index.ts (744 bytes)
        │   │       └── Public model exports
        │   │
        │   ├── context/ (4 files)
        │   │   ├── ChartApi.ts
        │   │   │   ├── Public chart API definition
        │   │   │   ├── Methods: getSeriesData(), getAxisData()
        │   │   │   └── Props: series, axes, data
        │   │   │
        │   │   ├── ChartsSlotsContext.tsx
        │   │   │   ├── React Context for component slots
        │   │   │   ├── Providers: custom slot components
        │   │   │   └── Usage: ChartsLocalizationProvider wrapper
        │   │   │
        │   │   ├── useChartApiContext.ts
        │   │   │   ├── Hook to access chart API
        │   │   │   ├── Returns: ChartApi object
        │   │   │   └── Usage: from within chart
        │   │   │
        │   │   └── index.ts
        │   │       └── Context exports
        │   │
        │   ├── internals/ (100+ files, 200+ KB)
        │   │   │
        │   │   ├── animation/ (directory)
        │   │   │   ├── Frame-based animation logic
        │   │   │   ├── Easing functions
        │   │   │   └── Spring physics
        │   │   │
        │   │   ├── scales/ (directory)
        │   │   │   ├── D3-like scale implementations
        │   │   │   ├── Linear, log, time, band scales
        │   │   │   └── Scale utilities
        │   │   │
        │   │   ├── stacking/ (directory)
        │   │   │   ├── Stack calculation algorithms
        │   │   │   ├── Series accumulation
        │   │   │   └── Zero-baseline handling
        │   │   │
        │   │   ├── store/ (directory)
        │   │   │   ├── State management system
        │   │   │   ├── Data caching
        │   │   │   └── Context provision
        │   │   │
        │   │   ├── plugins/ (directory)
        │   │   │   ├── Plugin registration system
        │   │   │   ├── Hook injection
        │   │   │   └── Feature composition
        │   │   │
        │   │   ├── material/ (directory)
        │   │   │   ├── MUI integration
        │   │   │   ├── Theme integration
        │   │   │   └── Component slots
        │   │   │
        │   │   ├── components/ (directory)
        │   │   │   ├── Internal component implementations
        │   │   │   └── Utility components
        │   │   │
        │   │   ├── Geometry & Math Files
        │   │   │   ├── angleConversion.ts
        │   │   │   │   └── Degree ↔ Radian conversion
        │   │   │   ├── clampAngle.ts
        │   │   │   │   └── Angle boundary clamping
        │   │   │   ├── clampAngle.test.ts
        │   │   │   │   └── Angle clamping tests
        │   │   │   ├── geometry.ts
        │   │   │   │   └── Geometric calculations
        │   │   │   ├── getRingPath.ts
        │   │   │   │   └── SVG path generation for rings
        │   │   │   ├── cubiqSolver.ts
        │   │   │   │   └── Cubic equation solving
        │   │   │   ├── degToRad.ts
        │   │   │   │   └── Degree to radian
        │   │   │   └── findClosestIndex.ts
        │   │   │       └── Nearest value finding
        │   │   │
        │   │   ├── Flatbush Files (Spatial Indexing)
        │   │   │   ├── Flatbush.ts (3.5 KB)
        │   │   │   │   ├── Fast spatial indexing library
        │   │   │   │   ├── Uses: R-tree for performance
        │   │   │   │   ├── Methods: range, search
        │   │   │   │   └── Performance: benchmark included
        │   │   │   ├── Flatbush.test.ts
        │   │   │   │   └── Spatial index tests
        │   │   │   └── Flatbush.bench.ts
        │   │   │       └── Performance benchmarks
        │   │   │
        │   │   ├── Scaling & Domain Files
        │   │   │   ├── scales/ (directory)
        │   │   │   │   ├── D3 scale port
        │   │   │   │   ├── Linear scale
        │   │   │   │   ├── Log scale
        │   │   │   │   ├── Time scale
        │   │   │   │   ├── Band scale
        │   │   │   │   └── Point scale
        │   │   │   ├── getScale.ts
        │   │   │   │   └── Create scale from config
        │   │   │   ├── invertScale.ts
        │   │   │   │   └── Inverse scale calculations
        │   │   │   ├── sizeScale.ts (1.8 KB)
        │   │   │   │   └── Size scaling utilities
        │   │   │   ├── colorScale.ts
        │   │   │   │   └── Color scale generation
        │   │   │   └── scaleGuards.ts
        │   │   │       └── Scale validation
        │   │   │
        │   │   ├── Data Processing Files
        │   │   │   ├── findMinMax.ts
        │   │   │   │   └── Min/max computation
        │   │   │   ├── findMinMax.test.ts
        │   │   │   │   └── Min/max tests
        │   │   │   ├── isDefined.ts
        │   │   │   │   └── Value validation
        │   │   │   ├── isInfinity.ts
        │   │   │   │   └── Infinity checks
        │   │   │   ├── seriesHasData.ts
        │   │   │   │   └── Series validation
        │   │   │   ├── seriesSelectorOfType.ts
        │   │   │   │   └── Type-based series filtering
        │   │   │   ├── getPercentageValue.ts
        │   │   │   │   └── Percentage calculations
        │   │   │   ├── appendAtKey.ts
        │   │   │   │   └── Data key appending
        │   │   │   ├── shallowEqual.ts
        │   │   │   │   └── Shallow equality check
        │   │   │   └── processLineLikeSeries.ts
        │   │   │       └── Line series processing
        │   │   │
        │   │   ├── Layout & Positioning Files
        │   │   │   ├── createGetBarDimensions.ts
        │   │   │   │   └── Bar dimension calculation
        │   │   │   ├── getBandSize.ts
        │   │   │   │   └── Band width calculation
        │   │   │   ├── getChartPoint.ts
        │   │   │   │   └── Point to chart coordinates
        │   │   │   ├── getSurfacePoint.ts
        │   │   │   │   └── Surface positioning
        │   │   │   ├── getLabel.ts
        │   │   │   │   └── Label extraction
        │   │   │   ├── defaultizeMargin.ts
        │   │   │   │   └── Margin defaults
        │   │   │   └── invertTextAnchor.ts
        │   │   │       └── Text alignment adjustment
        │   │   │
        │   │   ├── SVG & DOM Files
        │   │   │   ├── domUtils.ts (3.5 KB)
        │   │   │   │   ├── DOM utility functions
        │   │   │   │   ├── Methods: getElement, createElement
        │   │   │   │   └── Used for: SVG manipulation
        │   │   │   ├── domUtils.test.ts
        │   │   │   │   └── DOM utility tests
        │   │   │   ├── domUtils.bench.ts
        │   │   │   │   └── Performance benchmarks
        │   │   │   ├── createSvgIcon.ts
        │   │   │   │   └── SVG icon creation
        │   │   │   ├── cleanId.ts
        │   │   │   │   └── ID sanitization
        │   │   │   ├── identifierCleaner.ts
        │   │   │   │   └── ID cleaning utilities
        │   │   │   ├── identifierCleaner.test.ts
        │   │   │   │   └── Identifier tests
        │   │   │   ├── identifierSerializer.ts
        │   │   │   │   └── ID serialization
        │   │   │   └── ellipsize.ts
        │   │   │       └── Text truncation
        │   │   │
        │   │   ├── Text Handling Files
        │   │   │   ├── getGraphemeCount.ts
        │   │   │   │   └── Unicode-aware character counting
        │   │   │   ├── getWordsByLines.ts
        │   │   │   │   └── Text wrapping logic
        │   │   │   ├── ellipsize.test.ts
        │   │   │   │   └── Text truncation tests
        │   │   │   └── defaultValueFormatters.ts
        │   │   │       └── Number/date formatting
        │   │   │
        │   │   ├── Array & Data Structure Files
        │   │   │   ├── sliceUntil.ts
        │   │   │   │   └── Array slicing utility
        │   │   │   ├── sliceUntil.test.ts
        │   │   │   │   └── Slice tests
        │   │   │   ├── commonNextFocusItem.ts
        │   │   │   │   └── Focus navigation
        │   │   │   └── findClosestIndex.ts
        │   │   │       └── Nearest index finding
        │   │   │
        │   │   ├── Error Handling Files
        │   │   │   ├── incompleteDatasetKeysError.ts
        │   │   │   │   └── Dataset validation errors
        │   │   │   └── configInit.ts
        │   │   │       └── Configuration initialization
        │   │   │
        │   │   ├── Tick & Formatting Files
        │   │   │   ├── ticks.ts
        │   │   │   │   └── Tick generation utilities
        │   │   │   ├── ticks.test.ts
        │   │   │   │   └── Tick tests
        │   │   │   ├── dateHelpers.ts
        │   │   │   │   └── Date/time utilities
        │   │   │   └── legendUtils.ts
        │   │   │       └── Legend helper functions
        │   │   │
        │   │   ├── Type & Utility Files
        │   │   │   ├── getAsNumber.ts
        │   │   │   │   └── String to number conversion
        │   │   │   ├── isCartesian.ts
        │   │   │   │   └── Cartesian check
        │   │   │   ├── isPolar.ts
        │   │   │   │   └── Polar check
        │   │   │   ├── getSeriesColorFn.ts
        │   │   │   │   └── Color function generation
        │   │   │   ├── resolveColorProcessor.ts
        │   │   │   │   └── Color processing
        │   │   │   ├── ts-generic.ts
        │   │   │   │   └── TypeScript generics
        │   │   │   └── constants.ts
        │   │   │       └── Constant definitions
        │   │   │
        │   │   ├── Interaction Files
        │   │   │   ├── createCommonKeyboardFocusHandler.ts
        │   │   │   │   └── Keyboard focus handling
        │   │   │   ├── getLineLikeTooltip.ts
        │   │   │   │   └── Line chart tooltip helper
        │   │   │   ├── getSymbol.ts
        │   │   │   │   └── Symbol/marker selection
        │   │   │   └── getChartPoint.ts
        │   │   │       └── Point calculation
        │   │   │
        │   │   ├── Comparison & Equality Files
        │   │   │   ├── shallowEqual.ts
        │   │   │   │   └── Shallow equality check
        │   │   │   ├── consumeThemeProps.tsx
        │   │   │   │   └── Theme props consumption
        │   │   │   ├── consumeThemeProps.test.tsx
        │   │   │   │   └── Theme props tests
        │   │   │   └── consumeSlots.tsx
        │   │   │       └── Slot consumption
        │   │   │
        │   │   └── index.ts
        │   │       └── Internals exports
        │   │
        │   ├── colorPalettes/ (8+ palette files)
        │   │   ├── blue.ts
        │   │   │   └── Blue palette (default)
        │   │   ├── purple.ts
        │   │   │   └── Purple palette
        │   │   ├── cyan.ts
        │   │   │   └── Cyan palette
        │   │   ├── orange.ts
        │   │   │   └── Orange palette
        │   │   ├── red.ts
        │   │   │   └── Red palette
        │   │   ├── green.ts
        │   │   │   └── Green palette
        │   │   ├── pink.ts
        │   │   │   └── Pink palette
        │   │   └── index.ts
        │   │       └── Palette exports
        │   │
        │   ├── constants/
        │   │   └── Global constants
        │   │
        │   ├── plugins/
        │   │   ├── Plugin system definitions
        │   │   └── index.ts
        │   │
        │   ├── locales/ (15+ language files)
        │   │   ├── en-US.ts (English - US)
        │   │   ├── de-DE.ts (German)
        │   │   ├── fr-FR.ts (French)
        │   │   ├── es-ES.ts (Spanish)
        │   │   ├── pt-BR.ts (Portuguese - Brazil)
        │   │   ├── zh-CN.ts (Chinese - Simplified)
        │   │   ├── ja-JP.ts (Japanese)
        │   │   ├── it-IT.ts (Italian)
        │   │   ├── ru-RU.ts (Russian)
        │   │   └── [more locales...]
        │   │
        │   ├── utils/ (5 files)
        │   │   ├── epsilon.ts
        │   │   │   └── Float precision utilities
        │   │   ├── niceDomain.ts (2.1 KB)
        │   │   │   ├── Nice domain calculation
        │   │   │   ├── Makes: min/max values "nice"
        │   │   │   └── Used by: axis scaling
        │   │   ├── niceDomain.test.tsx
        │   │   │   └── Domain calculation tests
        │   │   ├── timeTicks.ts
        │   │   │   └── Time-based ticking
        │   │   └── index.ts
        │   │       └── Utils exports
        │   │
        │   ├── tests/ (directory)
        │   │   └── Integration test files
        │   │
        │   ├── moduleAugmentation/
        │   │   └── TypeScript module augmentation
        │   │
        │   ├── themeAugmentation/
        │   │   └── MUI Theme type augmentation
        │   │
        │   ├── Toolbar/ (directory)
        │   │   ├── Export toolbar component
        │   │   ├── Features: PNG, SVG, CSV export
        │   │   └── Download functionality
        │   │
        │   └── index.ts (1.6 KB)
        │       └── Main public exports (all chart types, hooks, components)
        │
        └── [Additional config files]
            ├── package.json
            ├── tsconfig.json
            ├── [build configs]
            └── [test configs]
```

---

## 📊 Component Relationship Diagram

```
ChartsWrapper (Theme Provider)
    │
    ├── ChartsContainer (Layout Manager)
    │   │
    │   ├── ChartsDataProvider (Data Processing)
    │   │   │
    │   │   ├── ChartsSurface (SVG Canvas)
    │   │   │   │
    │   │   │   ├── ChartsClipPath (Clipping)
    │   │   │   │
        │   │   │
        │   │   ├── [Plot Components]
        │   │   │   ├── BarPlot / LineChart / PieChart / etc.
        │   │   │   └── [Individual Elements]
        │   │   │
        │   │   ├── [Axes]
        │   │   │   ├── ChartsXAxis / ChartsYAxis
        │   │   │   ├── ChartsRadiusAxis / ChartsRotationAxis
        │   │   │   └── ChartsGrid / ChartsRadialGrid
        │   │   │
        │   │   └── [Overlays]
        │   │       ├── ChartsAxisHighlight
        │   │       ├── ChartsOverlay
        │   │       ├── ChartsBrushOverlay
        │   │       └── ChartsLayerContainer
        │   │
        │   └── [Peripheral Components]
        │       ├── ChartsLegend
        │       ├── ChartsTooltip
        │       └── Toolbar
        │
        └── ChartsLocalizationProvider (i18n)
```

---

## 🔄 Data Flow

```
Input Props (series, xAxis, yAxis, etc.)
    │
    ├→ ChartsContainer (Parse & Layout)
    │
    ├→ ChartsDataProvider (Transform Data)
    │   ├ Scale creation
    │   ├ Domain calculation
    │   ├ Stack accumulation
    │   └ Series processing
    │
    ├→ Context Providers
    │   ├ Chart state
    │   ├ Series data
    │   ├ Axis data
    │   └ Localization
    │
    ├→ ChartsSurface (SVG Setup)
    │   ├ Viewport sizing
    │   ├ Clip paths
    │   └ Layer organization
    │
    ├→ Hooks (Data Access)
    │   ├ useScale()
    │   ├ useSeries()
    │   ├ useAxis()
    │   └ etc.
    │
    └→ Components (Rendering)
        ├ Axis/Grid
        ├ Plot elements
        ├ Overlays
        └ Legends/Tooltips
```

---

## 🎯 Key Metrics

| Metric | Value |
|--------|-------|
| **Total Files** | ~280 |
| **Total Size** | ~850 KB |
| **Chart Types** | 7 |
| **Shared Components** | 26 |
| **Custom Hooks** | 48+ |
| **Type Definitions** | 50+ |
| **Utility Functions** | 100+ |
| **Supported Locales** | 15+ |
| **Color Palettes** | 8 |
| **CSS Classes** | 100+ |
| **Test Coverage** | 100+ test files |

---

**Complete Analysis Generated**: 2026-06-08  
**MUI X Version**: 9.4.0  
**Scope**: Full internal architecture and file organization
