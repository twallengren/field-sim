export type Boundary = 'neumann' | 'periodic';
export type Model = 'civilization' | 'agriculture' | 'chemotaxis' | 'ecology';
export type FieldName = 'population' | 'food' | 'infrastructure' | 'soil' | 'fertility' | 'water' | 'waterSources' | 'cultivation';
export type ParameterKey = string;
export type Parameters = Record<ParameterKey, number>;
export interface LayerConfig { field: FieldName; opacity: number; visible: boolean; }
export interface TileConfig { id: string; layers: LayerConfig[]; paintField: FieldName; }
export interface FieldDescriptor {
  key: FieldName; label: string; palette: [number, number, number][];
  scale: number; color: string; bounded?: [number, number];
  editable: boolean; kind: 'dynamic' | 'static' | 'derived';
}
export interface Setup { version: 1 | 2; preset: string; seed: number; n: number; boundary: Boundary; parameters: Parameters; tiles?: TileConfig[]; }
export interface Preset { id: string; title: string; kicker: string; description: string; model: Model; parameters: Parameters; fields: FieldName[]; duration: number; }
export type Fields = Record<string, Float64Array>;
export interface FieldMetric { total: number; min: number; max: number; mean: number; }
export interface WaterBudget { initial: number; recharged: number; domesticUse: number; agriculturalUse: number; interventions: number; current: number; residual: number; }
export interface Snapshot { n: number; time: number; step: number; dt: number; fields: Fields; metrics: Record<string, FieldMetric>; waterBudget?: WaterBudget; truncatedMass: number; interventionMass: Partial<Record<FieldName, number>>; }
export interface Brush { field: FieldName; x: number; y: number; radius: number; amount: number; }
// Coordinates and radius are fractions of the domain. Positive amount adds density;
// negative amount removes it. Soil and water sources are limited to [0, 1].
// Other paintable stocks stay nonnegative; derived cultivation rejects painting.
export type WorkerCommand =
  | { type: 'init'; generation: number; setup: Setup }
  | { type: 'advance'; generation: number; duration: number; maxSteps: number }
  | { type: 'step'; generation: number }
  | { type: 'parameters'; generation: number; parameters: Parameters }
  | { type: 'paint'; generation: number; brush: Brush };
export type WorkerResponse =
  | { type: 'snapshot'; generation: number; snapshot: Snapshot }
  | { type: 'error'; generation: number; message: string };
