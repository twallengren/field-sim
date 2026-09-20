export type Boundary = 'neumann' | 'periodic';
export type Model = 'civilization' | 'agriculture' | 'chemotaxis';
export type FieldName = 'population' | 'food' | 'infrastructure' | 'soil' | 'fertility';
export type ParameterKey = 'dp' | 'df' | 'chiFood' | 'chiInfra' | 'growth' | 'regrowth' | 'consumption' | 'investment' | 'infraDecay' | 'infraBoost' | 'foodCost' | 'soilRecovery' | 'erosion';
export type Parameters = Record<ParameterKey, number>;
export interface Setup { version: 1; preset: string; seed: number; n: number; boundary: Boundary; parameters: Parameters; }
export interface Preset { id: string; title: string; kicker: string; description: string; model: Model; parameters: Parameters; fields: FieldName[]; duration: number; }
export type Fields = Record<FieldName, Float64Array>;
export interface FieldMetric { total: number; min: number; max: number; mean: number; }
export interface Snapshot { n: number; time: number; step: number; dt: number; fields: Fields; metrics: Record<FieldName, FieldMetric>; truncatedMass: number; interventionMass: Partial<Record<FieldName, number>>; }
export interface Brush { field: FieldName; x: number; y: number; radius: number; amount: number; }
// Coordinates and radius are fractions of the domain. Positive amount adds density;
// negative amount removes it. Soil is limited to [0, 1], others to >= 0.
export type WorkerCommand =
  | { type: 'init'; generation: number; setup: Setup }
  | { type: 'advance'; generation: number; duration: number; maxSteps: number }
  | { type: 'step'; generation: number }
  | { type: 'parameters'; generation: number; parameters: Parameters }
  | { type: 'paint'; generation: number; brush: Brush };
export type WorkerResponse =
  | { type: 'snapshot'; generation: number; snapshot: Snapshot }
  | { type: 'error'; generation: number; message: string };
