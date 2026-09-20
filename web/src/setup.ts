import { defaultSetup, getParameterDefinitions, getFieldDescriptors, getPreset, presets } from './catalog';
import type { FieldName, Parameters, Setup, TileConfig } from './contracts';

export interface LoadedSetup {
  setup: Setup;
  message?: string;
}

const VALID_RESOLUTIONS = new Set([32, 64, 128]);
const VALID_BOUNDARIES = new Set(['neumann', 'periodic']);
const PRESET_IDS = new Set(presets.map((preset) => preset.id));


function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Validate untrusted setup data and return a fresh, contract-shaped object. */
export function validateSharedSetup(value: unknown): Setup | undefined {
  if (!isRecord(value) || (value.version !== 1 && value.version !== 2) || typeof value.preset !== 'string') return undefined;
  if (!PRESET_IDS.has(value.preset)) return undefined;
  const model = getPreset(value.preset).model;
  if (value.version === 1 && model === 'ecology') return undefined;
  const parameterDefinitions = getParameterDefinitions(model);
  const PARAMETER_KEYS = new Set(parameterDefinitions.map(definition => definition.key));
  if (!Number.isInteger(value.seed) || (value.seed as number) < 0 || (value.seed as number) > 0xffff_ffff) return undefined;
  if (!Number.isInteger(value.n) || !VALID_RESOLUTIONS.has(value.n as number)) return undefined;
  if (typeof value.boundary !== 'string' || !VALID_BOUNDARIES.has(value.boundary)) return undefined;
  if (!isRecord(value.parameters)) return undefined;

  const suppliedKeys = Object.keys(value.parameters);
  if (suppliedKeys.length !== parameterDefinitions.length || suppliedKeys.some((key) => !PARAMETER_KEYS.has(key as keyof Parameters))) {
    return undefined;
  }

  const parameters = {} as Parameters;
  for (const definition of parameterDefinitions) {
    const parameter = value.parameters[definition.key];
    if (typeof parameter !== 'number' || !Number.isFinite(parameter)) return undefined;
    if (parameter < definition.min || parameter > definition.max) return undefined;
    parameters[definition.key] = parameter;
  }

  const tiles = value.tiles === undefined ? undefined : validateTiles(value.tiles, model);
  if (value.tiles !== undefined && (!tiles || value.version !== 2)) return undefined;
  return {
    version: value.version,
    preset: value.preset,
    seed: value.seed as number,
    n: value.n as number,
    boundary: value.boundary as Setup['boundary'],
    parameters,
    ...(tiles ? {tiles} : {}),
  };
}

/** Read `#setup=<encoded JSON>`, returning safe defaults for absent or bad input. */
export function loadSetupFragment(fragment: string): LoadedSetup {
  const fallback = defaultSetup();
  const source = fragment.startsWith('#') ? fragment.slice(1) : fragment;
  if (!source) return { setup: defaultSetup('water_settlement') };

  try {
    const encoded = new URLSearchParams(source).get('setup');
    if (!encoded) throw new Error('Missing setup payload');
    if (encoded.length > 1_000_000) throw new Error('Setup payload is too large');
    const setup = validateSharedSetup(JSON.parse(encoded));
    if (!setup) throw new Error('Invalid setup payload');
    return { setup };
  } catch {
    return {
      setup: fallback,
      message: 'This setup link is invalid or out of date. Safe defaults were loaded instead.',
    };
  }
}

/** Serialize only reproducible setup inputs; interventions and evolved fields are never included. */
export function serializeSetupFragment(setup: Setup): string {
  const safe = validateSharedSetup(setup);
  if (!safe) throw new RangeError('Cannot share an invalid simulation setup.');
  return `setup=${encodeURIComponent(JSON.stringify(safe))}`;
}

function validateTiles(value: unknown, model: ReturnType<typeof getPreset>['model']): TileConfig[] | undefined {
  if (!Array.isArray(value) || value.length === 0) return undefined;
  const fields = new Set(getFieldDescriptors(model).map(field => field.key));
  const ids = new Set<string>();
  const tiles: TileConfig[] = [];
  for (const tile of value) {
    if (!isRecord(tile) || typeof tile.id !== 'string' || !/^[a-zA-Z0-9_-]{1,80}$/.test(tile.id) || ids.has(tile.id)) return undefined;
    ids.add(tile.id);
    if (!Array.isArray(tile.layers) || !tile.layers.length) return undefined;
    const layers: TileConfig['layers'] = [];
    const seen = new Set<string>();
    for (const layer of tile.layers) {
      if (!isRecord(layer) || typeof layer.field !== 'string' || !fields.has(layer.field as FieldName) || seen.has(layer.field)) return undefined;
      if (typeof layer.opacity !== 'number' || !Number.isFinite(layer.opacity) || layer.opacity < 0 || layer.opacity > 1 || typeof layer.visible !== 'boolean') return undefined;
      seen.add(layer.field);
      layers.push({field:layer.field as FieldName,opacity:layer.opacity,visible:layer.visible});
    }
    if (typeof tile.paintField !== 'string' || !seen.has(tile.paintField)) return undefined;
    if (layers.length === 1 && tile.paintField !== layers[0].field) return undefined;
    tiles.push({id:tile.id,layers,paintField:tile.paintField as FieldName});
  }
  return tiles;
}
