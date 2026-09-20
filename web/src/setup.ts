import { defaultSetup, parameterDefinitions, presets } from './catalog';
import type { Parameters, Setup } from './contracts';

export interface LoadedSetup {
  setup: Setup;
  message?: string;
}

const VALID_RESOLUTIONS = new Set([32, 64, 128]);
const VALID_BOUNDARIES = new Set(['neumann', 'periodic']);
const PRESET_IDS = new Set(presets.map((preset) => preset.id));
const PARAMETER_KEYS = new Set(parameterDefinitions.map((definition) => definition.key));

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Validate untrusted setup data and return a fresh, contract-shaped object. */
export function validateSharedSetup(value: unknown): Setup | undefined {
  if (!isRecord(value) || value.version !== 1 || typeof value.preset !== 'string') return undefined;
  if (!PRESET_IDS.has(value.preset)) return undefined;
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

  return {
    version: 1,
    preset: value.preset,
    seed: value.seed as number,
    n: value.n as number,
    boundary: value.boundary as Setup['boundary'],
    parameters,
  };
}

/** Read `#setup=<encoded JSON>`, returning safe defaults for absent or bad input. */
export function loadSetupFragment(fragment: string): LoadedSetup {
  const fallback = defaultSetup();
  const source = fragment.startsWith('#') ? fragment.slice(1) : fragment;
  if (!source) return { setup: fallback };

  try {
    const encoded = new URLSearchParams(source).get('setup');
    if (!encoded) throw new Error('Missing setup payload');
    if (encoded.length > 10_000) throw new Error('Setup payload is too large');
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
