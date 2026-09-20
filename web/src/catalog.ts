import raw from '../../src/fieldsim/catalog.json';
import type { Parameters, ParameterKey, Preset, Setup } from './contracts';
export const catalog = raw;
export const presets = raw.presets as Preset[];
export const parameterDefinitions = raw.parameters as { key: ParameterKey; label: string; default: number; min: number; max: number; step: number }[];
export function getPreset(id: string): Preset { const p = presets.find(p => p.id === id); if (!p) throw new Error(`Unknown experiment: ${id}`); return p; }
export function defaultSetup(id = 'settlement'): Setup { return {version: 1, preset: id, seed: 0, n: 64, boundary: 'neumann', parameters: {...getPreset(id).parameters} as Parameters}; }
