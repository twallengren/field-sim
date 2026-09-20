import { describe, expect, it } from 'vitest';

import { defaultSetup } from '../src/catalog';
import { loadSetupFragment, serializeSetupFragment, validateSharedSetup } from '../src/setup';

describe('shared setup links', () => {
  it('round-trips every reproducible setup input', () => {
    const setup = defaultSetup('overshoot');
    setup.seed = 0xffff_ffff;
    setup.n = 128;
    setup.boundary = 'periodic';
    setup.parameters.erosion = 0.123;

    const fragment = serializeSetupFragment(setup);
    const loaded = loadSetupFragment(`#${fragment}`);

    expect(loaded).toEqual({ setup });
    expect(fragment).not.toContain('fields');
    expect(fragment).not.toContain('intervention');
  });

  it('uses a fresh safe default for malformed JSON', () => {
    const loaded = loadSetupFragment('#setup=%7Bbad');
    expect(loaded.setup).toEqual(defaultSetup());
    expect(loaded.setup).not.toBe(defaultSetup());
    expect(loaded.message).toMatch(/invalid|out of date/i);
  });

  it.each([
    ['version', { version: 3 }],
    ['preset', { preset: 'not-a-preset' }],
    ['seed', { seed: -1 }],
    ['resolution', { n: 48 }],
    ['boundary', { boundary: 'open' }],
  ])('rejects an invalid %s', (_label, patch) => {
    expect(validateSharedSetup({ ...defaultSetup(), ...patch })).toBeUndefined();
  });

  it('rejects missing, extra, non-finite, and out-of-range parameters', () => {
    const missing = defaultSetup();
    delete (missing.parameters as Partial<typeof missing.parameters>).erosion;
    expect(validateSharedSetup(missing)).toBeUndefined();

    const extra = defaultSetup() as unknown as { parameters: Record<string, number> };
    extra.parameters.untrusted = 1;
    expect(validateSharedSetup(extra)).toBeUndefined();

    const infinite = defaultSetup();
    infinite.parameters.dp = Number.POSITIVE_INFINITY;
    expect(validateSharedSetup(infinite)).toBeUndefined();

    const outside = defaultSetup();
    outside.parameters.erosion = 999;
    expect(validateSharedSetup(outside)).toBeUndefined();
  });

  it('does not expose caller-owned parameter objects', () => {
    const setup = defaultSetup();
    const validated = validateSharedSetup(setup)!;
    setup.parameters.dp = 0.2;
    expect(validated.parameters.dp).not.toBe(0.2);
  });
});

describe('version 2 field views', () => {
  it('round-trips arbitrary tiles and overlays without evolved state', () => {
    const setup = defaultSetup('water_settlement');
    setup.tiles = Array.from({length:12}, (_,index) => ({id:`tile-${index}`, layers:[{field:'water' as const,opacity:1,visible:true},{field:'population' as const,opacity:0.4,visible:true}],paintField:'population' as const}));
    expect(loadSetupFragment(serializeSetupFragment(setup)).setup).toEqual(setup);
  });
  it('keeps legacy setup meanings when adding views', () => {
    const legacy = defaultSetup('overshoot');
    const upgraded = {...legacy,version:2 as const,tiles:[{id:'one',layers:[{field:'population' as const,opacity:1,visible:true}],paintField:'population' as const}]};
    expect(validateSharedSetup(upgraded)?.parameters).toEqual(legacy.parameters);
    expect(validateSharedSetup(legacy)).toEqual(legacy);
  });
  it('rejects unavailable fields, duplicate IDs, bad opacity, and invalid paint targets', () => {
    const setup = defaultSetup('water_settlement');
    const tile = {id:'one',layers:[{field:'water',opacity:1,visible:true}],paintField:'water'};
    for (const tiles of [[],[tile,tile],[{...tile,layers:[{field:'infrastructure',opacity:1,visible:true}]}],[{...tile,layers:[{field:'water',opacity:2,visible:true}]}],[{...tile,paintField:'population'}]]) {
      expect(validateSharedSetup({...setup,tiles})).toBeUndefined();
    }
  });
});
