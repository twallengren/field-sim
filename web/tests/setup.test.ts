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
    ['version', { version: 2 }],
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
