import { describe, expect, it } from 'vitest';

import reference from './fixtures/reference.json';
import type { FieldName, Fields, Model, Setup } from '../src/contracts';
import { initializeFields, Simulation } from '../src/engine/simulation';

interface ReferenceCase {
  name: string;
  model: Model;
  setup: Setup;
  fields: Partial<Record<FieldName, number[]>>;
  dt: number;
  safe_dt: number;
  rate: number;
  steps: number;
  expected: Partial<Record<FieldName, number[]>>;
}

const fixture = reference as { version: number; cases: ReferenceCase[] };

function expectArraysClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  context: string,
): void {
  expect(actual.length, `${context} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const tolerance = 1e-9 + 1e-9 * Math.abs(expected[index]);
    expect(
      Math.abs(actual[index] - expected[index]),
      `${context}[${index}]: expected ${expected[index]}, received ${actual[index]}`,
    ).toBeLessThanOrEqual(tolerance);
  }
}

describe('Python/browser reference parity', () => {
  it('uses the versioned fixture contract', () => {
    expect(fixture.version).toBe(1);
    expect(fixture.cases.length).toBeGreaterThan(0);
  });

  for (const referenceCase of fixture.cases) {
    it(referenceCase.name, () => {
      const supplied = Object.fromEntries(
        Object.entries(referenceCase.fields).map(([name, values]) => [
          name,
          Float64Array.from(values),
        ]),
      ) as unknown as Fields;
      const simulation = new Simulation(referenceCase.setup, supplied);
      expect(simulation.snapshot().dt).toBeCloseTo(referenceCase.safe_dt, 12);
      expect(simulation.rateBound()).toBeCloseTo(referenceCase.rate, 12);
      for (let step = 0; step < referenceCase.steps; step += 1) {
        simulation.step(referenceCase.dt);
      }
      const actual = simulation.snapshot().fields;
      for (const [name, expected] of Object.entries(referenceCase.expected)) {
        expectArraysClose(actual[name as FieldName], expected, `${referenceCase.name}/${name}`);
      }
    });
  }

  for (const referenceCase of fixture.cases.filter(
    (candidate) => candidate.model === 'civilization' && candidate.steps === 1,
  )) {
    it(`initializer-${referenceCase.name}`, () => {
      const actual = initializeFields(
        referenceCase.setup.n,
        referenceCase.setup.seed,
        referenceCase.setup.boundary,
      );
      for (const [name, expected] of Object.entries(referenceCase.fields)) {
        expectArraysClose(actual[name as FieldName], expected, `initializer/${name}`);
      }
    });
  }
});
