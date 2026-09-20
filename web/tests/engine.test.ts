import { describe, expect, it } from 'vitest';

import { defaultSetup } from '../src/catalog';
import type { Fields, ParameterKey, Parameters, Setup } from '../src/contracts';
import { Simulation } from '../src/engine/simulation';
import { processCommand } from '../src/engine/worker';

const parameterKeys: ParameterKey[] = [
  'dp', 'df', 'chiFood', 'chiInfra', 'growth', 'regrowth', 'consumption',
  'investment', 'infraDecay', 'infraBoost', 'foodCost', 'soilRecovery', 'erosion',
];

function setup(
  preset: Setup['preset'],
  n: number,
  boundary: Setup['boundary'],
  parameterPatch: Partial<Parameters> = {},
): Setup {
  const result = defaultSetup(preset);
  return {
    ...result,
    n,
    boundary,
    parameters: { ...result.parameters, ...parameterPatch } as Parameters,
  };
}

function fields(n: number, values: Partial<Record<keyof Fields, number[]>> = {}): Fields {
  const size = n * n;
  const make = (name: keyof Fields, fallback: number): Float64Array => {
    const supplied = values[name];
    return supplied === undefined
      ? new Float64Array(size).fill(fallback)
      : Float64Array.from(supplied);
  };
  return {
    population: make('population', 0),
    food: make('food', 0),
    infrastructure: make('infrastructure', 0),
    soil: make('soil', 1),
    fertility: make('fertility', 0),
  };
}

function withoutDynamics(base: Parameters): Parameters {
  const result = { ...base };
  for (const key of parameterKeys) result[key] = 0;
  return result;
}

describe('finite-volume operators', () => {
  it('uses conservative Neumann and periodic diffusion stencils', () => {
    const n = 3;
    const initial = fields(n, { population: [1, 0, 0, 0, 0, 0, 0, 0, 0] });
    const base = setup('chemotaxis_demo', n, 'neumann');
    const parameters = { ...withoutDynamics(base.parameters), dp: 0.1 };
    const neumann = new Simulation({ ...base, parameters }, initial);
    const periodic = new Simulation({ ...base, boundary: 'periodic', parameters }, initial);

    neumann.step(0.01);
    periodic.step(0.01);
    const nState = neumann.snapshot();
    const pState = periodic.snapshot();
    const coefficient = 0.01 * 0.1 / ((10 / n) ** 2);

    expect(nState.fields.population[0]).toBeCloseTo(1 - 2 * coefficient, 15);
    expect(pState.fields.population[0]).toBeCloseTo(1 - 4 * coefficient, 15);
    expect(nState.metrics.population.total).toBeCloseTo(100 / 9, 14);
    expect(pState.metrics.population.total).toBeCloseTo(100 / 9, 14);
    expect(nState.truncatedMass).toBe(0);
    expect(pState.truncatedMass).toBe(0);
  });

  it('moves population up a gradient across the periodic seam without losing mass', () => {
    const n = 4;
    const population = new Array<number>(n * n).fill(0);
    population[1 * n + 3] = 1;
    const food = new Array<number>(n * n).fill(0);
    for (let row = 0; row < n; row += 1) food[row * n] = 10;
    const base = setup('chemotaxis_demo', n, 'periodic');
    const parameters = { ...withoutDynamics(base.parameters), chiFood: 0.3 };
    const simulation = new Simulation(
      { ...base, parameters },
      fields(n, { population, food }),
    );
    const mass = simulation.snapshot().metrics.population.total;

    simulation.step(0.01);
    const snapshot = simulation.snapshot();

    expect(snapshot.fields.population[1 * n]).toBeGreaterThan(0);
    expect(snapshot.fields.population[1 * n + 3]).toBeLessThan(1);
    expect(snapshot.metrics.population.total).toBeCloseTo(mass, 14);
    expect(snapshot.fields.population.every((value) => value >= 0)).toBe(true);
  });
});

describe('model integration and adaptive stepping', () => {
  it('evaluates all civilization right-hand sides from the old state', () => {
    const n = 2;
    const base = setup('collective_investment', n, 'neumann', {
      dp: 0,
      df: 0,
      chiFood: 0,
      chiInfra: 0,
    });
    const initial = fields(n, {
      population: [0.5, 0.5, 0.5, 0.5],
      food: [1, 1, 1, 1],
      infrastructure: [0.25, 0.25, 0.25, 0.25],
      soil: [0.8, 0.8, 0.8, 0.8],
      fertility: [2, 2, 2, 2],
    });
    const simulation = new Simulation(base, initial);
    const dt = 0.01;
    simulation.step(dt);
    const snapshot = simulation.snapshot();
    const p = base.parameters;
    const build = p.investment * 0.5 * 1 / 2;
    const populationRhs = p.growth * 0.5 * (1 - 0.5) / (1 + 0.5 + 1e-6);
    const foodRhs = p.regrowth * (2 * 0.8 * (1 + p.infraBoost * 0.25 / 1.25) - 1)
      - p.consumption * 0.5 - p.foodCost * build;
    const infrastructureRhs = build - p.infraDecay * 0.25;
    const soilRhs = p.soilRecovery * 0.2 - p.erosion * 0.5 * 0.8;

    expect(snapshot.fields.population[0]).toBeCloseTo(0.5 + dt * populationRhs, 15);
    expect(snapshot.fields.food[0]).toBeCloseTo(1 + dt * foodRhs, 15);
    expect(snapshot.fields.infrastructure[0]).toBeCloseTo(0.25 + dt * infrastructureRhs, 15);
    expect(snapshot.fields.soil[0]).toBeCloseTo(0.8 + dt * soilRhs, 15);
  });

  it('rejects unsafe overrides and caps advance at the exact requested duration', () => {
    const base = setup('chemotaxis_demo', 2, 'neumann');
    const parameters = withoutDynamics(base.parameters);
    const simulation = new Simulation({ ...base, parameters }, fields(2));

    expect(() => simulation.step(0.100001)).toThrow(/safe bound/);
    simulation.advance(0.25, 10);
    const snapshot = simulation.snapshot();
    expect(snapshot.time).toBeCloseTo(0.25, 15);
    expect(snapshot.step).toBe(3);
    expect(snapshot.dt).toBeCloseTo(0.05, 15);

    const limited = new Simulation({ ...base, parameters }, fields(2));
    limited.advance(1, 2);
    expect(limited.snapshot().step).toBe(2);
    expect(limited.snapshot().time).toBeCloseTo(0.2, 15);
  });

  it('tightens the adaptive timestep after a population brush spike', () => {
    const simulation = new Simulation(setup('agriculture', 8, 'neumann'));
    const before = simulation.snapshot().dt;
    simulation.paint({ field: 'population', x: 0.5, y: 0.5, radius: 0.2, amount: 1e5 });
    const after = simulation.snapshot().dt;
    expect(after).toBeLessThan(before / 100);
  });
});

describe('state, brushes, and reset behavior', () => {
  it('returns detached snapshots and validates setup, parameters, and state', () => {
    const base = setup('settlement', 4, 'neumann');
    const simulation = new Simulation(base);
    const snapshot = simulation.snapshot();
    const original = snapshot.fields.population[0];
    snapshot.fields.population[0] = 12345;
    expect(simulation.snapshot().fields.population[0]).toBe(original);

    expect(() => new Simulation({ ...base, n: 129 })).toThrow(/\[2, 128\]/);
    expect(() => new Simulation(
      { ...base, parameters: { ...base.parameters, dp: -1 } },
    )).toThrow(/Parameter dp/);
    const invalid = fields(4);
    invalid.food[3] = Number.NaN;
    expect(() => new Simulation(base, invalid)).toThrow(/non-finite/);
  });

  it('wraps periodic brushes, clamps soil, and keeps intervention mass separate', () => {
    const n = 4;
    const simulation = new Simulation(
      setup('settlement', n, 'periodic'),
      fields(n, { soil: new Array<number>(n * n).fill(0.9) }),
    );
    simulation.paint({ field: 'soil', x: 0.99, y: 0.125, radius: 0.2, amount: 0.5 });
    const snapshot = simulation.snapshot();

    expect(snapshot.fields.soil[0]).toBe(1); // wrapped from the opposite edge
    expect(snapshot.fields.soil[3]).toBe(1);
    expect(snapshot.interventionMass.soil).toBeCloseTo(2 * 0.1 * (10 / n) ** 2, 14);
    expect(snapshot.truncatedMass).toBe(0);
  });

  it('rejects a brush that would destroy the timestep bound without changing state', () => {
    const base = setup('collective_investment', 2, 'neumann', {
      dp: 0.3,
      df: 0.3,
      chiFood: 0.8,
      chiInfra: 0.5,
      growth: 1,
      regrowth: 1,
      consumption: 0.8,
      investment: 0.5,
      infraDecay: 0.2,
      infraBoost: 6,
      foodCost: 2,
      soilRecovery: 0.2,
      erosion: 0.3,
    });
    const simulation = new Simulation(base, fields(2));

    expect(() => simulation.paint({
      field: 'population',
      x: 0.5,
      y: 0.5,
      radius: 1,
      amount: Number.MAX_VALUE,
    })).toThrow(/invalid rate/);
    const snapshot = simulation.snapshot();
    expect(snapshot.fields.population.every((value) => value === 0)).toBe(true);
    expect(snapshot.interventionMass.population).toBeUndefined();
  });

  it('worker reset remains available after an error and batches are bounded', () => {
    const valid = setup('chemotaxis_demo', 2, 'neumann');
    const invalid = processCommand({
      type: 'init',
      generation: 40,
      setup: { ...valid, n: 129 },
    });
    expect(invalid?.type).toBe('error');

    const reset = processCommand({ type: 'init', generation: 41, setup: valid });
    expect(reset?.type).toBe('snapshot');
    const advanced = processCommand({
      type: 'advance',
      generation: 41,
      duration: 100,
      maxSteps: 10_000,
    });
    expect(advanced?.type).toBe('snapshot');
    if (advanced?.type === 'snapshot') expect(advanced.snapshot.step).toBeLessThanOrEqual(100);
    expect(processCommand({ type: 'step', generation: 40 })).toBeUndefined();
  });
});
