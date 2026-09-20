import { describe, expect, it } from 'vitest';

import { defaultSetup } from '../src/catalog';
import type { Fields, Parameters, Setup } from '../src/contracts';
import { Simulation } from '../src/engine/simulation';

function ecologySetup(
  n = 4,
  boundary: Setup['boundary'] = 'neumann',
  parameterPatch: Partial<Parameters> = {},
): Setup {
  const base = defaultSetup('water_settlement');
  return {
    ...base,
    n,
    boundary,
    parameters: { ...base.parameters, ...parameterPatch } as Parameters,
  };
}

function ecologyFields(n: number, values: Partial<Record<string, number[]>> = {}): Fields {
  const size = n * n;
  const make = (name: string, fallback: number): Float64Array => Float64Array.from(
    values[name] ?? new Array<number>(size).fill(fallback),
  );
  return {
    population: make('population', 0),
    food: make('food', 0),
    water: make('water', 0),
    soil: make('soil', 1),
    fertility: make('fertility', 1),
    waterSources: make('waterSources', 0),
    cultivation: make('cultivation', 0),
  };
}

function transportOnly(parameters: Parameters): Parameters {
  return {
    ...parameters,
    growth: 0,
    yield: 0,
    waterConsumption: 0,
    harvestWaterCost: 0,
    consumption: 0,
    spoilage: 0,
    replenishmentRate: 0,
    soilRecovery: 0,
    erosion: 0,
    settlementErosion: 0,
  };
}

describe('ecology transport and sources', () => {
  it('conserves every transported field and stays non-negative', () => {
    const n = 4;
    const population = new Array<number>(n * n).fill(0);
    const food = new Array<number>(n * n).fill(0);
    const water = new Array<number>(n * n).fill(0);
    population[n - 1] = 1;
    food[0] = 8;
    water[0] = 4;
    const base = ecologySetup(n, 'periodic');
    const parameters = transportOnly({
      ...base.parameters,
      dp: 0.04,
      df: 0.06,
      dw: 0.12,
      chiFood: 0.08,
      chiWater: 0.12,
    });
    const simulation = new Simulation(
      { ...base, parameters },
      ecologyFields(n, { population, food, water }),
    );
    const before = simulation.snapshot();

    for (let step = 0; step < 100; step += 1) simulation.step();
    const after = simulation.snapshot();

    for (const name of ['population', 'food', 'water'] as const) {
      expect(after.metrics[name].total).toBeCloseTo(before.metrics[name].total, 12);
      expect(after.fields[name].every((value) => value >= 0)).toBe(true);
    }
    expect(after.truncatedMass).toBe(0);
    expect(after.waterBudget?.recharged).toBe(0);
    expect(after.waterBudget?.domesticUse).toBe(0);
    expect(after.waterBudget?.agriculturalUse).toBe(0);
    expect(after.waterBudget?.residual).toBeCloseTo(0, 12);
  });

  it('evaluates coupled sources from the old state and closes the water budget', () => {
    const n = 2;
    const base = ecologySetup(n);
    const simulation = new Simulation(base, ecologyFields(n, {
      population: [0.5, 0.5, 0.5, 0.5],
      food: [1, 1, 1, 1],
      water: [0.8, 0.8, 0.8, 0.8],
      soil: [0.7, 0.7, 0.7, 0.7],
      fertility: [0.9, 0.9, 0.9, 0.9],
      waterSources: [0.4, 0.4, 0.4, 0.4],
      cultivation: [1, 1, 1, 1], // ignored: cultivation is always derived
    }));
    const dt = 0.01;
    const p = base.parameters;
    const cultivation = 0.5 / (0.5 + p.cultivationScale);
    const waterSaturation = 0.8 / 1.8;
    const harvest = p.yield * cultivation * 0.9 * 0.7 * waterSaturation;
    const recharge = p.replenishmentRate * 0.4 * (1 - 0.8 / p.sourceCapacity);
    const domesticUse = p.waterConsumption * 0.5 * waterSaturation;
    const agriculturalUse = p.harvestWaterCost * harvest;

    simulation.step(dt);
    const snapshot = simulation.snapshot();
    const budget = snapshot.waterBudget;
    expect(snapshot.fields.water[0]).toBeCloseTo(
      0.8 + dt * (recharge - domesticUse - agriculturalUse),
      15,
    );
    expect(budget?.initial).toBeCloseTo(80, 14);
    expect(budget?.recharged).toBeCloseTo(dt * recharge * 100, 14);
    expect(budget?.domesticUse).toBeCloseTo(dt * domesticUse * 100, 14);
    expect(budget?.agriculturalUse).toBeCloseTo(dt * agriculturalUse * 100, 14);
    expect(budget?.residual).toBeCloseTo(0, 12);
  });

  it('keeps a long coupled run positive and within bounded-field limits', () => {
    const simulation = new Simulation(ecologySetup(12, 'periodic'));
    for (let step = 0; step < 200; step += 1) simulation.step();
    const snapshot = simulation.snapshot();

    for (const name of ['population', 'food', 'water', 'soil', 'fertility', 'waterSources', 'cultivation']) {
      expect(snapshot.fields[name].every((value) => Number.isFinite(value) && value >= 0)).toBe(true);
    }
    for (const name of ['soil', 'waterSources', 'cultivation']) {
      expect(snapshot.fields[name].every((value) => value <= 1)).toBe(true);
    }
    expect(snapshot.truncatedMass).toBe(0);
    expect(snapshot.waterBudget?.residual).toBeCloseTo(0, 9);
  });
});

describe('ecology derived fields, painting, and snapshots', () => {
  it('recomputes cultivation after population changes and rejects direct painting', () => {
    const simulation = new Simulation(
      ecologySetup(2),
      ecologyFields(2, {
        population: [0.5, 0.5, 0.5, 0.5],
        cultivation: [1, 1, 1, 1],
      }),
    );
    const before = simulation.snapshot();
    expect(before.fields.cultivation[0]).toBeCloseTo(0.5 / 1.1, 15);
    expect(before.fields.infrastructure).toBeUndefined();

    expect(() => simulation.paint({
      field: 'cultivation', x: 0.5, y: 0.5, radius: 1, amount: 0.2,
    })).toThrow(/derived/);
    simulation.paint({ field: 'population', x: 0.5, y: 0.5, radius: 1, amount: 0.3 });
    const after = simulation.snapshot();
    expect(after.fields.population[0]).toBeCloseTo(0.8, 15);
    expect(after.fields.cultivation[0]).toBeCloseTo(0.8 / 1.4, 15);
  });

  it('counts actual water paint in the budget while source paint does not inject water', () => {
    const simulation = new Simulation(
      ecologySetup(2),
      ecologyFields(2, {
        water: [0.8, 0.8, 0.8, 0.8],
        waterSources: [0.4, 0.4, 0.4, 0.4],
      }),
    );
    simulation.paint({ field: 'water', x: 0.5, y: 0.5, radius: 1, amount: 0.2 });
    const waterPainted = simulation.snapshot();
    expect(waterPainted.waterBudget?.interventions).toBeCloseTo(20, 14);
    expect(waterPainted.waterBudget?.current).toBeCloseTo(100, 14);
    expect(waterPainted.waterBudget?.residual).toBeCloseTo(0, 12);

    simulation.paint({ field: 'waterSources', x: 0.5, y: 0.5, radius: 1, amount: 1 });
    const sourcePainted = simulation.snapshot();
    expect(sourcePainted.fields.waterSources.every((value) => value === 1)).toBe(true);
    expect(sourcePainted.waterBudget).toEqual(waterPainted.waterBudget);
    expect(sourcePainted.interventionMass.waterSources).toBeCloseTo(60, 14);
  });
});

describe('ecology extremes and recovery', () => {
  it.each(['neumann', 'periodic'] as const)('balances transport under %s boundaries', boundary => {
    const setup = ecologySetup(8,boundary);
    setup.parameters = transportOnly(setup.parameters);
    const simulation = new Simulation(setup);
    const initial = simulation.snapshot().metrics.water.total;
    simulation.advance(3,1000);
    expect(simulation.snapshot().metrics.water.total).toBeCloseTo(initial,10);
  });
  it('approaches source capacity without creating an overshoot in a dry source', () => {
    const setup = ecologySetup(2,'neumann',{replenishmentRate:1,sourceCapacity:0.1,dw:0});
    const simulation = new Simulation(setup,ecologyFields(2,{waterSources:[1,1,1,1]}));
    for (let step=0;step<30;step++) {
      simulation.step();
      expect(simulation.snapshot().metrics.water.max).toBeLessThanOrEqual(0.1);
    }
    expect(simulation.snapshot().metrics.water.mean).toBeCloseTo(0.1,8);
  });
  it('survives strong population, water, and fertility interventions without flooring', () => {
    const simulation = new Simulation(ecologySetup(16));
    for (const field of ['population','water','fertility'] as const) simulation.paint({field,x:0.5,y:0.5,radius:0.08,amount:1000});
    for(let step=0;step<20;step++) simulation.step();
    const snapshot=simulation.snapshot();
    expect(snapshot.truncatedMass).toBe(0);
    for (const values of Object.values(snapshot.fields)) expect(values.every(value=>Number.isFinite(value)&&value>=0)).toBe(true);
    expect(Math.abs(snapshot.waterBudget!.residual)).toBeLessThan(1e-9);
  });
  it('stored food alone does not change fallow soil recovery', () => {
    const fields=ecologyFields(2,{water:[1,1,1,1],soil:[0.25,0.25,0.25,0.25]});
    const a=new Simulation(ecologySetup(2),fields);
    const b=new Simulation(ecologySetup(2),{...fields,food:new Float64Array(4).fill(100)});
    a.step(0.01); b.step(0.01);
    expect(a.snapshot().fields.soil).toEqual(b.snapshot().fields.soil);
    expect(a.snapshot().fields.soil[0]).toBeGreaterThan(0.25);
  });
});

it('moves population toward water even when food attraction and growth are disabled', () => {
  const setup = ecologySetup(2,'neumann');
  setup.parameters = {...transportOnly(setup.parameters),dp:0,df:0,dw:0,chiFood:0,chiWater:0.12};
  const simulation = new Simulation(setup,ecologyFields(2,{population:[1,1,1,1],water:[2,0,2,0]}));
  simulation.step(0.1);
  const values=simulation.snapshot().fields.population;
  expect(values[0]).toBeGreaterThan(1);
  expect(values[1]).toBeLessThan(1);
  expect(simulation.snapshot().metrics.population.total).toBeCloseTo(100,12);
});
