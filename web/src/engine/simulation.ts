import { catalog, getPreset, parameterDefinitions } from '../catalog';
import type {
  Boundary,
  Brush,
  FieldMetric,
  FieldName,
  Fields,
  Model,
  Parameters,
  Snapshot,
  Setup,
} from '../contracts';

const FIELD_NAMES: readonly FieldName[] = [
  'population',
  'food',
  'infrastructure',
  'soil',
  'fertility',
];

const DOMAIN_LENGTH = catalog.domainLength;
const EPSILON = catalog.epsilon;
const FOOD_HALF_SATURATION = catalog.foodHalfSaturation;
const INFRA_HALF_SATURATION = catalog.infrastructureHalfSaturation;
const SAFETY = 0.8;
const MAX_DT = 0.1;

type MutableFields = Record<FieldName, Float64Array>;

interface Bump {
  x: number;
  y: number;
  amplitude: number;
  sigma: number;
}

/** Mulberry32 with the uint32 coercions made explicit for cross-language parity. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= (t + Math.imul(t ^ (t >>> 7), t | 61)) | 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeBumps(
  random: () => number,
  count: number,
  amplitudeRange: readonly [number, number],
  sigmaRange: readonly [number, number],
): Bump[] {
  const bumps: Bump[] = [];
  for (let i = 0; i < count; i += 1) {
    const x = DOMAIN_LENGTH * random();
    const y = DOMAIN_LENGTH * random();
    const amplitude = amplitudeRange[0] + (amplitudeRange[1] - amplitudeRange[0]) * random();
    const sigma = sigmaRange[0] + (sigmaRange[1] - sigmaRange[0]) * random();
    random(); // Reserved so future bump attributes do not change later fields.
    bumps.push({ x, y, amplitude, sigma });
  }
  return bumps;
}

function distance(a: number, b: number, boundary: Boundary): number {
  const direct = Math.abs(a - b);
  return boundary === 'periodic' ? Math.min(direct, DOMAIN_LENGTH - direct) : direct;
}

function evaluateBumps(
  n: number,
  boundary: Boundary,
  bumps: readonly Bump[],
  floor: number,
): Float64Array {
  const values = new Float64Array(n * n);
  const dx = DOMAIN_LENGTH / n;
  for (let row = 0; row < n; row += 1) {
    const y = (row + 0.5) * dx;
    for (let column = 0; column < n; column += 1) {
      const x = (column + 0.5) * dx;
      let value = floor;
      for (const bump of bumps) {
        const deltaX = distance(x, bump.x, boundary);
        const deltaY = distance(y, bump.y, boundary);
        value += bump.amplitude * Math.exp(
          -(deltaX * deltaX + deltaY * deltaY) / (2 * bump.sigma * bump.sigma),
        );
      }
      values[row * n + column] = value;
    }
  }
  return values;
}

/** Deterministic initial conditions shared by every browser model. */
export function initializeFields(n: number, seed: number, boundary: Boundary): Fields {
  const random = mulberry32(seed);
  const populationBumps = makeBumps(random, 6, [0.4, 1.2], [0.3, 0.8]);
  const foodBumps = makeBumps(random, 8, [0.5, 1.5], [0.5, 1.2]);
  const fertilityBumps = makeBumps(random, 8, [0.5, 1.5], [0.6, 1.5]);
  const size = n * n;

  return {
    population: evaluateBumps(n, boundary, populationBumps, 0.05),
    food: evaluateBumps(n, boundary, foodBumps, 0.2),
    infrastructure: new Float64Array(size),
    soil: new Float64Array(size).fill(1),
    fertility: evaluateBumps(n, boundary, fertilityBumps, 0.25),
  };
}

function copyAndValidateFields(defaults: Fields, supplied: Fields | undefined, size: number): MutableFields {
  const result = {} as MutableFields;
  const partial = supplied as Partial<Record<FieldName, ArrayLike<number>>> | undefined;

  for (const name of FIELD_NAMES) {
    const source = partial?.[name] ?? defaults[name];
    if (source.length !== size) {
      throw new RangeError(`Field ${name} must contain ${size} values; received ${source.length}.`);
    }
    const values = Float64Array.from(source);
    for (let index = 0; index < values.length; index += 1) {
      const value = values[index];
      if (!Number.isFinite(value)) {
        throw new RangeError(`Field ${name} contains a non-finite value at index ${index}.`);
      }
      if (value < 0 || (name === 'soil' && value > 1)) {
        throw new RangeError(
          `Field ${name} contains an out-of-range value ${value} at index ${index}.`,
        );
      }
    }
    result[name] = values;
  }
  return result;
}

function validateParameters(parameters: Parameters): Parameters {
  const copy = {} as Parameters;
  for (const definition of parameterDefinitions) {
    const value = parameters[definition.key];
    if (!Number.isFinite(value)) {
      throw new RangeError(`Parameter ${definition.key} must be finite.`);
    }
    if (value < definition.min || value > definition.max) {
      throw new RangeError(
        `Parameter ${definition.key} must be in [${definition.min}, ${definition.max}]; received ${value}.`,
      );
    }
    copy[definition.key] = value;
  }
  return copy;
}

function validateSetup(setup: Setup): { model: Model; parameters: Parameters } {
  if (setup.version !== 1) throw new RangeError(`Unsupported setup version ${String(setup.version)}.`);
  if (!Number.isInteger(setup.n) || setup.n < 2 || setup.n > 128) {
    throw new RangeError(`Grid size n must be an integer in [2, 128]; received ${setup.n}.`);
  }
  if (setup.boundary !== 'neumann' && setup.boundary !== 'periodic') {
    throw new RangeError(`Unsupported boundary condition ${String(setup.boundary)}.`);
  }
  // initializeFields deliberately applies ToUint32 (`>>> 0`), matching the
  // shared seed-coercion contract for negative and fractional finite seeds.
  if (!Number.isFinite(setup.seed)) throw new RangeError('Seed must be finite.');
  const preset = getPreset(setup.preset);
  return { model: preset.model, parameters: validateParameters(setup.parameters) };
}

function addDiffusion(
  rhs: Float64Array,
  values: Float64Array,
  alpha: number,
  n: number,
  inverseDxSquared: number,
  boundary: Boundary,
): void {
  if (alpha === 0) return;
  for (let row = 0; row < n; row += 1) {
    for (let column = 0; column < n; column += 1) {
      const index = row * n + column;
      const center = values[index];
      let difference = 0;
      if (column > 0) difference += values[index - 1] - center;
      else if (boundary === 'periodic') difference += values[index + n - 1] - center;
      if (column + 1 < n) difference += values[index + 1] - center;
      else if (boundary === 'periodic') difference += values[index - n + 1] - center;
      if (row > 0) difference += values[index - n] - center;
      else if (boundary === 'periodic') difference += values[index + n * (n - 1)] - center;
      if (row + 1 < n) difference += values[index + n] - center;
      else if (boundary === 'periodic') difference += values[index - n * (n - 1)] - center;
      rhs[index] += alpha * inverseDxSquared * difference;
    }
  }
}

/**
 * Add -div(chi * density * grad(attractant)) to rhs using donor-cell faces.
 * Every face update is equal and opposite, including the periodic seam.
 */
function addGradientFlux(
  rhs: Float64Array,
  density: Float64Array,
  attractant: Float64Array,
  chi: number,
  n: number,
  inverseDx: number,
  boundary: Boundary,
): void {
  if (chi === 0) return;
  const addFace = (left: number, right: number): void => {
    const velocity = chi * (attractant[right] - attractant[left]) * inverseDx;
    const flux = velocity >= 0 ? velocity * density[left] : velocity * density[right];
    const change = flux * inverseDx;
    rhs[left] -= change;
    rhs[right] += change;
  };

  for (let row = 0; row < n; row += 1) {
    const offset = row * n;
    for (let column = 0; column + 1 < n; column += 1) {
      addFace(offset + column, offset + column + 1);
    }
    if (boundary === 'periodic') addFace(offset + n - 1, offset);
  }
  for (let column = 0; column < n; column += 1) {
    for (let row = 0; row + 1 < n; row += 1) {
      addFace(row * n + column, (row + 1) * n + column);
    }
    if (boundary === 'periodic') addFace((n - 1) * n + column, column);
  }
}

function gradientFluxRate(
  attractant: Float64Array,
  chi: number,
  n: number,
  inverseDx: number,
  boundary: Boundary,
): number {
  if (chi === 0) return 0;
  const outflow = new Float64Array(n * n);
  const addFace = (left: number, right: number): void => {
    const velocity = chi * (attractant[right] - attractant[left]) * inverseDx;
    if (velocity >= 0) outflow[left] += velocity * inverseDx;
    else outflow[right] -= velocity * inverseDx;
  };

  for (let row = 0; row < n; row += 1) {
    const offset = row * n;
    for (let column = 0; column + 1 < n; column += 1) {
      addFace(offset + column, offset + column + 1);
    }
    if (boundary === 'periodic') addFace(offset + n - 1, offset);
  }
  for (let column = 0; column < n; column += 1) {
    for (let row = 0; row + 1 < n; row += 1) {
      addFace(row * n + column, (row + 1) * n + column);
    }
    if (boundary === 'periodic') addFace((n - 1) * n + column, column);
  }

  let maximum = 0;
  for (const rate of outflow) maximum = Math.max(maximum, rate);
  return maximum;
}

function maximum(values: Float64Array): number {
  let result = 0;
  for (const value of values) result = Math.max(result, value);
  return result;
}

export class Simulation {
  private readonly n: number;
  private readonly boundary: Boundary;
  private readonly model: Model;
  private readonly dx: number;
  private readonly cellArea: number;
  private parameters: Parameters;
  private fields: MutableFields;
  private currentTime = 0;
  private stepCount = 0;
  private lastDt: number;
  private cumulativeTruncatedMass = 0;
  private readonly cumulativeInterventionMass: Partial<Record<FieldName, number>> = {};

  constructor(setup: Setup, initialFields?: Fields) {
    const validated = validateSetup(setup);
    this.n = setup.n;
    this.boundary = setup.boundary;
    this.model = validated.model;
    this.parameters = validated.parameters;
    this.dx = DOMAIN_LENGTH / this.n;
    this.cellArea = this.dx * this.dx;
    const defaults = initializeFields(this.n, setup.seed, this.boundary);
    this.fields = copyAndValidateFields(defaults, initialFields, this.n * this.n);
    this.lastDt = this.stableTimeStep();
  }

  step(dtOverride?: number): void {
    const safeDt = this.stableTimeStep();
    let dt = safeDt;
    if (dtOverride !== undefined) {
      if (!Number.isFinite(dtOverride) || dtOverride <= 0) {
        throw new RangeError(`Timestep must be finite and positive; received ${dtOverride}.`);
      }
      const tolerance = Math.max(1e-15, safeDt * 32 * Number.EPSILON);
      if (dtOverride > safeDt + tolerance) {
        throw new RangeError(
          `Requested timestep ${dtOverride} exceeds the current safe bound ${safeDt}.`,
        );
      }
      dt = dtOverride;
    }

    const { population, food, infrastructure, soil, fertility } = this.fields;
    const size = this.n * this.n;
    const populationRhs = new Float64Array(size);
    const foodRhs = new Float64Array(size);
    const infrastructureRhs = new Float64Array(size);
    const soilRhs = new Float64Array(size);
    const inverseDx = 1 / this.dx;
    const inverseDxSquared = inverseDx * inverseDx;
    const p = this.parameters;

    addDiffusion(populationRhs, population, p.dp, this.n, inverseDxSquared, this.boundary);
    addDiffusion(foodRhs, food, p.df, this.n, inverseDxSquared, this.boundary);
    addGradientFlux(
      populationRhs,
      population,
      food,
      p.chiFood,
      this.n,
      inverseDx,
      this.boundary,
    );

    if (this.model === 'civilization') {
      addGradientFlux(
        populationRhs,
        population,
        infrastructure,
        p.chiInfra,
        this.n,
        inverseDx,
        this.boundary,
      );
      for (let index = 0; index < size; index += 1) {
        const populationValue = population[index];
        const foodValue = food[index];
        const infrastructureValue = infrastructure[index];
        const soilValue = soil[index];
        const build = p.investment * populationValue * foodValue
          / (FOOD_HALF_SATURATION + foodValue);
        populationRhs[index] += p.growth * populationValue * (foodValue - populationValue)
          / (foodValue + populationValue + EPSILON);
        foodRhs[index] += p.regrowth * (
          fertility[index] * soilValue
          * (1 + p.infraBoost * infrastructureValue
            / (INFRA_HALF_SATURATION + infrastructureValue))
          - foodValue
        ) - p.consumption * populationValue * foodValue - p.foodCost * build;
        infrastructureRhs[index] = build - p.infraDecay * infrastructureValue;
        soilRhs[index] = p.soilRecovery * (1 - soilValue)
          - p.erosion * populationValue * soilValue;
      }
    } else if (this.model === 'agriculture') {
      for (let index = 0; index < size; index += 1) {
        const populationValue = population[index];
        const foodValue = food[index];
        populationRhs[index] += p.growth * populationValue * (foodValue - populationValue)
          / (foodValue + populationValue + EPSILON);
        foodRhs[index] += p.regrowth * (fertility[index] - foodValue)
          - p.consumption * populationValue * foodValue;
      }
    }

    const next = {
      population: new Float64Array(size),
      food: new Float64Array(size),
      infrastructure: this.model === 'civilization'
        ? new Float64Array(size)
        : infrastructure,
      soil: this.model === 'civilization' ? new Float64Array(size) : soil,
      fertility,
    } satisfies MutableFields;

    for (let index = 0; index < size; index += 1) {
      next.population[index] = population[index] + dt * populationRhs[index];
      next.food[index] = food[index] + dt * foodRhs[index];
      if (this.model === 'civilization') {
        next.infrastructure[index] = infrastructure[index] + dt * infrastructureRhs[index];
        next.soil[index] = soil[index] + dt * soilRhs[index];
      }
    }

    const dynamic: readonly FieldName[] = this.model === 'civilization'
      ? ['population', 'food', 'infrastructure', 'soil']
      : ['population', 'food'];
    let stepTruncatedMass = 0;
    for (const name of dynamic) {
      const values = next[name];
      let fieldCorrection = 0;
      let postFloorMass = 0;
      for (let index = 0; index < size; index += 1) {
        const value = values[index];
        if (!Number.isFinite(value)) {
          throw new Error(`Non-finite ${name} state produced at index ${index}.`);
        }
        if (value < 0) {
          fieldCorrection += -value * this.cellArea;
          values[index] = 0;
        } else if (name === 'soil' && value > 1) {
          fieldCorrection += (value - 1) * this.cellArea;
          values[index] = 1;
        }
        postFloorMass += values[index] * this.cellArea;
      }
      const correctionBudget = 1e-8 * postFloorMass + 1e-30;
      if (!Number.isFinite(fieldCorrection) || fieldCorrection > correctionBudget) {
        throw new Error(
          `Numerical correction for ${name} (${fieldCorrection}) exceeds budget ${correctionBudget}; step was not committed.`,
        );
      }
      stepTruncatedMass += fieldCorrection;
    }

    this.fields = next;
    this.cumulativeTruncatedMass += stepTruncatedMass;
    this.currentTime += dt;
    this.stepCount += 1;
    this.lastDt = dt;
  }

  advance(duration: number, maxSteps: number): void {
    if (!Number.isFinite(duration) || duration < 0) {
      throw new RangeError(`Duration must be finite and non-negative; received ${duration}.`);
    }
    if (!Number.isInteger(maxSteps) || maxSteps < 0) {
      throw new RangeError(`maxSteps must be a non-negative integer; received ${maxSteps}.`);
    }
    let remaining = duration;
    let steps = 0;
    while (remaining > 0 && steps < maxSteps) {
      const dt = Math.min(this.stableTimeStep(), remaining);
      this.step(dt);
      remaining -= dt;
      if (remaining <= Math.max(Number.EPSILON * duration * 8, Number.MIN_VALUE)) remaining = 0;
      steps += 1;
    }
  }

  setParameters(parameters: Parameters): void {
    const replacement = validateParameters(parameters);
    const nextDt = this.stableTimeStep(this.fields, replacement);
    this.parameters = replacement;
    this.lastDt = nextDt;
  }

  paint(brush: Brush): void {
    if (!FIELD_NAMES.includes(brush.field)) {
      throw new RangeError(`Unknown brush field ${String(brush.field)}.`);
    }
    for (const [label, value] of [
      ['x', brush.x],
      ['y', brush.y],
      ['radius', brush.radius],
      ['amount', brush.amount],
    ] as const) {
      if (!Number.isFinite(value)) throw new RangeError(`Brush ${label} must be finite.`);
    }
    if (brush.radius <= 0) throw new RangeError('Brush radius must be positive.');
    if (this.boundary === 'neumann'
      && (brush.x < 0 || brush.x > 1 || brush.y < 0 || brush.y > 1)) {
      throw new RangeError('Brush coordinates must lie in [0, 1] for Neumann boundaries.');
    }

    const centerX = this.boundary === 'periodic' ? ((brush.x % 1) + 1) % 1 : brush.x;
    const centerY = this.boundary === 'periodic' ? ((brush.y % 1) + 1) % 1 : brush.y;
    const oldValues = this.fields[brush.field];
    const values = oldValues.slice();
    let densityChange = 0;
    for (let row = 0; row < this.n; row += 1) {
      const y = (row + 0.5) / this.n;
      let dy = Math.abs(y - centerY);
      if (this.boundary === 'periodic') dy = Math.min(dy, 1 - dy);
      for (let column = 0; column < this.n; column += 1) {
        const x = (column + 0.5) / this.n;
        let dx = Math.abs(x - centerX);
        if (this.boundary === 'periodic') dx = Math.min(dx, 1 - dx);
        if (dx * dx + dy * dy > brush.radius * brush.radius) continue;
        const index = row * this.n + column;
        const oldValue = oldValues[index];
        let newValue = Math.max(0, oldValue + brush.amount);
        if (brush.field === 'soil') newValue = Math.min(1, newValue);
        if (!Number.isFinite(newValue)) {
          throw new RangeError(`Brush application would make ${brush.field} non-finite.`);
        }
        values[index] = newValue;
        densityChange += newValue - oldValue;
      }
    }
    const massChange = densityChange * this.cellArea;
    const candidateFields = { ...this.fields, [brush.field]: values } as MutableFields;
    const nextDt = this.stableTimeStep(candidateFields);
    this.fields[brush.field] = values;
    this.cumulativeInterventionMass[brush.field] =
      (this.cumulativeInterventionMass[brush.field] ?? 0) + massChange;
    this.lastDt = nextDt;
  }

  snapshot(): Snapshot {
    const fields = {} as Fields;
    const metrics = {} as Record<FieldName, FieldMetric>;
    for (const name of FIELD_NAMES) {
      const copy = this.fields[name].slice();
      fields[name] = copy;
      let sum = 0;
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      for (const value of copy) {
        sum += value;
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
      metrics[name] = {
        total: sum * this.cellArea,
        min,
        max,
        mean: sum / copy.length,
      };
    }
    return {
      n: this.n,
      time: this.currentTime,
      step: this.stepCount,
      dt: this.lastDt,
      fields,
      metrics,
      truncatedMass: this.cumulativeTruncatedMass,
      interventionMass: { ...this.cumulativeInterventionMass },
    };
  }

  /** Combined fractional loss-rate bound, available for numerical diagnostics. */
  rateBound(
    fields: MutableFields = this.fields,
    parameters: Parameters = this.parameters,
  ): number {
    const p = parameters;
    const inverseDx = 1 / this.dx;
    const diffusionScale = 4 * inverseDx * inverseDx;
    const maxPopulation = maximum(fields.population);
    const foodFluxRate = gradientFluxRate(
      fields.food,
      p.chiFood,
      this.n,
      inverseDx,
      this.boundary,
    );
    const infrastructureFluxRate = this.model === 'civilization'
      ? gradientFluxRate(
        fields.infrastructure,
        p.chiInfra,
        this.n,
        inverseDx,
        this.boundary,
      )
      : 0;

    let populationRate = diffusionScale * p.dp + foodFluxRate;
    let foodRate = diffusionScale * p.df;
    let infrastructureRate = 0;
    let soilRate = 0;
    if (this.model === 'civilization') {
      populationRate += infrastructureFluxRate + p.growth;
      foodRate += p.regrowth + p.consumption * maxPopulation
        + p.foodCost * p.investment * maxPopulation / FOOD_HALF_SATURATION;
      infrastructureRate = p.infraDecay;
      soilRate = p.soilRecovery + p.erosion * maxPopulation;
    } else if (this.model === 'agriculture') {
      populationRate += p.growth;
      foodRate += p.regrowth + p.consumption * maxPopulation;
    }
    const rate = Math.max(populationRate, foodRate, infrastructureRate, soilRate);
    if (!Number.isFinite(rate) || rate < 0) {
      throw new Error(`Cannot derive a timestep from invalid rate ${rate}.`);
    }
    return rate;
  }

  private stableTimeStep(
    fields: MutableFields = this.fields,
    parameters: Parameters = this.parameters,
  ): number {
    const rate = this.rateBound(fields, parameters);
    const dt = rate === 0 ? MAX_DT : Math.min(MAX_DT, SAFETY / rate);
    if (!Number.isFinite(dt) || dt <= 0) {
      throw new Error(`Cannot derive a finite positive timestep from rate ${rate}.`);
    }
    return dt;
  }
}
