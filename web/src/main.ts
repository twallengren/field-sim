import './styles.css';

import { defaultSetup, getPreset } from './catalog';
import type { Brush, FieldName, Parameters, Setup, Snapshot, WorkerCommand, WorkerResponse } from './contracts';
import { LabRenderer } from './rendering';
import { loadSetupFragment, serializeSetupFragment, validateSharedSetup } from './setup';
import { createLab } from './ui/lab';

const root = document.querySelector<HTMLElement>('#app');
if (!root) throw new Error('The application root is missing.');

const loaded = loadSetupFragment(window.location.hash);
let setup = loaded.setup;
let generation = 0;
let snapshot: Snapshot | undefined;
let playing = false;
let pendingAdvance = false;
let speed = 1;
let animationFrame: number | undefined;
let lastWallTime = performance.now();
let targetSimulationTime = 0;
let commandResponses: WorkerCommand['type'][] = [];
let startupMessage = loaded.message;
let startupIsError = Boolean(loaded.message);

const worker = new Worker(new URL('./engine/worker.ts', import.meta.url), { type: 'module' });
const lab = createLab(root, {
  onPreset(id) {
    const presetSetup = defaultSetup(id);
    presetSetup.seed = setup.seed;
    presetSetup.n = setup.n;
    presetSetup.boundary = setup.boundary;
    const preset = getPreset(id);
    const clearComparison = lab.elements.comparisonSelect.value
      && !preset.fields.includes(lab.elements.comparisonSelect.value as FieldName);
    beginGeneration(presetSetup, `${preset.title} loaded. Paused at its initial state.`);
    lab.elements.fieldSelect.value = preset.fields[0];
    lab.elements.fieldSelect.dispatchEvent(new Event('change'));
    if (clearComparison) {
      lab.elements.comparisonSelect.value = '';
      lab.elements.comparisonSelect.dispatchEvent(new Event('change'));
    }
  },
  onSetup(patch) {
    const replacement = validateSharedSetup({ ...setup, ...patch });
    if (!replacement) {
      lab.setSetup(setup);
      lab.setStatus('Those field conditions are invalid. The current setup was kept.', true);
      return;
    }
    beginGeneration(replacement, 'Field conditions applied. Paused at a new initial state.');
  },
  onParameters(parameters) {
    const replacement = validateSharedSetup({ ...setup, parameters });
    if (!replacement) {
      lab.setSetup(setup);
      lab.setStatus('A parameter was outside its allowed range. The current values were kept.', true);
      return;
    }
    setup = replacement;
    lab.setSetup(setup);
    send({ type: 'parameters', generation, parameters: cloneParameters(parameters) });
  },
  onPlay() {
    setPlaying(!playing);
    if (playing) {
      lastWallTime = performance.now();
      targetSimulationTime = snapshot?.time ?? 0;
      scheduleAdvance();
      lab.setStatus('Running. 1× targets four simulation-time units per second.');
    } else {
      lab.setStatus('Paused.');
    }
  },
  onStep() {
    setPlaying(false);
    send({ type: 'step', generation });
    lab.setStatus('Advancing one stable numerical step…');
  },
  onReset() {
    beginGeneration(setup, 'Reset to the current setup. Paused.');
  },
  onShare() {
    void shareSetup();
  },
  onSpeed(value) {
    if (Number.isFinite(value) && value > 0) speed = value;
  },
});
const renderer = new LabRenderer(lab.elements);

function cloneParameters(parameters: Parameters): Parameters {
  return { ...parameters };
}

function send(command: WorkerCommand): void {
  commandResponses.push(command.type);
  worker.postMessage(command);
}

function setPlaying(value: boolean): void {
  playing = value;
  lab.setPlaying(value);
  if (!value && animationFrame !== undefined) {
    cancelAnimationFrame(animationFrame);
    animationFrame = undefined;
  }
}

function beginGeneration(replacement: Setup, message: string): void {
  setup = {
    ...replacement,
    parameters: cloneParameters(replacement.parameters),
  };
  generation += 1;
  snapshot = undefined;
  commandResponses = [];
  pendingAdvance = false;
  startupMessage = message;
  startupIsError = false;
  setPlaying(false);
  renderer.reset();
  lab.setSetup(setup);
  lab.setStatus('Preparing the field…');
  send({ type: 'init', generation, setup });
}

function scheduleAdvance(): void {
  if (!playing || pendingAdvance || animationFrame !== undefined || !snapshot) return;
  animationFrame = requestAnimationFrame(advanceFrame);
}

function advanceFrame(now: number): void {
  animationFrame = undefined;
  if (!playing || pendingAdvance || !snapshot) return;
  const elapsedSeconds = Math.max(0, Math.min(0.25, (now - lastWallTime) / 1000));
  lastWallTime = now;
  targetSimulationTime += elapsedSeconds * speed * 4;
  const duration = Math.min(0.75, Math.max(0, targetSimulationTime - snapshot.time));
  if (duration <= 1e-5) {
    scheduleAdvance();
    return;
  }
  pendingAdvance = true;
  const maxSteps = setup.n === 128 ? 12 : setup.n === 64 ? 25 : 50;
  send({ type: 'advance', generation, duration, maxSteps });
}

worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
  const response = event.data;
  if (response.generation !== generation) return;
  const command = commandResponses.shift();
  if (command === 'advance') pendingAdvance = false;

  if (response.type === 'error') {
    setPlaying(false);
    lab.setStatus(`Simulation stopped: ${response.message} Reset or change the setup to recover.`, true);
    return;
  }

  const hadSnapshot = snapshot !== undefined;
  snapshot = response.snapshot;
  lab.setSnapshot(snapshot);
  renderer.render(snapshot);
  if (command === 'init') {
    targetSimulationTime = snapshot.time;
    lastWallTime = performance.now();
    lab.setStatus(startupMessage ?? 'Ready. The experiment starts paused.', startupIsError && !hadSnapshot);
    startupMessage = undefined;
    startupIsError = false;
  } else if (command === 'step') {
    lab.setStatus('Advanced one stable numerical step. Paused.');
  }
  scheduleAdvance();
});

worker.addEventListener('error', (event) => {
  setPlaying(false);
  pendingAdvance = false;
  lab.setStatus(`The simulation worker stopped: ${event.message || 'unknown worker error'}. Reload this page to reconnect.`, true);
});

lab.elements.fieldSelect.addEventListener('change', () => renderer.redraw());
lab.elements.comparisonSelect.addEventListener('change', () => renderer.redraw());

let painting = false;
let lastPaint: { x: number; y: number } | undefined;

function canvasFractions(event: PointerEvent): { x: number; y: number } | undefined {
  const rectangle = lab.elements.primaryCanvas.getBoundingClientRect();
  if (rectangle.width <= 0 || rectangle.height <= 0) return undefined;
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rectangle.left) / rectangle.width)),
    y: Math.max(0, Math.min(1, 1 - (event.clientY - rectangle.top) / rectangle.height)),
  };
}

function paintAt(point: { x: number; y: number }): void {
  const radius = Number(lab.elements.brushRadius.value);
  const strength = Number(lab.elements.brushStrength.value);
  const sign = lab.elements.brushMode.value === 'remove' ? -1 : 1;
  const brush: Brush = {
    field: lab.elements.brushFieldSelect.value as FieldName,
    x: point.x,
    y: point.y,
    radius,
    amount: sign * strength,
  };
  send({ type: 'paint', generation, brush });
}

function continueStroke(point: { x: number; y: number }): void {
  if (!lastPaint) {
    paintAt(point);
    lastPaint = point;
    return;
  }
  const dx = point.x - lastPaint.x;
  const dy = point.y - lastPaint.y;
  const distance = Math.hypot(dx, dy);
  if (distance < 1e-8) return;
  const spacing = Math.max(Number(lab.elements.brushRadius.value) * 0.45, 1 / (snapshot?.n ?? setup.n));
  const count = Math.max(1, Math.ceil(distance / spacing));
  for (let index = 1; index <= count; index += 1) {
    paintAt({ x: lastPaint.x + dx * index / count, y: lastPaint.y + dy * index / count });
  }
  lastPaint = point;
}

lab.elements.primaryCanvas.style.touchAction = 'none';
lab.elements.primaryCanvas.addEventListener('pointerdown', (event) => {
  if (event.pointerType === 'mouse' && event.button !== 0) return;
  const point = canvasFractions(event);
  if (!point || !snapshot) return;
  event.preventDefault();
  setPlaying(false);
  lab.setStatus('Paused for field intervention. Press Run to continue.');
  painting = true;
  lastPaint = undefined;
  lab.elements.primaryCanvas.setPointerCapture(event.pointerId);
  continueStroke(point);
});
lab.elements.primaryCanvas.addEventListener('pointermove', (event) => {
  renderer.inspectAt(event.clientX, event.clientY);
  if (!painting) return;
  const coalesced = event.getCoalescedEvents?.();
  const events = coalesced?.length ? coalesced : [event];
  for (const sample of events) {
    const point = canvasFractions(sample);
    if (point) continueStroke(point);
  }
});
const endStroke = (event: PointerEvent): void => {
  if (!painting) return;
  const point = canvasFractions(event);
  if (point) continueStroke(point);
  painting = false;
  lastPaint = undefined;
  if (lab.elements.primaryCanvas.hasPointerCapture(event.pointerId)) lab.elements.primaryCanvas.releasePointerCapture(event.pointerId);
};
lab.elements.primaryCanvas.addEventListener('pointerup', endStroke);
lab.elements.primaryCanvas.addEventListener('pointercancel', endStroke);

document.addEventListener('visibilitychange', () => {
  if (document.hidden && playing) {
    setPlaying(false);
    lab.setStatus('Paused because this page is hidden.');
  }
});

const resizeObserver = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(() => renderer.redraw());
resizeObserver?.observe(lab.elements.primaryCanvas);
resizeObserver?.observe(lab.elements.secondaryCanvas);
resizeObserver?.observe(lab.elements.chartCanvas);

async function shareSetup(): Promise<void> {
  const url = new URL(window.location.href);
  url.hash = serializeSetupFragment(setup);
  const text = url.toString();
  let copied = false;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch {
    // The selectable fallback below remains available when clipboard access is denied.
  }

  if (!copied) {
    const temporary = document.createElement('textarea');
    temporary.value = text;
    temporary.setAttribute('readonly', '');
    temporary.style.position = 'fixed';
    temporary.style.opacity = '0';
    document.body.append(temporary);
    temporary.select();
    try { copied = document.execCommand('copy'); } catch { copied = false; }
    temporary.remove();
  }

  if (copied) {
    lab.setStatus('Setup link copied. It includes setup and parameters only; painted interventions and evolved state are excluded.');
    return;
  }
  lab.setStatus('Clipboard access is unavailable. Select and copy this setup link:');
  const fallback = document.createElement('input');
  fallback.readOnly = true;
  fallback.value = text;
  fallback.setAttribute('aria-label', 'Setup link');
  fallback.style.width = '100%';
  lab.elements.status.append(document.createElement('br'), fallback);
  fallback.select();
}

lab.setSetup(setup);
renderer.reset();
send({ type: 'init', generation, setup });
