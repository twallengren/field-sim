import './styles.css';

import { defaultSetup, defaultTiles, getFieldDescriptors, getPreset } from './catalog';
import type { Brush, FieldName, Parameters, Setup, Snapshot, WorkerCommand, WorkerResponse } from './contracts';
import { LabRenderer } from './rendering';
import { loadSetupFragment, serializeSetupFragment, validateSharedSetup } from './setup';
import { createLab } from './ui/lab';
import { createTileGrid } from './ui/tiles';

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
    beginGeneration(presetSetup, `${preset.title} loaded. Paused at its initial state.`);
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
const grid = createTileGrid(lab.elements.tileContainer, getFieldDescriptors(getPreset(setup.preset).model), setup.tiles ?? defaultTiles(getPreset(setup.preset).model), {
  onChange(tiles) {
    setup = {...setup, version:2, tiles};
    renderer.redraw();
  },
  onCanvas(canvas, tileId) { bindCanvas(canvas,tileId); },
});
renderer.configure(getFieldDescriptors(getPreset(setup.preset).model),grid);

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
  painting = false; lastPaint = undefined; activeCanvas = undefined; activePointerId = undefined;
  generation += 1;
  snapshot = undefined;
  commandResponses = [];
  pendingAdvance = false;
  startupMessage = message;
  startupIsError = false;
  setPlaying(false);
  const model = getPreset(setup.preset).model;
  grid.setFields(getFieldDescriptors(model));
  grid.setTiles(setup.tiles ?? defaultTiles(model));
  renderer.configure(getFieldDescriptors(model), grid);
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

let painting = false;
let strokeField: FieldName = 'population';
let activeCanvas: HTMLCanvasElement | undefined;
let activePointerId: number | undefined;
let lastPaint: { x: number; y: number } | undefined;

function canvasFractions(canvas: HTMLCanvasElement, event: PointerEvent): { x: number; y: number } | undefined {
  const rectangle = canvas.getBoundingClientRect();
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
    field: strokeField,
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

function bindCanvas(canvas: HTMLCanvasElement, tileId: string): void {
  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', event => {
    if ((event.pointerType === 'mouse' && event.button !== 0) || painting) return;
    const tile = grid.getTiles().find(tile => tile.id === tileId);
    if (!tile || !snapshot) return;
    const field = getFieldDescriptors(getPreset(setup.preset).model).find(field => field.key === tile.paintField);
    if (!field?.editable) { lab.setStatus('Cultivation is derived from population and cannot be painted.'); return; }
    const point = canvasFractions(canvas,event); if (!point) return;
    event.preventDefault(); setPlaying(false);
    strokeField = tile.paintField; activeCanvas = canvas; activePointerId = event.pointerId;
    lab.setStatus(`Paused for intervention: painting ${field.label.toLowerCase()}. Press Run to continue.`);
    painting = true; lastPaint = undefined;
    canvas.setPointerCapture(event.pointerId); continueStroke(point);
  });
  canvas.addEventListener('pointermove', event => {
    renderer.inspectAt(canvas,event.clientX,event.clientY);
    if (!painting || activeCanvas !== canvas || activePointerId !== event.pointerId) return;
    const coalesced = event.getCoalescedEvents?.();
    for (const sample of coalesced?.length ? coalesced : [event]) {
      const point = canvasFractions(canvas,sample); if (point) continueStroke(point);
    }
  });
  const endStroke = (event: PointerEvent): void => {
    if (!painting || activeCanvas !== canvas || activePointerId !== event.pointerId) return;
    if (event.type === 'pointerup') {
      const point = canvasFractions(canvas,event); if (point) continueStroke(point);
    }
    painting = false; lastPaint = undefined; activeCanvas = undefined; activePointerId = undefined;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  };
  canvas.addEventListener('pointerup',endStroke);
  canvas.addEventListener('pointercancel',endStroke);
  canvas.addEventListener('lostpointercapture',endStroke);
}

window.addEventListener('hashchange', () => {
  const incoming = loadSetupFragment(window.location.hash);
  beginGeneration(incoming.setup, incoming.message ?? 'Shared setup loaded. Paused at its initial state.');
  startupIsError = Boolean(incoming.message);
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden && playing) {
    setPlaying(false);
    lab.setStatus('Paused because this page is hidden.');
  }
});

const resizeObserver = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(() => renderer.redraw());
resizeObserver?.observe(lab.elements.tileContainer);
window.addEventListener('scroll', () => renderer.redraw(), {passive:true});
window.addEventListener('resize', () => renderer.redraw());
resizeObserver?.observe(lab.elements.chartCanvas);

async function shareSetup(): Promise<void> {
  const url = new URL(window.location.href);
  url.hash = serializeSetupFragment({...setup,version:2,tiles:grid.getTiles()});
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
    lab.setStatus('Setup link copied. It includes setup, parameters, and view layout; painted interventions and evolved state are excluded.');
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
