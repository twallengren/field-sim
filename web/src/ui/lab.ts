import { parameterDefinitions, presets } from '../catalog';
import type { FieldName, Parameters, Setup, Snapshot } from '../contracts';

export interface LabCallbacks {
  onPreset?: (id: string) => void;
  onSetup?: (patch: Partial<Setup>) => void;
  onParameters?: (parameters: Parameters) => void;
  onPlay?: () => void;
  onStep?: () => void;
  onReset?: () => void;
  onShare?: () => void;
  onSpeed?: (speed: number) => void;
}

export interface LabElements {
  primaryCanvas: HTMLCanvasElement;
  secondaryCanvas: HTMLCanvasElement;
  fieldSelect: HTMLSelectElement;
  comparisonSelect: HTMLSelectElement;
  brushFieldSelect: HTMLSelectElement;
  brushRadius: HTMLInputElement;
  brushStrength: HTMLInputElement;
  brushMode: HTMLSelectElement;
  chartCanvas: HTMLCanvasElement;
  inspect: HTMLElement;
  status: HTMLElement;
  playButton: HTMLButtonElement;
  stepButton: HTMLButtonElement;
  parameterInputs: Map<keyof Parameters, HTMLInputElement>;
}

export interface Lab {
  elements: LabElements;
  setSetup(setup: Setup): void;
  setSnapshot(snapshot: Snapshot): void;
  setPlaying(playing: boolean): void;
  setStatus(message: string, isError?: boolean): void;
  destroy(): void;
}

const fieldLabels: Record<FieldName, string> = {
  population: 'Population', food: 'Food', infrastructure: 'Infrastructure', soil: 'Soil', fertility: 'Fertility',
};
const fieldColors: Record<FieldName, string> = {
  population: '#e67b4c', food: '#efd783', infrastructure: '#8bc5a3', soil: '#ad7960', fertility: '#c8e0a5',
};
const fieldOrder: FieldName[] = ['population', 'food', 'infrastructure', 'soil', 'fertility'];

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function option(value: string, label: string): HTMLOptionElement {
  const o = document.createElement('option'); o.value = value; o.textContent = label; return o;
}
function labelFor(input: HTMLElement, text: string): HTMLLabelElement {
  const label = el('label', 'control-label'); label.append(input, el('span', undefined, text)); return label;
}
function format(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toFixed(digits);
}

export function createLab(root: HTMLElement, callbacks: LabCallbacks = {}): Lab {
  root.replaceChildren();
  root.classList.add('lab-shell');
  let currentSetup: Setup | undefined;
  let currentPreset = presets[0];
  const listeners: Array<() => void> = [];
  const listen = <T extends EventTarget>(node: T, event: string, fn: EventListener) => { node.addEventListener(event, fn); listeners.push(() => node.removeEventListener(event, fn)); };

  const mast = el('header', 'lab-masthead');
  const mark = el('div', 'lab-mark'); mark.innerHTML = '<span class="lab-mark-dot"></span><span>FIELD<br>SIMULATION</span>';
  const eyebrow = el('p', 'eyebrow', 'A computational field laboratory / 01—05');
  const title = el('h1', undefined, 'Civilization, as a field.');
  const intro = el('p', 'lab-intro', 'Explore settlement, collective investment, and ecological limits through continuous fields. No individual agents; just local interactions and the patterns they create.');
  const headerMeta = el('div', 'mast-meta'); headerMeta.append(el('span', undefined, 'INTERACTIVE ATLAS'), el('span', undefined, 'L = 10 UNITS'), el('span', undefined, 'V. 1.0'));
  mast.append(mark, el('div', 'mast-copy')); (mast.children[1] as HTMLElement).append(eyebrow, title, intro); mast.append(headerMeta);

  const gallery = el('nav', 'experiment-gallery'); gallery.setAttribute('aria-label', 'Experiments');
  const cards = new Map<string, HTMLButtonElement>();
  presets.forEach((preset, index) => {
    const card = el('button', 'experiment-card') as HTMLButtonElement; card.type = 'button'; card.dataset.preset = preset.id;
    card.innerHTML = `<span class="card-number">${String(index + 1).padStart(2, '0')}</span><span class="card-kicker">${preset.kicker.replace(/^\d+\s*\/\s*/, '')}</span><strong>${preset.title}</strong><small>${preset.description}</small><span class="card-arrow" aria-hidden="true">↗</span>`;
    listen(card, 'click', () => callbacks.onPreset?.(preset.id)); cards.set(preset.id, card); gallery.append(card);
  });

  const workspace = el('main', 'lab-workspace');
  const mapColumn = el('section', 'map-column');
  const mapBar = el('div', 'map-bar');
  const mapTitle = el('div', 'map-title'); mapTitle.append(el('span', 'eyebrow', 'LIVE FIELD /'), el('strong', 'primary-field-name', 'Population'));
  const mapActions = el('div', 'map-actions');
  const fieldSelect = el('select', 'select-compact') as HTMLSelectElement; fieldSelect.setAttribute('aria-label', 'Primary field');
  const comparisonSelect = el('select', 'select-compact') as HTMLSelectElement; comparisonSelect.setAttribute('aria-label', 'Comparison field'); comparisonSelect.append(option('', 'No comparison'));
  fieldOrder.forEach((field) => { fieldSelect.append(option(field, fieldLabels[field])); comparisonSelect.append(option(field, fieldLabels[field])); });
  mapActions.append(labelFor(fieldSelect, 'SHOW'), labelFor(comparisonSelect, 'COMPARE'));
  mapBar.append(mapTitle, mapActions);

  const canvases = el('div', 'map-canvases');
  const mainFrame = el('div', 'map-frame map-frame-primary'); const primaryCanvas = el('canvas'); primaryCanvas.setAttribute('aria-label', 'Primary simulation field map'); primaryCanvas.width = 800; primaryCanvas.height = 800; const primaryScale = el('div', 'map-scale'); primaryScale.innerHTML = '<span>0</span><span>POPULATION DENSITY</span><span>2</span>'; mainFrame.append(primaryCanvas, primaryScale);
  const secondaryFrame = el('div', 'map-frame map-frame-secondary'); const secondaryCanvas = el('canvas'); secondaryCanvas.setAttribute('aria-label', 'Comparison simulation field map'); secondaryCanvas.width = 400; secondaryCanvas.height = 400; secondaryFrame.append(secondaryCanvas, el('div', 'map-scale')); secondaryFrame.hidden = true; canvases.append(mainFrame, secondaryFrame);
  const legend = el('div', 'legend-row'); legend.setAttribute('aria-label', 'Field intensity legend: low, medium, high. Color ranges retain observed peaks.'); legend.innerHTML = '<span class="legend-swatch legend-low" aria-hidden="true"></span><span>LOW</span><span class="legend-swatch legend-mid" aria-hidden="true"></span><span>MEDIUM</span><span class="legend-swatch legend-high" aria-hidden="true"></span><span>HIGH</span><span class="legend-note">Color ranges retain observed peaks</span>';
  const paintHint = el('p', 'paint-hint', 'Drag to paint the selected field · Painting pauses the experiment');
  mapColumn.append(mapBar, canvases, legend, paintHint);

  const timeline = el('section', 'timeline-panel');
  const metrics = el('div', 'metric-strip');
  const metricNodes: Record<string, HTMLElement> = {};
  [['time', 'TIME', '0.0'], ['step', 'STEP', '0'], ['dt', 'ΔT', '0.000'], ['mass', 'POPULATION', '—']].forEach(([key, label, value]) => { const item = el('div', 'metric'); const valueNode = el('strong', undefined, value); item.append(el('span', 'metric-label', label), valueNode); metricNodes[key] = valueNode; metrics.append(item); });
  const chartFrame = el('div', 'chart-frame'); const chartCanvas = el('canvas'); chartCanvas.width = 800; chartCanvas.height = 150; chartCanvas.setAttribute('aria-label', 'Field totals over simulation time'); chartFrame.append(chartCanvas, el('span', 'chart-axis-label', 'FIELD TOTALS / TIME'));
  timeline.append(metrics, chartFrame);
  mapColumn.append(timeline);

  const side = el('aside', 'control-panel');
  const runHead = el('div', 'panel-heading'); runHead.append(el('span', 'eyebrow', 'CONTROL ROOM'), el('h2', undefined, 'Run the field'));
  const transport = el('div', 'transport-controls'); const playButton = el('button', 'button button-primary') as HTMLButtonElement; playButton.type = 'button'; playButton.innerHTML = '<span class="play-icon">▶</span><span>Run experiment</span>'; const stepButton = el('button', 'button button-square') as HTMLButtonElement; stepButton.type = 'button'; stepButton.setAttribute('aria-label', 'Advance one step'); stepButton.innerHTML = '＋<span>STEP</span>'; const resetButton = el('button', 'button button-square') as HTMLButtonElement; resetButton.type = 'button'; resetButton.setAttribute('aria-label', 'Reset experiment'); resetButton.innerHTML = '↺<span>RESET</span>'; transport.append(playButton, stepButton, resetButton);
  listen(playButton, 'click', () => callbacks.onPlay?.()); listen(stepButton, 'click', () => callbacks.onStep?.()); listen(resetButton, 'click', () => callbacks.onReset?.());
  const speed = el('input') as HTMLInputElement; speed.type = 'range'; speed.min = '0.1'; speed.max = '4'; speed.step = '0.1'; speed.value = '1'; speed.setAttribute('aria-label', 'Simulation speed'); const speedValue = el('output', undefined, '1×'); const speedControl = el('div', 'range-row'); speedControl.append(el('span', undefined, 'SPEED'), speed, speedValue); listen(speed, 'input', () => { speedValue.textContent = `${speed.value}×`; callbacks.onSpeed?.(Number(speed.value)); });
  const share = el('button', 'share-button') as HTMLButtonElement; share.type = 'button'; share.innerHTML = '<span>↗</span> Copy setup link'; listen(share, 'click', () => callbacks.onShare?.());

  const setupSection = el('section', 'control-section'); setupSection.append(el('h3', undefined, 'FIELD CONDITIONS'));
  const setupGrid = el('div', 'setup-grid');
  const seed = el('input') as HTMLInputElement; seed.type = 'number'; seed.min = '0'; seed.max = '4294967295'; seed.step = '1'; seed.inputMode = 'numeric'; setupGrid.append(labelFor(seed, 'RANDOM SEED'));
  const resolution = el('select') as HTMLSelectElement; [32, 64, 128].forEach(n => resolution.append(option(String(n), `${n} × ${n}`))); setupGrid.append(labelFor(resolution, 'RESOLUTION'));
  const boundary = el('select') as HTMLSelectElement; boundary.append(option('neumann', 'Closed edges (Neumann / no flux)'), option('periodic', 'Wraparound (periodic)')); boundary.title = 'Closed edges stop flux at the boundary; wraparound connects opposite edges.'; setupGrid.append(labelFor(boundary, 'BOUNDARIES')); setupSection.append(setupGrid);
  const patchSetup = () => callbacks.onSetup?.({ seed: Number(seed.value) || 0, n: Number(resolution.value) as 32 | 64 | 128, boundary: boundary.value as Setup['boundary'] });
  listen(seed, 'change', patchSetup); listen(resolution, 'change', patchSetup); listen(boundary, 'change', patchSetup);

  const parameterSection = el('section', 'control-section parameter-section'); parameterSection.append(el('h3', undefined, 'MODEL PARAMETERS')); const parameterGrid = el('div', 'parameter-grid'); const parameterInputs = new Map<keyof Parameters, HTMLInputElement>();
  parameterDefinitions.forEach((definition) => { const input = el('input') as HTMLInputElement; input.type = 'range'; input.min = String(definition.min); input.max = String(definition.max); input.step = String(definition.step); input.dataset.key = definition.key; const value = el('output', undefined, String(definition.default)); input.setAttribute('aria-label', definition.label); const row = el('label', 'parameter-row'); row.append(el('span', undefined, definition.label), value, input); listen(input, 'input', () => { value.textContent = input.value; if (currentSetup) callbacks.onParameters?.({...currentSetup.parameters, [definition.key]: Number(input.value)}); }); parameterGrid.append(row); parameterInputs.set(definition.key, input); }); parameterSection.append(parameterGrid);
  const advanced = el('details', 'advanced-control'); const advancedSummary = el('summary', undefined, 'Model parameters'); advanced.append(advancedSummary, parameterSection); const advancedCopy = el('p', undefined, 'Dimmed parameters have no effect in this model. Changing a coefficient affects the current world.'); advanced.append(advancedCopy);

  const brush = el('section', 'control-section brush-section'); brush.append(el('h3', undefined, 'FIELD INTERVENTION'), el('p', 'control-help', 'Painting pauses the run while you work. Press run to resume.')); const brushGrid = el('div', 'brush-grid'); const brushFieldSelect = el('select') as HTMLSelectElement; fieldOrder.forEach(field => brushFieldSelect.append(option(field, fieldLabels[field]))); brushFieldSelect.setAttribute('aria-label', 'Field to paint'); const brushMode = el('select') as HTMLSelectElement; brushMode.append(option('add', 'Add density'), option('remove', 'Remove density')); brushMode.setAttribute('aria-label', 'Brush mode'); const brushRadius = el('input') as HTMLInputElement; brushRadius.type = 'range'; brushRadius.min = '0.01'; brushRadius.max = '0.25'; brushRadius.step = '0.01'; brushRadius.value = '0.08'; brushRadius.setAttribute('aria-label', 'Brush radius'); const brushStrength = el('input') as HTMLInputElement; brushStrength.type = 'range'; brushStrength.min = '0.01'; brushStrength.max = '1'; brushStrength.step = '0.01'; brushStrength.value = '0.2'; brushStrength.setAttribute('aria-label', 'Brush strength'); brushGrid.append(labelFor(brushFieldSelect, 'FIELD'), labelFor(brushMode, 'ACTION'), labelFor(brushRadius, 'RADIUS'), labelFor(brushStrength, 'STRENGTH')); brush.append(brushGrid);
  const inspect = el('div', 'inspect-panel'); inspect.setAttribute('aria-live', 'polite'); inspect.innerHTML = '<span class="eyebrow">INSPECTED CELL</span><strong>Move over the map</strong><span>Values will appear here</span>';
  const status = el('div', 'lab-status'); status.setAttribute('role', 'status');
  side.append(runHead, transport, speedControl, share, setupSection, brush, advanced, inspect, status);
  workspace.append(mapColumn, side); const footer = el('footer', 'lab-footer'); footer.innerHTML = '<span>A toy model of spatial feedback, not a reconstruction of history.</span><a href="https://github.com/twallengren/field-sim#readme" target="_blank" rel="noopener noreferrer">Models, methods & source ↗</a>'; root.append(mast, gallery, workspace, footer);

  const elements: LabElements = { primaryCanvas, secondaryCanvas, fieldSelect, comparisonSelect, brushFieldSelect, brushRadius, brushStrength, brushMode, chartCanvas, inspect, status, playButton, stepButton, parameterInputs };
  listen(fieldSelect, 'change', () => { (mapTitle.querySelector('.primary-field-name') as HTMLElement).textContent = fieldLabels[fieldSelect.value as FieldName]; });
  listen(comparisonSelect, 'change', () => { secondaryFrame.hidden = !comparisonSelect.value; });

  const lab: Lab = {
    elements,
    setSetup(setup) { currentSetup = setup; currentPreset = presets.find(p => p.id === setup.preset) || currentPreset; seed.value = String(setup.seed); resolution.value = String(setup.n); boundary.value = setup.boundary; cards.forEach((card, id) => { card.classList.toggle('is-active', id === setup.preset); card.setAttribute('aria-pressed', String(id === setup.preset)); }); parameterInputs.forEach((input, key) => { const value = setup.parameters[key]; input.value = String(value); const output = input.parentElement?.querySelector('output'); if (output) output.textContent = String(value); const definition = parameterDefinitions.find(d => d.key === key); const relevant = currentPreset.model === 'chemotaxis' ? ['dp', 'df', 'chiFood'].includes(String(key)) : currentPreset.model === 'agriculture' ? ['dp', 'df', 'chiFood', 'growth', 'regrowth', 'consumption'].includes(String(key)) : true; input.disabled = !relevant; input.parentElement?.classList.toggle('is-muted', !relevant); if (definition) input.title = `${definition.label}: ${format(value)}`; }); },
    setSnapshot(snapshot) { metricNodes.time.textContent = format(snapshot.time, 2); metricNodes.step.textContent = String(snapshot.step); metricNodes.dt.textContent = format(snapshot.dt, 4); metricNodes.mass.textContent = format(snapshot.metrics.population.total, 2); },
    setPlaying(value) { playButton.classList.toggle('is-playing', value); playButton.querySelector('span:last-child')!.textContent = value ? 'Pause experiment' : 'Run experiment'; playButton.querySelector('.play-icon')!.textContent = value ? 'Ⅱ' : '▶'; },
    setStatus(message, isError = false) { status.textContent = message; status.classList.toggle('is-error', isError); },
    destroy() { listeners.forEach(remove => remove()); root.replaceChildren(); },
  };
  lab.setSetup({ version: 1, preset: currentPreset.id, seed: 0, n: 64, boundary: 'neumann', parameters: {...currentPreset.parameters} as Parameters });
  return lab;
}

export type { FieldName, Parameters, Setup, Snapshot };
