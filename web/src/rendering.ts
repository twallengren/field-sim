import type { FieldName, Snapshot } from './contracts';
import type { LabElements } from './ui/lab';

const FIELDS: readonly FieldName[] = ['population', 'food', 'infrastructure', 'soil', 'fertility'];
const LABELS: Record<FieldName, string> = {
  population: 'Population', food: 'Food', infrastructure: 'Infrastructure', soil: 'Soil', fertility: 'Fertility',
};
const BASE_SCALES: Record<FieldName, number> = {
  population: 2, food: 3, infrastructure: 0.5, soil: 1, fertility: 4,
};
const LINE_COLORS: Record<FieldName, string> = {
  population: '#d96139', food: '#b58b20', infrastructure: '#3c9472', soil: '#9d654c', fertility: '#6c984a',
};
const PALETTES: Record<FieldName, readonly [number, number, number][]> = {
  population: [[22, 48, 43], [105, 86, 55], [210, 101, 55], [255, 218, 148]],
  food: [[24, 48, 42], [70, 111, 76], [205, 170, 76], [255, 235, 164]],
  infrastructure: [[19, 42, 43], [50, 92, 91], [86, 166, 137], [207, 239, 187]],
  soil: [[49, 36, 31], [115, 70, 50], [181, 128, 82], [226, 211, 161]],
  fertility: [[24, 49, 38], [67, 103, 57], [151, 173, 83], [230, 230, 162]],
};

export interface HistorySample {
  time: number;
  step: number;
  totals: Record<FieldName, number>;
}

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

function interpolate(palette: readonly [number, number, number][], value: number): [number, number, number] {
  const scaled = Math.max(0, Math.min(1, value)) * (palette.length - 1);
  const index = Math.min(palette.length - 2, Math.floor(scaled));
  const mix = scaled - index;
  const a = palette[index];
  const b = palette[index + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * mix),
    Math.round(a[1] + (b[1] - a[1]) * mix),
    Math.round(a[2] + (b[2] - a[2]) * mix),
  ];
}

function sizeCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const rectangle = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round((rectangle.width || canvas.width) * ratio));
  const height = Math.max(1, Math.round((rectangle.height || canvas.height) * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return canvas.getContext('2d');
}

function drawHeatmap(canvas: HTMLCanvasElement, snapshot: Snapshot, field: FieldName, scale: number): void {
  const context = sizeCanvas(canvas);
  if (!context) return;
  const raster = document.createElement('canvas');
  raster.width = snapshot.n;
  raster.height = snapshot.n;
  const rasterContext = raster.getContext('2d');
  if (!rasterContext) return;
  const pixels = rasterContext.createImageData(snapshot.n, snapshot.n);
  const values = snapshot.fields[field];
  for (let screenRow = 0; screenRow < snapshot.n; screenRow += 1) {
    const modelRow = snapshot.n - 1 - screenRow;
    for (let column = 0; column < snapshot.n; column += 1) {
      const value = values[modelRow * snapshot.n + column] / scale;
      const [red, green, blue] = interpolate(PALETTES[field], value);
      const target = (screenRow * snapshot.n + column) * 4;
      pixels.data[target] = red;
      pixels.data[target + 1] = green;
      pixels.data[target + 2] = blue;
      pixels.data[target + 3] = 255;
    }
  }
  rasterContext.putImageData(pixels, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.imageSmoothingEnabled = true;
  context.drawImage(raster, 0, 0, canvas.width, canvas.height);
}

function updateScale(canvas: HTMLCanvasElement, snapshot: Snapshot, field: FieldName, scale: number): void {
  const scaleElement = canvas.parentElement?.querySelector<HTMLElement>('.map-scale');
  if (!scaleElement) return;
  const metric = snapshot.metrics[field];
  const parts = [
    `MIN ${formatValue(metric.min)}`,
    `${LABELS[field].toUpperCase()} · COLOR 0–${formatValue(scale)}`,
    `MAX ${formatValue(metric.max)}`,
  ];
  if (scaleElement.children.length !== 3) {
    scaleElement.replaceChildren(...parts.map((text) => Object.assign(document.createElement('span'), { textContent: text })));
  } else {
    parts.forEach((text, index) => { scaleElement.children[index].textContent = text; });
  }
  canvas.setAttribute('aria-label', `${LABELS[field]} map. Minimum ${formatValue(metric.min)}, maximum ${formatValue(metric.max)}.`);
}

export class LabRenderer {
  private snapshot?: Snapshot;
  private readonly historySamples: HistorySample[] = [];
  private readonly scales = new Map<FieldName, number>();

  constructor(private readonly elements: LabElements) {
    this.reset();
  }

  get history(): readonly HistorySample[] { return this.historySamples; }

  reset(): void {
    this.snapshot = undefined;
    this.historySamples.length = 0;
    this.scales.clear();
    FIELDS.forEach((field) => this.scales.set(field, BASE_SCALES[field]));
    for (const canvas of [this.elements.primaryCanvas, this.elements.secondaryCanvas, this.elements.chartCanvas]) {
      canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    }
    this.elements.inspect.innerHTML = '<span class="eyebrow">INSPECTED CELL</span><strong>Move over the map</strong><span>Values will appear here</span>';
  }

  render(snapshot: Snapshot): void {
    this.snapshot = snapshot;
    for (const field of FIELDS) {
      const retained = this.scales.get(field) ?? BASE_SCALES[field];
      this.scales.set(field, Math.max(retained, snapshot.metrics[field].max * 1.05));
    }
    const totals = {} as Record<FieldName, number>;
    FIELDS.forEach((field) => { totals[field] = snapshot.metrics[field].total; });
    const last = this.historySamples.at(-1);
    if (!last || last.step !== snapshot.step || last.time !== snapshot.time) {
      this.historySamples.push({ time: snapshot.time, step: snapshot.step, totals });
      if (this.historySamples.length > 400) {
        const latest = this.historySamples.at(-1)!;
        const reduced = this.historySamples.filter((_sample, index) => index % 2 === 0);
        if (reduced.at(-1) !== latest) reduced.push(latest);
        this.historySamples.splice(0, this.historySamples.length, ...reduced);
      }
    } else {
      last.totals = totals;
    }
    this.redraw();
  }

  redraw(): void {
    if (!this.snapshot) return;
    const primary = this.elements.fieldSelect.value as FieldName;
    drawHeatmap(this.elements.primaryCanvas, this.snapshot, primary, this.scales.get(primary)!);
    updateScale(this.elements.primaryCanvas, this.snapshot, primary, this.scales.get(primary)!);
    const comparison = this.elements.comparisonSelect.value as FieldName | '';
    if (comparison) {
      drawHeatmap(this.elements.secondaryCanvas, this.snapshot, comparison, this.scales.get(comparison)!);
      updateScale(this.elements.secondaryCanvas, this.snapshot, comparison, this.scales.get(comparison)!);
    }
    const legend = this.elements.primaryCanvas.closest('.map-column')?.querySelector('.legend-row');
    const palette = PALETTES[primary];
    const stops = [palette[0], palette[1], palette[palette.length - 1]];
    legend?.querySelectorAll<HTMLElement>('.legend-swatch').forEach((swatch, index) => {
      swatch.style.background = `rgb(${stops[index].join(',')})`;
    });
    legend?.setAttribute('aria-label', `${LABELS[primary]} color scale from 0 to ${formatValue(this.scales.get(primary)!)}`);
    this.drawChart(primary, comparison || undefined);
  }

  inspectAt(clientX: number, clientY: number): void {
    if (!this.snapshot) return;
    const rectangle = this.elements.primaryCanvas.getBoundingClientRect();
    if (rectangle.width <= 0 || rectangle.height <= 0) return;
    const xFraction = Math.max(0, Math.min(1 - Number.EPSILON, (clientX - rectangle.left) / rectangle.width));
    const yFraction = Math.max(0, Math.min(1 - Number.EPSILON, 1 - (clientY - rectangle.top) / rectangle.height));
    const column = Math.floor(xFraction * this.snapshot.n);
    const row = Math.floor(yFraction * this.snapshot.n);
    const index = row * this.snapshot.n + column;
    const detail = FIELDS.map((field) => `${LABELS[field]} ${formatValue(this.snapshot!.fields[field][index])}`).join(' · ');
    this.elements.inspect.replaceChildren();
    const eyebrow = document.createElement('span'); eyebrow.className = 'eyebrow'; eyebrow.textContent = 'INSPECTED CELL';
    const title = document.createElement('strong'); title.textContent = `x ${(xFraction * 10).toFixed(2)} · y ${(yFraction * 10).toFixed(2)}`;
    const values = document.createElement('span'); values.textContent = detail;
    this.elements.inspect.append(eyebrow, title, values);
  }

  private drawChart(primary: FieldName, comparison?: FieldName): void {
    const canvas = this.elements.chartCanvas;
    const context = sizeCanvas(canvas);
    if (!context) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.width / ratio;
    const height = canvas.height / ratio;
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.historySamples.length) return;
    context.save();
    context.scale(ratio, ratio);
    const visible = comparison && comparison !== primary ? [primary, comparison] : [primary];
    const left = 38, right = width - 8, top = 34, bottom = height - 23;
    const first = this.historySamples[0];
    const last = this.historySamples.at(-1)!;
    const timeSpan = last.time - first.time;
    let peak = 0;
    for (const sample of this.historySamples) {
      for (const field of visible) peak = Math.max(peak, sample.totals[field]);
    }
    peak = Math.max(peak * 1.05, 1);
    context.font = '10px ui-monospace, monospace';
    context.strokeStyle = 'rgba(23,41,39,.12)';
    context.fillStyle = '#69736b';
    context.lineWidth = 1;
    for (let index = 0; index <= 2; index += 1) {
      const y = top + (bottom - top) * index / 2;
      context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke();
      context.textAlign = 'right';
      context.fillText(formatValue(peak * (1 - index / 2)), left - 6, y + 3);
    }
    context.textAlign = 'left';
    context.fillText(formatValue(first.time), left, height - 5);
    context.textAlign = 'right';
    context.fillText(formatValue(last.time), right, height - 5);
    context.textAlign = 'center';
    context.fillText('simulation time', (left + right) / 2, height - 5);
    for (const field of visible) {
      context.beginPath();
      this.historySamples.forEach((sample, index) => {
        const x = timeSpan <= Number.EPSILON ? left : left + ((sample.time - first.time) / timeSpan) * (right - left);
        const y = bottom - (sample.totals[field] / peak) * (bottom - top);
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.strokeStyle = LINE_COLORS[field];
      context.lineWidth = 1.6;
      context.stroke();
      context.fillStyle = LINE_COLORS[field];
      context.textAlign = 'left';
      context.fillText(`${LABELS[field]} ${formatValue(last.totals[field])}`, left, 11 + visible.indexOf(field) * 13);
    }
    context.restore();
    canvas.setAttribute('aria-label', `Field totals over simulation time ${formatValue(first.time)} to ${formatValue(last.time)}. ${visible.map(field => `${LABELS[field]} ${formatValue(last.totals[field])}`).join(', ')}.`);
  }
}
