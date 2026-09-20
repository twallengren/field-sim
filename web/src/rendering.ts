import type { FieldDescriptor, FieldName, Snapshot, TileConfig } from './contracts';
import type { LabElements } from './ui/lab';

export interface HistorySample {
  time: number;
  step: number;
  totals: Record<string, number>;
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

function rasterize(snapshot: Snapshot, field: FieldDescriptor, scale: number): HTMLCanvasElement {
  const raster = document.createElement('canvas'); raster.width = raster.height = snapshot.n;
  const context = raster.getContext('2d')!;
  const pixels = context.createImageData(snapshot.n, snapshot.n);
  const values = snapshot.fields[field.key];
  for (let row = 0; row < snapshot.n; row++) for (let column = 0; column < snapshot.n; column++) {
    const [r,g,b] = interpolate(field.palette, values[(snapshot.n-1-row)*snapshot.n+column]/scale);
    const target = (row*snapshot.n+column)*4;
    pixels.data[target]=r; pixels.data[target+1]=g; pixels.data[target+2]=b; pixels.data[target+3]=255;
  }
  context.putImageData(pixels,0,0); return raster;
}
export interface RenderViews { getTiles(): TileConfig[]; getCanvases(): Map<string,HTMLCanvasElement>; }

export class LabRenderer {
  private snapshot?: Snapshot;
  private readonly historySamples: HistorySample[] = [];
  private readonly scales = new Map<FieldName, number>();

  private descriptors: FieldDescriptor[] = [];
  private rasters = new Map<FieldName, HTMLCanvasElement>();
  private views?: RenderViews;
  configure(fields: FieldDescriptor[], views: RenderViews): void {
    this.descriptors = fields; this.views = views;
  }
  constructor(private readonly elements: LabElements) {
    this.reset();
  }

  get history(): readonly HistorySample[] { return this.historySamples; }

  reset(): void {
    this.snapshot = undefined;
    this.historySamples.length = 0;
    this.scales.clear();
    this.rasters.clear();
    this.descriptors.forEach(field => this.scales.set(field.key, field.scale));
    for (const canvas of [...(this.views?.getCanvases().values() ?? []), this.elements.chartCanvas]) {
      canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    }
    this.elements.inspect.innerHTML = '<span class="eyebrow">INSPECTED CELL</span><strong>Move over the map</strong><span>Values will appear here</span>';
  }

  render(snapshot: Snapshot): void {
    this.snapshot = snapshot;
    this.rasters.clear();
    for (const field of this.descriptors) {
      const retained = this.scales.get(field.key) ?? field.scale;
      this.scales.set(field.key, field.bounded ? field.bounded[1] : Math.max(retained, snapshot.metrics[field.key].max * 1.05));
    }
    const totals: Record<string, number> = {};
    this.descriptors.forEach(field => { totals[field.key] = snapshot.metrics[field.key].total; });
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
    if (!this.snapshot || !this.views) return;
    const canvases = this.views.getCanvases();
    const selected = new Set<FieldName>();
    for (const tile of this.views.getTiles()) {
      tile.layers.filter(layer => layer.visible).forEach(layer => selected.add(layer.field));
      const canvas = canvases.get(tile.id); if (!canvas) continue;
      const rectangle = canvas.getBoundingClientRect();
      if (rectangle.bottom < 0 || rectangle.top > window.innerHeight || rectangle.right < 0 || rectangle.left > window.innerWidth) continue;
      const context = sizeCanvas(canvas); if (!context) continue;
      context.clearRect(0,0,canvas.width,canvas.height);
      context.fillStyle = '#172d29'; context.fillRect(0,0,canvas.width,canvas.height);
      for (const layer of tile.layers) {
        const field = this.descriptors.find(field => field.key === layer.field); if (!field) continue;
        const scale = this.scales.get(field.key) ?? field.scale;
        const metric = this.snapshot.metrics[field.key];
        const legend = canvas.closest('.field-tile')?.querySelector<HTMLElement>(`[data-field-legend="${field.key}"]`);
        if (legend) {
          legend.textContent = `${field.label} · color 0–${formatValue(scale)} · min ${formatValue(metric.min)} · max ${formatValue(metric.max)}`;
          legend.style.borderLeftColor = field.color;
          legend.style.setProperty('--field-ramp', `linear-gradient(to right, ${field.palette.map(rgb=>`rgb(${rgb.join(',')})`).join(',')})`);
        }
        if (!layer.visible || layer.opacity === 0) continue;
        let raster = this.rasters.get(field.key);
        if (!raster) { raster = rasterize(this.snapshot,field,scale); this.rasters.set(field.key,raster); }
        context.globalAlpha = layer.opacity;
        context.drawImage(raster,0,0,canvas.width,canvas.height);
      }
      context.globalAlpha = 1;
      canvas.setAttribute('aria-label', `${tile.layers.filter(l=>l.visible).map(l=>this.descriptors.find(f=>f.key===l.field)?.label).join(' + ')} field map. Paint target: ${tile.paintField}.`);
    }
    this.drawChart([...selected]);
  }

  inspectAt(canvas: HTMLCanvasElement, clientX: number, clientY: number): void {
    if (!this.snapshot) return;
    const rectangle = canvas.getBoundingClientRect();
    if (rectangle.width <= 0 || rectangle.height <= 0) return;
    const xFraction = Math.max(0, Math.min(1 - Number.EPSILON, (clientX - rectangle.left) / rectangle.width));
    const yFraction = Math.max(0, Math.min(1 - Number.EPSILON, 1 - (clientY - rectangle.top) / rectangle.height));
    const column = Math.floor(xFraction * this.snapshot.n);
    const row = Math.floor(yFraction * this.snapshot.n);
    const index = row * this.snapshot.n + column;
    const detail = this.descriptors.map(field => `${field.label} ${formatValue(this.snapshot!.fields[field.key][index])}`).join(' · ');
    this.elements.inspect.replaceChildren();
    const eyebrow = document.createElement('span'); eyebrow.className = 'eyebrow'; eyebrow.textContent = 'INSPECTED CELL';
    const title = document.createElement('strong'); title.textContent = `x ${(xFraction * 10).toFixed(2)} · y ${(yFraction * 10).toFixed(2)}`;
    const values = document.createElement('span'); values.textContent = detail;
    this.elements.inspect.append(eyebrow, title, values);
  }

  private drawChart(visible: FieldName[]): void {
    const canvas = this.elements.chartCanvas;
    canvas.style.height = `${Math.max(180, 120 + visible.length * 13)}px`;
    const context = sizeCanvas(canvas);
    if (!context) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.width / ratio;
    const height = canvas.height / ratio;
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.historySamples.length) return;
    context.save();
    context.scale(ratio, ratio);
    const left = 38, right = width - 8, top = 15 + visible.length * 13, bottom = height - 23;
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
      context.strokeStyle = this.descriptors.find(f=>f.key===field)!.color;
      context.lineWidth = 1.6;
      context.stroke();
      context.fillStyle = this.descriptors.find(f=>f.key===field)!.color;
      context.textAlign = 'left';
      context.fillText(`${this.descriptors.find(f=>f.key===field)!.label} ${formatValue(last.totals[field])}`, left, 11 + visible.indexOf(field) * 13);
    }
    context.restore();
    canvas.setAttribute('aria-label', `Field totals over simulation time ${formatValue(first.time)} to ${formatValue(last.time)}. ${visible.map(field => `${this.descriptors.find(f=>f.key===field)!.label} ${formatValue(last.totals[field])}`).join(', ')}.`);
  }
}
