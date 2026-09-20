import type { FieldDescriptor, FieldName, LayerConfig, TileConfig } from '../contracts';

export interface TileGridCallbacks {
  onChange(tiles: TileConfig[]): void;
  onCanvas(canvas: HTMLCanvasElement, tileId: string): void;
}

export interface TileGrid {
  getTiles(): TileConfig[];
  getCanvases(): Map<string, HTMLCanvasElement>;
  setTiles(tiles: TileConfig[]): void;
  setFields(fields: FieldDescriptor[]): void;
  destroy(): void;
}

interface TileView {
  root: HTMLElement;
  frame: HTMLElement;
  canvas: HTMLCanvasElement;
}

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function iconButton(label: string, glyph: string, className = ''): HTMLButtonElement {
  const button = element('button', `tile-icon-button ${className}`.trim(), glyph);
  button.type = 'button';
  button.setAttribute('aria-label', label);
  button.title = label;
  return button;
}

function copyTiles(tiles: readonly TileConfig[]): TileConfig[] {
  return tiles.map((tile) => ({
    id: tile.id,
    layers: tile.layers.map((layer) => ({ ...layer })),
    paintField: tile.paintField,
  }));
}

function clampOpacity(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.max(0, Math.min(1, value));
}

export function createTileGrid(
  container: HTMLElement,
  initialFields: FieldDescriptor[],
  initialTiles: TileConfig[],
  callbacks: TileGridCallbacks,
): TileGrid {
  let fields = initialFields.map((field) => ({ ...field }));
  let tiles: TileConfig[] = [];
  let destroyed = false;
  let nextTileNumber = 1;
  const views = new Map<string, TileView>();

  container.classList.add('field-tile-grid');
  const tileList = element('div', 'field-tile-list');
  const emptyState = element('div', 'field-tile-empty');
  emptyState.append(
    element('strong', undefined, 'No field views'),
    element('span', undefined, 'Add a view to inspect or paint the simulation.'),
  );
  const addTileButton = element('button', 'add-tile-button', '+ Add field view');
  addTileButton.type = 'button';
  container.replaceChildren(tileList, emptyState, addTileButton);

  function fieldMap(): Map<FieldName, FieldDescriptor> {
    return new Map(fields.map((field) => [field.key, field]));
  }

  function nextId(reserved: ReadonlySet<string> = new Set()): string {
    const used = new Set([...tiles.map((tile) => tile.id), ...reserved]);
    let candidate = `view-${nextTileNumber}`;
    while (used.has(candidate)) {
      nextTileNumber += 1;
      candidate = `view-${nextTileNumber}`;
    }
    nextTileNumber += 1;
    return candidate;
  }

  function normalize(source: readonly TileConfig[]): TileConfig[] {
    const available = fieldMap();
    const usedIds = new Set<string>();
    const normalized: TileConfig[] = [];
    for (const sourceTile of source) {
      let id = typeof sourceTile.id === 'string' && sourceTile.id.length > 0
        ? sourceTile.id
        : nextId(usedIds);
      if (usedIds.has(id)) id = nextId(usedIds);
      usedIds.add(id);

      const usedFields = new Set<FieldName>();
      const layers: LayerConfig[] = [];
      for (const layer of sourceTile.layers ?? []) {
        if (!available.has(layer.field) || usedFields.has(layer.field)) continue;
        usedFields.add(layer.field);
        layers.push({
          field: layer.field,
          opacity: clampOpacity(layer.opacity),
          visible: Boolean(layer.visible),
        });
      }
      if (layers.length === 0 && fields[0]) {
        layers.push({ field: fields[0].key, opacity: 1, visible: true });
      }
      if (layers.length === 0) continue;
      const paintField = layers.length === 1
        ? layers[0].field
        : layers.some((layer) => layer.field === sourceTile.paintField)
          ? sourceTile.paintField
          : layers.find((layer) => available.get(layer.field)?.editable)?.field ?? layers[0].field;
      normalized.push({ id, layers, paintField });
    }
    return normalized;
  }

  function paintStatus(tile: TileConfig): string {
    const descriptor = fields.find((field) => field.key === tile.paintField);
    if (!descriptor) return 'No paint target';
    if (!descriptor.editable || descriptor.kind === 'derived') {
      return `${descriptor.label} is read only${descriptor.kind === 'derived' ? ' · derived field' : ''}`;
    }
    return `Paint target · ${descriptor.label}`;
  }

  function createView(tile: TileConfig): TileView {
    const root = element('article', 'field-tile');
    root.dataset.tileId = tile.id;
    const frame = element('div', 'map-frame');
    const canvas = element('canvas');
    canvas.width = 640;
    canvas.height = 640;
    frame.append(canvas);
    const view = { root, frame, canvas };
    views.set(tile.id, view);
    callbacks.onCanvas(canvas, tile.id);
    return view;
  }

  function makeFieldSelect(tile: TileConfig, layerIndex: number): HTMLSelectElement {
    const select = element('select', 'tile-field-select');
    select.setAttribute('aria-label', `Field for layer ${layerIndex + 1} in ${tile.id}`);
    const selectedElsewhere = new Set(
      tile.layers.filter((_layer, index) => index !== layerIndex).map((layer) => layer.field),
    );
    for (const descriptor of fields) {
      const option = element('option');
      option.value = descriptor.key;
      option.textContent = descriptor.kind === 'derived'
        ? `${descriptor.label} · derived`
        : descriptor.label;
      option.disabled = selectedElsewhere.has(descriptor.key);
      option.selected = descriptor.key === tile.layers[layerIndex].field;
      select.append(option);
    }
    select.addEventListener('change', () => {
      userEdit(() => {
        const current = tiles.find((candidate) => candidate.id === tile.id);
        const layer = current?.layers[layerIndex];
        if (!current || !layer) return;
        const previous = layer.field;
        layer.field = select.value as FieldName;
        if (current.paintField === previous || current.layers.length === 1) {
          current.paintField = layer.field;
        }
      });
    });
    return select;
  }

  function renderLayer(tile: TileConfig, layer: LayerConfig, layerIndex: number): HTMLElement {
    const descriptor = fields.find((field) => field.key === layer.field);
    const row = element('div', 'tile-layer');
    row.dataset.layer = layer.field;

    const visible = element('input', 'tile-layer-visible');
    visible.type = 'checkbox';
    visible.checked = layer.visible;
    visible.setAttribute('aria-label', `Show ${descriptor?.label ?? layer.field} in ${tile.id}`);
    visible.addEventListener('change', () => {
      userEdit(() => {
        const current = tiles.find((candidate) => candidate.id === tile.id)?.layers[layerIndex];
        if (current) current.visible = visible.checked;
      });
    });

    const select = makeFieldSelect(tile, layerIndex);
    const opacity = element('input', 'tile-layer-opacity');
    opacity.type = 'range';
    opacity.min = '0';
    opacity.max = '1';
    opacity.step = '0.05';
    opacity.value = String(layer.opacity);
    opacity.setAttribute('aria-label', `${descriptor?.label ?? layer.field} opacity in ${tile.id}`);
    const opacityValue = element('output', 'tile-layer-opacity-value', `${Math.round(layer.opacity * 100)}%`);
    opacity.addEventListener('input', () => {
      const value = clampOpacity(Number(opacity.value));
      opacityValue.textContent = `${Math.round(value * 100)}%`;
      const current = tiles.find((candidate) => candidate.id === tile.id)?.layers[layerIndex];
      if (!current || destroyed) return;
      current.opacity = value;
      callbacks.onChange(copyTiles(tiles));
    });

    const actions = element('div', 'tile-layer-actions');
    const moveUp = iconButton(`Move ${descriptor?.label ?? layer.field} layer up`, '↑');
    moveUp.disabled = layerIndex === 0;
    moveUp.addEventListener('click', () => userEdit(() => {
      const current = tiles.find((candidate) => candidate.id === tile.id);
      if (!current || layerIndex === 0) return;
      [current.layers[layerIndex - 1], current.layers[layerIndex]] =
        [current.layers[layerIndex], current.layers[layerIndex - 1]];
    }));
    const moveDown = iconButton(`Move ${descriptor?.label ?? layer.field} layer down`, '↓');
    moveDown.disabled = layerIndex === tile.layers.length - 1;
    moveDown.addEventListener('click', () => userEdit(() => {
      const current = tiles.find((candidate) => candidate.id === tile.id);
      if (!current || layerIndex >= current.layers.length - 1) return;
      [current.layers[layerIndex], current.layers[layerIndex + 1]] =
        [current.layers[layerIndex + 1], current.layers[layerIndex]];
    }));
    const remove = iconButton(`Remove ${descriptor?.label ?? layer.field} layer`, '×', 'is-remove');
    remove.disabled = tile.layers.length === 1;
    remove.addEventListener('click', () => userEdit(() => {
      const current = tiles.find((candidate) => candidate.id === tile.id);
      if (!current || current.layers.length === 1) return;
      current.layers.splice(layerIndex, 1);
      if (!current.layers.some((candidate) => candidate.field === current.paintField)) {
        current.paintField = current.layers[0].field;
      }
    }));
    actions.append(moveUp, moveDown, remove);

    const opacityControl = element('label', 'tile-opacity-control');
    opacityControl.append(element('span', undefined, 'Opacity'), opacity, opacityValue);
    row.append(visible, select, opacityControl, actions);
    return row;
  }

  function renderTile(tile: TileConfig, tileIndex: number): void {
    const view = views.get(tile.id) ?? createView(tile);
    view.root.dataset.tileId = tile.id;
    view.frame.classList.toggle('map-frame-primary', tileIndex === 0);
    const paintDescriptor = fields.find((field) => field.key === tile.paintField);
    view.root.classList.toggle('is-readonly', !paintDescriptor?.editable || paintDescriptor.kind === 'derived');

    const header = element('header', 'field-tile-header');
    const heading = element('div', 'field-tile-heading');
    heading.append(
      element('span', 'field-tile-number', String(tileIndex + 1).padStart(2, '0')),
      element('h3', undefined, tile.layers.length === 1
        ? fields.find((field) => field.key === tile.layers[0].field)?.label ?? tile.layers[0].field
        : `${tile.layers.length}-field overlay`),
    );
    const tileActions = element('div', 'field-tile-actions');
    const moveLeft = iconButton(`Move view ${tileIndex + 1} earlier`, '←');
    moveLeft.dataset.focusAction = 'move-view-earlier';
    moveLeft.disabled = tileIndex === 0;
    moveLeft.addEventListener('click', () => userEdit(() => {
      if (tileIndex === 0) return;
      [tiles[tileIndex - 1], tiles[tileIndex]] = [tiles[tileIndex], tiles[tileIndex - 1]];
    }));
    const moveRight = iconButton(`Move view ${tileIndex + 1} later`, '→');
    moveRight.dataset.focusAction = 'move-view-later';
    moveRight.disabled = tileIndex === tiles.length - 1;
    moveRight.addEventListener('click', () => userEdit(() => {
      if (tileIndex >= tiles.length - 1) return;
      [tiles[tileIndex], tiles[tileIndex + 1]] = [tiles[tileIndex + 1], tiles[tileIndex]];
    }));
    const removeTile = iconButton(`Remove view ${tileIndex + 1}`, '×', 'is-remove');
    removeTile.disabled = tiles.length === 1;
    removeTile.addEventListener('click', () => userEdit(() => {
      if (tiles.length === 1) return;
      tiles.splice(tileIndex, 1);
    }));
    tileActions.append(moveLeft, moveRight, removeTile);
    header.append(heading, tileActions);

    const layerPanel = element('div', 'field-tile-controls');
    const layerHeading = element('div', 'tile-control-heading');
    layerHeading.append(element('strong', undefined, 'LAYERS'));
    const addLayer = element('button', 'tile-text-button', '+ Add layer');
    addLayer.type = 'button';
    addLayer.disabled = tile.layers.length >= fields.length;
    addLayer.addEventListener('click', () => userEdit(() => {
      const current = tiles.find((candidate) => candidate.id === tile.id);
      if (!current) return;
      const used = new Set(current.layers.map((candidate) => candidate.field));
      const descriptor = fields.find((candidate) => !used.has(candidate.key));
      if (descriptor) current.layers.push({ field: descriptor.key, opacity: 0.7, visible: true });
    }));
    layerHeading.append(addLayer);
    layerPanel.append(layerHeading);
    tile.layers.forEach((layer, index) => layerPanel.append(renderLayer(tile, layer, index)));

    const paintControl = element('div', 'tile-paint-control');
    if (tile.layers.length > 1) {
      const paintLabel = element('label');
      paintLabel.append(element('span', undefined, 'PAINT TARGET'));
      const paintSelect = element('select', 'tile-paint-select');
      paintSelect.setAttribute('aria-label', `Paint target for ${tile.id}`);
      for (const layer of tile.layers) {
        const descriptor = fields.find((field) => field.key === layer.field);
        const option = element('option');
        option.value = layer.field;
        option.textContent = descriptor
          ? `${descriptor.label}${!descriptor.editable || descriptor.kind === 'derived' ? ' · read only' : ''}`
          : layer.field;
        option.selected = tile.paintField === layer.field;
        paintSelect.append(option);
      }
      paintSelect.addEventListener('change', () => userEdit(() => {
        const current = tiles.find((candidate) => candidate.id === tile.id);
        if (current) current.paintField = paintSelect.value as FieldName;
      }));
      paintLabel.append(paintSelect);
      paintControl.append(paintLabel);
    }
    const status = element('span', 'tile-paint-status', paintStatus(tile));
    status.classList.toggle('is-readonly', !paintDescriptor?.editable || paintDescriptor.kind === 'derived');
    paintControl.append(status);
    layerPanel.append(paintControl);

    const legends = element('div', 'field-tile-legends');
    for (const layer of tile.layers) {
      const descriptor = fields.find((field) => field.key === layer.field);
      const legend = element('div', 'field-tile-legend', descriptor?.label ?? layer.field);
      legend.dataset.fieldLegend = layer.field;
      legend.style.borderLeftColor = descriptor?.color ?? 'transparent';
      legend.classList.toggle('is-hidden', !layer.visible);
      legends.append(legend);
    }

    view.root.replaceChildren(header, view.frame, legends, layerPanel);
    tileList.append(view.root);
  }

  function render(): void {
    if (destroyed) return;
    const active = new Set(tiles.map((tile) => tile.id));
    for (const [id, view] of views) {
      if (active.has(id)) continue;
      view.root.remove();
      views.delete(id);
    }
    tiles.forEach(renderTile);
    emptyState.hidden = tiles.length > 0;
    addTileButton.disabled = fields.length === 0;
  }

  function userEdit(edit: () => void): void {
    if (destroyed) return;
    const active = document.activeElement as HTMLElement | null;
    const focusedTile = active?.closest<HTMLElement>('.field-tile')?.dataset.tileId;
    const focusLabel = active?.getAttribute('aria-label');
    const focusAction = active?.dataset.focusAction;
    const focusLayer = active?.closest<HTMLElement>('.tile-layer')?.dataset.layer;
    const focusText = active?.tagName === 'BUTTON' ? active.textContent : undefined;
    edit();
    tiles = normalize(tiles);
    render();
    if (focusedTile) {
      const view = views.get(focusedTile);
      const replacement = view && [...view.root.querySelectorAll<HTMLElement>('button, input, select')].find(node =>
        focusAction ? node.dataset.focusAction === focusAction : focusLabel ? node.getAttribute('aria-label') === focusLabel : Boolean(focusText) && node.textContent === focusText,
      );
      const layer = view && [...view.root.querySelectorAll<HTMLElement>('.tile-layer')].find(node=>node.dataset.layer===focusLayer);
      const fallback = (layer ?? view?.root)?.querySelector<HTMLElement>('button:not(:disabled), select:not(:disabled), input:not(:disabled)') ?? addTileButton;
      (replacement && !(replacement as HTMLButtonElement).disabled ? replacement : fallback).focus({preventScroll:true});
    }
    callbacks.onChange(copyTiles(tiles));
  }

  addTileButton.addEventListener('click', () => userEdit(() => {
    if (fields.length === 0) return;
    const descriptor = fields[tiles.length % fields.length];
    tiles.push({
      id: nextId(),
      layers: [{ field: descriptor.key, opacity: 1, visible: true }],
      paintField: descriptor.key,
    });
  }));

  tiles = normalize(initialTiles);
  render();

  return {
    getTiles: () => copyTiles(tiles),
    getCanvases: () => new Map(tiles.flatMap((tile) => {
      const view = views.get(tile.id);
      return view ? [[tile.id, view.canvas] as const] : [];
    })),
    setTiles(replacement) {
      if (destroyed) return;
      tiles = normalize(copyTiles(replacement));
      render();
    },
    setFields(replacement) {
      if (destroyed) return;
      fields = replacement.map((field) => ({ ...field }));
      tiles = normalize(tiles);
      render();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      views.clear();
      container.replaceChildren();
      container.classList.remove('field-tile-grid');
    },
  };
}
