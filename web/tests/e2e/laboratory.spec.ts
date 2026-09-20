import { devices, expect, test } from '@playwright/test';

const ecologyPresets = ['water_settlement', 'water_overuse', 'soil_recovery'];
const presets = [
  'settlement',
  'collective_investment',
  'overshoot',
  'agriculture',
  'chemotaxis_demo',
  ...ecologyPresets,
];

type Page = import('@playwright/test').Page;

function metric(page: Page, label: string) {
  return page.locator('.metric').filter({ hasText: label }).locator('strong');
}

function tile(page: Page, id: string) {
  return page.locator(`.field-tile[data-tile-id="${id}"]`);
}

async function ready(page: Page) {
  await page.goto('./');
  await expect(page.locator('.lab-status')).toHaveText(/initial state|Ready\./, { timeout: 10_000 });
  await expect(page.getByRole('button', { name: /Run experiment/ })).toBeVisible();
  await expect(page.locator('[data-preset="water_settlement"]')).toHaveAttribute('aria-pressed', 'true');
}

async function expectCanvasHasInk(page: Page, tileId: string) {
  const canvas = tile(page, tileId).locator('canvas');
  await canvas.scrollIntoViewIfNeeded();
  await expect.poll(async () => canvas.evaluate((node) => {
    const context = (node as HTMLCanvasElement).getContext('2d');
    if (!context) return false;
    const pixels = context.getImageData(0, 0, (node as HTMLCanvasElement).width, (node as HTMLCanvasElement).height).data;
    let low = 255;
    let high = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      const luminance = pixels[index] + pixels[index + 1] + pixels[index + 2];
      low = Math.min(low, luminance);
      high = Math.max(high, luminance);
    }
    return high - low > 12;
  })).toBe(true);
}

async function grantClipboard(page: Page) {
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'], { origin: new URL(page.url()).origin });
}

async function readClipboardSetupLink(page: Page) {
  await expect.poll(async () => page.evaluate(async () => navigator.clipboard.readText())).toMatch(/#setup=/);
  return page.evaluate(async () => navigator.clipboard.readText());
}

function paintedWaterBudget(text: string): number {
  return Number(text.match(/painted(?: water)?\s+([0-9.,-]+)/)?.[1]?.replaceAll(',', '') ?? Number.NaN);
}

test('loads the ecology default paused with a live worker and rendered field tiles', async ({ page }) => {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await ready(page);
  await expect(metric(page, 'TIME')).toHaveText('0.00');
  await expect(metric(page, 'STEP')).toHaveText('0');
  await expect(page.getByRole('button', { name: /Run experiment/ })).toHaveText(/Run experiment/);
  await expect(page.locator('.field-tile')).toHaveCount(3);
  await expect(page.locator('.field-tile-heading h3')).toHaveText(['Population', 'Water', 'Soil condition']);
  await expect(page.locator('.water-budget')).toBeVisible();
  await expect(page.locator('.water-budget')).toContainText('Water budget');
  await expectCanvasHasInk(page, 'view-1');
  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
  expect(pageErrors, `page errors: ${pageErrors.join('\n')}`).toEqual([]);
});

test('run, pause, step, reset, and setup controls update the paused state', async ({ page }) => {
  await ready(page);
  const run = page.locator('.button-primary');
  await run.click();
  await expect(run).toHaveText(/Pause experiment/);
  await expect(metric(page, 'TIME')).not.toHaveText('0.00', { timeout: 5_000 });
  await run.click();
  await expect(run).toHaveText(/Run experiment/);
  await expect(page.locator('.lab-status')).toHaveText('Paused.');

  const pausedStep = Number(await metric(page, 'STEP').innerText());
  await page.getByRole('button', { name: 'Advance one step' }).click();
  await expect(metric(page, 'STEP')).toHaveText(String(pausedStep + 1), { timeout: 5_000 });

  await page.getByRole('button', { name: 'Reset experiment' }).click();
  await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
  await page.getByLabel('RANDOM SEED').fill('42');
  await page.getByLabel('RANDOM SEED').press('Tab');
  await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
  const parameterDetails = page.locator('details');
  if (!(await parameterDetails.getAttribute('open'))) await parameterDetails.locator('summary').click();
  const growth = page.locator('input[aria-label="Population growth"]');
  await growth.fill('0.6');
  await expect(growth).toHaveValue('0.6');
  await expect(metric(page, 'STEP')).toHaveText('0');
});

test('painting a water tile changes water budget and inspected values without advancing', async ({ page }) => {
  await ready(page);
  const waterCanvas = tile(page, 'view-2').locator('canvas');
  await waterCanvas.scrollIntoViewIfNeeded();
  const box = await waterCanvas.boundingBox();
  expect(box).not.toBeNull();
  if (!box) return;

  await waterCanvas.hover({ position: { x: box.width / 2, y: box.height / 2 } });
  await expect(page.locator('.inspect-panel')).toContainText('Water');
  const beforeStep = await metric(page, 'STEP').innerText();
  const beforeBudget = await page.locator('.water-budget').innerText();
  const beforePainted = paintedWaterBudget(beforeBudget);
  expect(beforePainted).toBe(0);

  await waterCanvas.click({ position: { x: box.width / 2, y: box.height / 2 } });
  await expect(page.locator('.lab-status')).toHaveText(/Paused for intervention: painting water/);
  await expect(metric(page, 'STEP')).toHaveText(beforeStep);
  await expect.poll(async () => paintedWaterBudget(await page.locator('.water-budget').innerText())).toBeGreaterThan(0);
  await expect.poll(async () => page.locator('.water-budget').innerText()).not.toBe(beforeBudget);
});

test('boundary and parameter setup changes reset the ecology field while keeping all tiles', async ({ page }) => {
  await ready(page);
  await page.getByLabel('Boundaries').selectOption('periodic');
  await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
  await expect(page.locator('.lab-status')).toHaveText(/Paused at a new initial state/);
  await expect(page.locator('.field-tile canvas')).toHaveCount(3);
  await expectCanvasHasInk(page, 'view-2');
  await page.getByLabel('Boundaries').selectOption('neumann');
  await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
  await expect(page.locator('.field-tile')).toHaveCount(3);
});

test('all eight experiments load their initial fields and use generic field tile canvases', async ({ page }) => {
  await ready(page);
  for (const preset of presets) {
    const card = page.locator(`[data-preset="${preset}"]`);
    await card.click();
    await expect(card).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('.lab-status')).toHaveText(/loaded\. Paused at its initial state/);
    await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
    await expect(page.locator('.field-tile')).toHaveCount(ecologyPresets.includes(preset) ? 3 : 1);
    await expectCanvasHasInk(page, 'view-1');
  }
});

test('v2 setup links use the real clipboard and preserve layout inputs without evolved state', async ({ page }) => {
  await ready(page);
  const layoutTile = tile(page, 'view-1');
  await layoutTile.getByRole('button', { name: '+ Add layer' }).click();
  await layoutTile.getByLabel('Field for layer 2 in view-1').selectOption('water');
  await layoutTile.getByLabel('Population opacity in view-1').fill('0.35');
  await layoutTile.getByRole('button', { name: 'Move Population layer down' }).click();
  await layoutTile.getByLabel('Paint target for view-1').selectOption('water');
  await page.getByLabel('RANDOM SEED').fill('123');
  await page.getByLabel('RANDOM SEED').press('Tab');
  await page.getByLabel('RESOLUTION').selectOption('32');
  await page.getByLabel('Boundaries').selectOption('periodic');
  await expect(page.getByLabel('RANDOM SEED')).toHaveValue('123');
  await expect(page.getByLabel('RESOLUTION')).toHaveValue('32');
  await expect(page.getByLabel('Boundaries')).toHaveValue('periodic');

  await grantClipboard(page);
  await page.getByRole('button', { name: /Copy setup link/ }).click();
  await expect(page.locator('.lab-status')).toHaveText(/Setup link copied/);
  const sharedUrl = await readClipboardSetupLink(page);
  expect(sharedUrl).toContain('#setup=');
  const payload = await page.evaluate((url) => {
    const encoded = new URL(url).hash.slice(1).replace(/^setup=/, '');
    return JSON.parse(decodeURIComponent(encoded));
  }, sharedUrl);
  expect(payload.version).toBe(2);
  expect(payload.tiles).toHaveLength(3);
  expect(payload.tiles[0].layers.map((layer: { field: string }) => layer.field)).toEqual(['water', 'population']);
  expect(payload.tiles[0].layers[1].opacity).toBe(0.35);
  expect(payload.tiles[0].paintField).toBe('water');

  await page.goto(sharedUrl, { waitUntil: 'load' });
  await expect(page).toHaveURL(/#setup=/);
  await expect(page.locator('.lab-status')).toHaveText(/Shared setup loaded\. Paused at its initial state/);
  await expect(page.locator('.field-tile')).toHaveCount(3);
  await expect(tile(page, 'view-1').locator('.tile-layer').nth(0)).toHaveAttribute('data-layer', 'water');
  await expect(tile(page, 'view-1').getByLabel('Paint target for view-1')).toHaveValue('water');
  await expect(tile(page, 'view-1').getByLabel('Population opacity in view-1')).toHaveValue('0.35');
  await page.getByRole('button', { name: 'Advance one step' }).click();
  await expect(metric(page, 'STEP')).toHaveText('1', { timeout: 5_000 });
  await page.reload();
  await expect(page.locator('.lab-status')).toHaveText(/initial state|Ready\./);
  await expect(metric(page, 'STEP')).toHaveText('0');
  await expect(page.getByLabel('RANDOM SEED')).toHaveValue('123');
  await expect(page.getByLabel('RESOLUTION')).toHaveValue('32');
  await expect(page.getByLabel('Boundaries')).toHaveValue('periodic');
  await expect(page.locator('.field-tile')).toHaveCount(3);
});

test('legacy v1 links load safely and upgrade to a single generic field view', async ({ page }) => {
  await ready(page);
  await page.locator('[data-preset="settlement"]').click();
  await expect(page.locator('.lab-status')).toHaveText(/loaded\. Paused at its initial state/);
  await grantClipboard(page);
  await page.getByRole('button', { name: /Copy setup link/ }).click();
  const currentUrl = await readClipboardSetupLink(page);
  const legacyUrl = await page.evaluate((url) => {
    const parsed = new URL(url);
    const encoded = parsed.hash.slice(1).replace(/^setup=/, '');
    const payload = JSON.parse(decodeURIComponent(encoded));
    payload.version = 1;
    delete payload.tiles;
    parsed.hash = `setup=${encodeURIComponent(JSON.stringify(payload))}`;
    return parsed.toString();
  }, currentUrl);

  await page.goto(legacyUrl, { waitUntil: 'load' });
  await expect(page.locator('.lab-status')).toHaveText(/Shared setup loaded\. Paused at its initial state/);
  await expect(page.locator('.field-tile')).toHaveCount(1);
  await expect(page.locator('.field-tile-heading h3')).toHaveText('Population');
  await expect(page.getByLabel('Field for layer 1 in view-1')).toHaveValue('population');
  await expectCanvasHasInk(page, 'view-1');
});

test('overlay controls support explicit target, opacity, visibility, order, and derived readonly fields', async ({ page }) => {
  await ready(page);
  const first = tile(page, 'view-1');
  await first.getByRole('button', { name: '+ Add layer' }).click();
  await expect(first.locator('.tile-layer')).toHaveCount(2);
  await first.getByLabel('Field for layer 2 in view-1').selectOption('water');
  await first.getByLabel('Population opacity in view-1').fill('0.35');
  await expect(first.locator('.tile-layer').first().locator('output')).toHaveText('35%');
  await first.getByLabel('Show Population in view-1').uncheck();
  await expect(first.locator('[data-field-legend="population"]')).toHaveClass(/is-hidden/);
  await first.getByRole('button', { name: 'Move Population layer down' }).click();
  await expect(first.locator('.tile-layer')).toHaveCount(2);
  await expect(first.locator('.tile-layer').nth(0)).toHaveAttribute('data-layer', 'water');
  await expect(first.locator('.tile-layer').nth(1)).toHaveAttribute('data-layer', 'population');
  await first.getByLabel('Paint target for view-1').selectOption('water');
  await expect(first.locator('.tile-paint-status')).toHaveText(/Paint target · Water/);
  await expect(first.locator('.field-tile-heading h3')).toHaveText('2-field overlay');
  const beforeStep = await metric(page, 'STEP').innerText();
  const beforeBudget = await page.locator('.water-budget').innerText();
  const overlayCanvas = first.locator('canvas');
  await overlayCanvas.scrollIntoViewIfNeeded();
  await overlayCanvas.click({ position: { x: 20, y: 20 } });
  await expect(page.locator('.lab-status')).toHaveText(/Paused for intervention: painting water/);
  await expect(metric(page, 'STEP')).toHaveText(beforeStep);
  await expect.poll(async () => paintedWaterBudget(await page.locator('.water-budget').innerText())).toBeGreaterThan(paintedWaterBudget(beforeBudget));

  const second = tile(page, 'view-2');
  await second.getByRole('button', { name: '+ Add layer' }).click();
  await second.getByLabel('Field for layer 2 in view-2').selectOption('cultivation');
  await second.getByLabel('Paint target for view-2').selectOption('cultivation');
  await expect(second).toHaveClass(/is-readonly/);
  await expect(second.locator('.tile-paint-status')).toHaveText('Cultivation is read only · derived field');
  const derivedCanvas = second.locator('canvas');
  await derivedCanvas.scrollIntoViewIfNeeded();
  await derivedCanvas.click({ position: { x: 20, y: 20 } });
  await expect(page.locator('.lab-status')).toHaveText('Cultivation is derived from population and cannot be painted.');
  await expect(metric(page, 'STEP')).toHaveText(beforeStep);
});

test('twelve arbitrary views share field scales, do not advance, and removed views stay removed', async ({ page }) => {
  await ready(page);
  const add = page.getByRole('button', { name: '+ Add field view' });
  for (let index = 0; index < 9; index += 1) await add.click();
  await expect(page.locator('.field-tile')).toHaveCount(12);
  await expect(metric(page, 'TIME')).toHaveText('0.00');
  await expect(metric(page, 'STEP')).toHaveText('0');

  for (let index = 0; index < 12; index += 1) await page.locator('.field-tile').nth(index).scrollIntoViewIfNeeded();
  const repeatedLegendScales = await page.locator('.field-tile').evaluateAll((nodes) => {
    const scales = new Map<string, string>();
    let repeated = 0;
    for (const node of nodes) {
      const legend = node.querySelector<HTMLElement>('.field-tile-legend:not(.is-hidden)');
      if (!legend) continue;
      const field = legend.dataset.fieldLegend ?? '';
      const scale = legend.textContent?.match(/color 0–[^·]+/)?.[0] ?? '';
      if (!scale) return false;
      if (scales.has(field)) {
        repeated += 1;
        if (scales.get(field) !== scale) return false;
      } else scales.set(field, scale);
    }
    return scales.size >= 3 && repeated >= 1;
  });
  expect(repeatedLegendScales).toBe(true);
  const sizes = await page.locator('.field-tile canvas').evaluateAll((nodes) => nodes.map((node) => {
    const box = node.getBoundingClientRect();
    return [box.width, box.height];
  }));
  expect(Math.max(...sizes.map(([width]) => width)) - Math.min(...sizes.map(([width]) => width))).toBeLessThan(1);
  expect(Math.max(...sizes.map(([, height]) => height)) - Math.min(...sizes.map(([, height]) => height))).toBeLessThan(1);

  while (await page.locator('.field-tile').count() > 1) {
    const last = page.locator('.field-tile').last();
    await last.getByRole('button', { name: /^Remove view/ }).click();
  }
  await expect(page.locator('.field-tile')).toHaveCount(1);
  await page.getByRole('button', { name: 'Advance one step' }).click();
  await expect(metric(page, 'STEP')).toHaveText('1', { timeout: 5_000 });
  await page.setViewportSize({ width: 900, height: 700 });
  await expect(page.locator('.field-tile')).toHaveCount(1);
  await expect(page.locator('.field-tile canvas')).toHaveCount(1);
});

test('invalid shared URLs fall back to safe legacy defaults', async ({ page }) => {
  await page.goto('./#setup=not-valid-json');
  await expect(page.locator('.lab-status')).toHaveText(/invalid or out of date/);
  await expect(page.getByLabel('RANDOM SEED')).toHaveValue('0');
  await expect(page.getByLabel('RESOLUTION')).toHaveValue('64');
  await expect(page.getByLabel('Boundaries')).toHaveValue('neumann');
  await expect(page.locator('[data-preset="settlement"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.field-tile')).toHaveCount(1);
});

test('mobile layout keeps field tile controls within the viewport', async ({ browser }) => {
  const context = await browser.newContext({ ...devices['iPhone 13'] });
  const page = await context.newPage();
  await ready(page);
  const widths = await page.evaluate(() => ({ body: document.body.scrollWidth, viewport: window.innerWidth }));
  expect(widths.body).toBeLessThanOrEqual(widths.viewport + 1);
  await expect(page.locator('.field-tile')).toHaveCount(3);
  await context.close();
});
