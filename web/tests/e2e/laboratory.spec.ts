import { devices, expect, test } from '@playwright/test';

const presets = ['settlement', 'collective_investment', 'overshoot', 'agriculture', 'chemotaxis_demo'];

async function ready(page: import('@playwright/test').Page) {
  await page.goto('./');
  await expect(page.locator('.lab-status')).toHaveText(/initial state|Ready\./, { timeout: 10_000 });
  await expect(page.getByRole('button', { name: 'Run experiment' })).toBeVisible();
}

function metric(page: import('@playwright/test').Page, label: string) {
  return page.locator('.metric').filter({ hasText: label }).locator('strong');
}

async function canvasHasInk(page: import('@playwright/test').Page, name: string) {
  const selector = name.startsWith('Comparison') ? '.map-frame-secondary canvas' : '.map-frame-primary canvas';
  return page.locator(selector).evaluate((canvas) => {
    const element = canvas as HTMLCanvasElement;
    const context = element.getContext('2d');
    if (!context) return false;
    const pixels = context.getImageData(0, 0, element.width, element.height).data;
    let variation = 0;
    for (let index = 0; index < pixels.length; index += 4) variation += pixels[index] + pixels[index + 1] + pixels[index + 2];
    return variation > 1_000;
  });
}

test('loads paused with a live worker, rendered maps, and no console errors', async ({ page }) => {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await ready(page);

  await expect(metric(page, 'TIME')).toHaveText('0.00');
  await expect(metric(page, 'STEP')).toHaveText('0');
  await expect(page.getByRole('button', { name: 'Run experiment' })).toHaveText(/Run experiment/);
  await expect(page.locator('.map-frame-primary canvas')).toBeVisible();
  expect(await canvasHasInk(page, 'Primary simulation field map')).toBe(true);
  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
  expect(pageErrors, `page errors: ${pageErrors.join('\n')}`).toEqual([]);
});

test('run, pause, step, reset, and parameter/setup controls update state', async ({ page }) => {
  await ready(page);
  const run = page.locator('.button-primary');
  await run.click();
  await expect(run).toHaveText(/Pause experiment/);
  await expect(metric(page, 'TIME')).not.toHaveText('0.00', { timeout: 5_000 });
  await run.click();
  await expect(run).toHaveText(/Run experiment/);
  await expect(page.locator('.lab-status')).toHaveText('Paused.');

  const stepBefore = Number(await metric(page, 'STEP').innerText());
  await page.getByRole('button', { name: 'Advance one step' }).click();
  await expect(metric(page, 'STEP')).toHaveText(String(stepBefore + 1), { timeout: 5_000 });

  await page.getByRole('button', { name: 'Reset experiment' }).click();
  await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
  await page.getByLabel('RANDOM SEED').fill('42');
  await page.getByLabel('RANDOM SEED').press('Tab');
  await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
  const parameterDetails = page.locator('details');
  if (await parameterDetails.count() && !(await parameterDetails.first().getAttribute('open'))) await parameterDetails.first().locator('summary').click();
  const growth = page.locator('input[aria-label="Population growth"]');
  await growth.fill('0.6');
  await expect(growth).toHaveValue('0.6');
  await expect(page.locator('.lab-status')).toHaveText(/Paused at a new initial state/);
});

test('painting pauses a run and increases population mass', async ({ page }) => {
  await ready(page);
  const run = page.locator('.button-primary');
  await run.click();
  await expect(run).toHaveText(/Pause experiment/);
  await expect(metric(page, 'TIME')).not.toHaveText('0.00', { timeout: 5_000 });
  await run.click();
  await expect(run).toHaveText(/Run experiment/);
  await expect(page.locator('.lab-status')).toHaveText('Paused.');
  const pausedStep = Number(await metric(page, 'STEP').innerText());
  const pausedMass = Number((await metric(page, 'POPULATION').innerText()).replaceAll(',', ''));
  const canvas = page.locator('.map-frame-primary canvas');
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  if (!box) return;
  await canvas.click({ position: { x: box.width / 2, y: box.height / 2 } });
  await expect(run).toHaveText(/Run experiment/);
  await expect(page.locator('.lab-status')).toHaveText(/Paused for field intervention/);
  await expect(metric(page, 'STEP')).toHaveText(String(pausedStep));
  await expect.poll(async () => Number((await metric(page, 'POPULATION').innerText()).replaceAll(',', ''))).toBeGreaterThan(pausedMass);
});

test('boundary selectors reset setup and comparison renders a second canvas', async ({ page }) => {
  await ready(page);
  await page.getByLabel('Boundaries').selectOption('periodic');
  await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
  await expect(page.locator('.lab-status')).toHaveText(/Paused at a new initial state/);
  await page.getByLabel('Comparison field').selectOption('food');
  await expect(page.locator('.map-frame-secondary canvas')).toBeVisible();
  expect(await canvasHasInk(page, 'Comparison simulation field map')).toBe(true);
  await page.getByLabel('Comparison field').selectOption('');
  await expect(page.locator('.map-frame-secondary canvas')).toBeHidden();
  await page.getByLabel('Boundaries').selectOption('neumann');
  await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
});

test('all five experiments load and select their initial field', async ({ page }) => {
  await ready(page);
  for (const preset of presets) {
    const card = page.locator(`[data-preset="${preset}"]`);
    await card.click();
    await expect(card).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('.lab-status')).toHaveText(/loaded\. Paused at its initial state/);
    await expect(metric(page, 'STEP')).toHaveText('0', { timeout: 5_000 });
  }
});

test('shared setup reload preserves setup inputs while starting evolved state at zero', async ({ page }) => {
  await ready(page);
  await page.getByLabel('RANDOM SEED').fill('123');
  await page.getByLabel('RANDOM SEED').press('Tab');
  await page.getByLabel('RESOLUTION').selectOption('32');
  await page.getByLabel('Boundaries').selectOption('periodic');
  await expect(page.getByLabel('RANDOM SEED')).toHaveValue('123');
  await expect(page.getByLabel('RESOLUTION')).toHaveValue('32');
  await expect(page.getByLabel('Boundaries')).toHaveValue('periodic');
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'], { origin: new URL(page.url()).origin });
  await page.getByRole('button', { name: /Copy setup link/ }).click();
  const fallbackLink = page.locator('input[aria-label="Setup link"]');
  await expect.poll(async () => {
    if (await fallbackLink.count()) return fallbackLink.inputValue();
    return await page.evaluate(async () => navigator.clipboard?.readText() || '');
  }).toMatch(/#setup=/);
  const hasFallback = await fallbackLink.count();
  const sharedUrl = hasFallback
    ? await fallbackLink.inputValue()
    : await page.evaluate(async () => navigator.clipboard?.readText() || '');
  expect(sharedUrl).toContain('#setup=');
  await page.goto(sharedUrl, { waitUntil: 'load' });
  await expect(page).toHaveURL(/#setup=/);
  await page.reload();
  await expect(page.locator('.lab-status')).toHaveText(/initial state|Ready\./);
  await page.getByRole('button', { name: 'Advance one step' }).click();
  await expect(metric(page, 'STEP')).toHaveText('1', { timeout: 5_000 });
  await page.reload();
  await expect(page.locator('.lab-status')).toHaveText(/initial state|Ready\./);
  await expect(metric(page, 'STEP')).toHaveText('0');
  await expect(page.getByLabel('RANDOM SEED')).toHaveValue('123');
  await expect(page.getByLabel('RESOLUTION')).toHaveValue('32');
  await expect(page.getByLabel('Boundaries')).toHaveValue('periodic');
});

test('invalid shared URL falls back to safe defaults with an error status', async ({ page }) => {
  await page.goto('./#setup=not-valid-json');
  await expect(page.locator('.lab-status')).toHaveText(/invalid or out of date/);
  await expect(page.getByLabel('RANDOM SEED')).toHaveValue('0');
  await expect(page.getByLabel('RESOLUTION')).toHaveValue('64');
  await expect(page.getByLabel('Boundaries')).toHaveValue('neumann');
});

test('mobile layout keeps the document within the viewport', async ({ browser }) => {
  const context = await browser.newContext({ ...devices['iPhone 13'] });
  const page = await context.newPage();
  await ready(page);
  const widths = await page.evaluate(() => ({ body: document.body.scrollWidth, viewport: window.innerWidth }));
  expect(widths.body).toBeLessThanOrEqual(widths.viewport + 1);
  await context.close();
});
