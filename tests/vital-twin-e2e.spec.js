const { test, expect } = require("@playwright/test");

const base = process.env.MEDPRO_BASE_URL || "http://127.0.0.1:5003";

async function canvasStats(page) {
    return page.locator("#vitalTwinCanvas").evaluate((canvas) => {
        const sample = document.createElement("canvas");
        sample.width = 80;
        sample.height = 80;
        const context = sample.getContext("2d", { willReadFrequently: true });
        context.drawImage(canvas, 0, 0, sample.width, sample.height);
        const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
        let visible = 0;
        let bright = 0;
        let colorRange = 0;
        for (let index = 0; index < pixels.length; index += 4) {
            if (pixels[index + 3] > 8) visible += 1;
            const maximum = Math.max(pixels[index], pixels[index + 1], pixels[index + 2]);
            const minimum = Math.min(pixels[index], pixels[index + 1], pixels[index + 2]);
            if (maximum > 40) bright += 1;
            if (maximum - minimum > 16) colorRange += 1;
        }
        return { visible, bright, colorRange, width: canvas.width, height: canvas.height };
    });
}

test("desktop Vital Twin renders a live body and practical controls", async ({ page }) => {
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await page.setViewportSize({ width: 1440, height: 920 });
    await page.goto(`${base}/#vital`, { waitUntil: "networkidle" });

    await expect(page.locator("[data-page-panel='vital']")).toBeVisible();
    await expect(page.locator("#vitalMemberName")).toHaveText("我");
    await expect(page.locator("#vitalTwinCanvas")).toBeVisible();
    await page.waitForTimeout(900);

    const stats = await canvasStats(page);
    expect(stats.width).toBeGreaterThan(500);
    expect(stats.height).toBeGreaterThan(500);
    expect(stats.visible).toBeGreaterThan(500);
    expect(stats.bright).toBeGreaterThan(80);
    expect(stats.colorRange).toBeGreaterThan(60);

    await page.locator("[data-signal-key='cardio']").click();
    await expect(page.locator("#vitalZoneTitle")).toHaveText("心血管");
    await page.locator("[data-vital-layer='respiratory']").click();
    await expect(page.locator("#vitalZoneTitle")).toHaveText("呼吸与血氧");

    await page.locator("#vitalAddMeasurement").click();
    await expect(page.locator("#vitalMeasurementDialog")).toBeVisible();
    await expect(page.locator("#vitalMeasurementMember")).toHaveValue(/\d+/);
    await page.locator("#vitalMeasurementDialog .modal-header [data-close-dialog]").click();

    await page.screenshot({ path: ".playwright-artifacts/vital-twin-desktop.png", fullPage: true });
    expect(pageErrors).toEqual([]);
});

test("mobile Vital Twin remains readable and keeps WebGL active", async ({ page }) => {
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(`${base}/#vital`, { waitUntil: "networkidle" });
    await expect(page.locator("[data-page-panel='vital']")).toBeVisible();
    await expect(page.locator("#vitalActionList")).toBeVisible();
    await page.waitForTimeout(800);

    const stats = await canvasStats(page);
    expect(stats.width).toBeGreaterThan(300);
    expect(stats.height).toBeGreaterThan(400);
    expect(stats.visible).toBeGreaterThan(280);

    await page.screenshot({ path: ".playwright-artifacts/vital-twin-mobile.png", fullPage: true });
    expect(pageErrors).toEqual([]);
});
