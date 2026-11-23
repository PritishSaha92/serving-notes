const assert = require('assert/strict');
module.exports = async function playbackQA(browser, base) {
  const page = await browser.newPage({viewport:{width:390,height:1000},reducedMotion:'no-preference'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  try {
    await page.goto(base,{waitUntil:'networkidle'});
    assert.equal(await page.locator('.playback-toggle').count(),5);
    const cache=page.locator('#cache-explorer');
    await cache.locator('#cache-replay-slots').scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>document.querySelector('#cache-count').textContent==='Step 2 of 3',{},{timeout:6000});
    await cache.locator('.playback-toggle').click();
    await page.waitForTimeout(3800);
    assert.equal(await page.locator('#cache-count').textContent(),'Step 2 of 3');
    await page.evaluate(()=>scrollTo(0,0));await cache.locator('#cache-replay-slots').scrollIntoViewIfNeeded();
    assert.equal(await cache.getAttribute('data-playback'),'paused');
    await page.locator('#cache-reset').click();await cache.locator('.playback-toggle').click();
    await page.evaluate(()=>scrollTo(0,0));
    await page.waitForFunction(()=>document.querySelector('#cache-explorer').dataset.playback==='offscreen');
    await page.waitForTimeout(3800);assert.equal(await page.locator('#cache-count').textContent(),'Step 1 of 3');
    await cache.locator('#cache-replay-slots').scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>document.querySelector('#cache-count').textContent==='Step 2 of 3',{},{timeout:6000});
    await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));});
    await page.waitForTimeout(3800);assert.equal(await page.locator('#cache-count').textContent(),'Step 2 of 3');
    await page.evaluate(()=>{delete document.hidden;document.dispatchEvent(new Event('visibilitychange'));});
    await page.waitForFunction(()=>document.querySelector('#cache-explorer').dataset.playback==='complete',{},{timeout:9000});
    assert.equal(await cache.locator('.playback-toggle').textContent(),'Play');
    await cache.locator('.playback-toggle').click();assert.equal(await page.locator('#cache-count').textContent(),'Step 1 of 3');
    await page.locator('#cache-next').click();assert.equal(await cache.getAttribute('data-playback'),'paused');
    await page.locator('#route-packing-view .route-comparison').scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>document.querySelector('#expert-route-count').textContent==='17',{},{timeout:9000});
    await page.locator('#route-packing-view .playback-toggle').click();
    await page.locator('[data-route-view="partition"]').click();await page.locator('#route-arrays').scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>document.querySelector('#partition-count').textContent==='Step 2 of 3',{},{timeout:6000});
    await page.locator('#route-partition-view .playback-toggle').click();
    await page.locator('[data-version="b"]').click();await page.locator('#attention-explorer .playback-toggle').click();
    await page.locator('.tensor-board').scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>document.querySelector('#attention-step-count').textContent==='Step 2 of 2',{},{timeout:6000});
    await page.locator('#attention-explorer .playback-toggle').click();
    await page.locator('.rounding-stages').scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>document.querySelector('#rounding-count').textContent==='Step 2 of 3',{},{timeout:7000});
    await page.locator('#rounding-controls .playback-toggle').click();
    await page.screenshot({path:'.preview/playback-mobile.png'});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    assert.deepEqual(errors,[]);
    const reduced=await browser.newPage({viewport:{width:390,height:1000},reducedMotion:'reduce'});
    try {
      await reduced.goto(base,{waitUntil:'networkidle'});await reduced.locator('#cache-replay-slots').scrollIntoViewIfNeeded();
      await reduced.waitForTimeout(3800);assert.equal(await reduced.locator('#cache-count').textContent(),'Step 1 of 3');
      assert.equal(await reduced.locator('#cache-explorer .playback-toggle').textContent(),'Play');
      await reduced.locator('#cache-explorer .playback-toggle').click();await reduced.locator('#cache-replay-slots').scrollIntoViewIfNeeded();
      await reduced.waitForFunction(()=>document.querySelector('#cache-count').textContent==='Step 2 of 3',{},{timeout:6000});
    } finally { await reduced.close(); }
    return 'passed: five viewport-driven sequences; manual pause, offscreen/hidden-tab pause, completion/replay and reduced-motion opt-in';
  } finally { await page.close(); }
};
