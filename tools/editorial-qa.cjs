const {chromium}=require('playwright');
const {createServer}=require('./server.cjs');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
process.chdir(path.resolve(__dirname,'..'));
(async()=>{
 const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const chrome='C:/Program Files/Google/Chrome/Application/chrome.exe';
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||(fs.existsSync(chrome)?chrome:undefined)});
 try{
  for(const width of [320,390,768,1440,1720]){
   const page=await browser.newPage({viewport:{width,height:1000},reducedMotion:'reduce'});
   await page.goto('http://127.0.0.1:'+server.address().port,{waitUntil:'networkidle'});
   await page.evaluate(()=>document.fonts.ready);
   assert.match(await page.locator('h1').innerText(),/tile\s+changes/);
   assert.equal(await page.locator('a[href$=".excalidraw"]').count(),0);
   assert.doesNotMatch(await page.locator('body').innerText(),/Editable diagram|page design were inspired/);
   for(const name of ['expert-routing','attention-fusion','cache-writes','rounding-boundary','rounding-midpoint','next-investigation']){
    assert.equal(await page.locator(`img[src="assets/diagrams/${name}.svg"]`).isVisible(),true);
   }
   if(width>1100){
    const spacing=await page.evaluate(()=>[...document.querySelectorAll('.annotation-row')].map(row=>({
     row:row.getBoundingClientRect().height,
     prose:row.querySelector('.note-prose').getBoundingClientRect().height
    })));
    assert.ok(spacing.every(x=>Math.abs(x.row-x.prose)<2),'Tall notes must not reserve extra main-column height');
   }
   const details=page.locator('details.technical-note');assert.equal(await details.count(),3);
   for(let i=0;i<3;i++){
    const item=details.nth(i);assert.equal(await item.getAttribute('open'),null);
    await item.locator('summary').click();assert.notEqual(await item.getAttribute('open'),null);
    assert.ok((await item.innerText()).length>350);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    if(i===0&&(width===390||width===1440))await item.screenshot({path:`.preview/editorial-details-${width}.png`});
    await item.locator('summary').click();assert.equal(await item.getAttribute('open'),null);
   }
   await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:`.preview/editorial-top-${width}.png`});
   await page.locator('#performance-budget').scrollIntoViewIfNeeded();
   await page.screenshot({path:`.preview/editorial-budget-${width}.png`});
   await page.locator('#next-investigation').scrollIntoViewIfNeeded();
   await page.screenshot({path:`.preview/editorial-next-${width}.png`});
   assert.equal(await page.locator('.margin-note').count(),14);
   assert.equal(await page.locator('.contents a[href="#next-investigation"]').count(),1);
   assert.match(await page.locator('img[src$="next-investigation.svg"]').getAttribute('alt'),/Proposed investigation/);
   if(width===1440||width===390){
    await page.locator('figure').filter({has:page.locator('img[src$="next-investigation.svg"]')}).screenshot({path:`.preview/next-diagram-${width}.png`});
   }
   if(width===1440||width===390){
    for(const [name,selector] of [['occupancy','img[src$="tile-note.svg"]'],['conversion','img[src$="rounding-note.svg"]'],['boundary','#rounding']]){
     await page.locator(selector).scrollIntoViewIfNeeded();
     await page.screenshot({path:`.preview/spacing-${name}-${width}.png`});
    }
   }
   await page.close();
  }
  fs.writeFileSync('.preview/editorial-qa.json',JSON.stringify({status:'passed',widths:[320,390,768,1440,1720],checks:['three disclosure notes open and close','no document overflow','no editable diagram links or design-credit sentence','budget and header screenshots']},null,2));
  console.log('PASS: final editorial layout and disclosure notes at five widths.');
 }finally{await browser.close();await new Promise(r=>server.close(r))}
})().catch(e=>{console.error(e);process.exitCode=1});
