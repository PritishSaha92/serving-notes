const {chromium}=require('playwright');
const {createServer}=require('./server.cjs');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {pathToFileURL}=require('url');
const root=path.resolve(__dirname,'..');process.chdir(root);
(async()=>{
 fs.mkdirSync('.preview',{recursive:true});const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const windowsChrome='C:/Program Files/Google/Chrome/Application/chrome.exe';
 const executablePath=process.env.CHROME_PATH||(fs.existsSync(windowsChrome)?windowsChrome:undefined);
 const browser=await chromium.launch({headless:true,executablePath});
 const base='http://127.0.0.1:'+server.address().port;const report=[];
 try{
 const playbackResult=await require('./playback-qa.cjs')(browser,base);
 for(const width of [320,390,768,1440,1720]){
  const page=await browser.newPage({viewport:{width,height:1000},reducedMotion:'reduce'});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.url().startsWith(base)&&r.status()>=400)errors.push(r.status()+' '+r.url())});
  await page.goto(base,{waitUntil:'networkidle'});await page.evaluate(()=>document.fonts.ready);await page.addStyleTag({content:'.progress{visibility:hidden}'});
  const info=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,broken:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),missing:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash),h1:document.querySelectorAll('h1').length,words:document.querySelector('article').innerText.split(/\s+/).length}));
  assert.ok(info.scroll<=width+1,JSON.stringify(info));assert.deepEqual(info.broken,[]);assert.deepEqual(info.missing,[]);assert.equal(info.h1,1);assert.ok(info.words>3500);
  await page.screenshot({path:`.preview/article-${width}-top.png`});
  for(const id of ['experts','attention','rounding','results','reading','sources']){
   await page.locator('#'+id).scrollIntoViewIfNeeded();await page.screenshot({path:`.preview/article-${width}-${id}.png`});
  }
  if(width===1440||width===390){
   for(let i=0;i<await page.locator('article figure:visible').count();i++)await page.locator('article figure:visible').nth(i).screenshot({path:`.preview/figure-${width}-${i+1}.png`});
   await page.locator('#tile-explorer').screenshot({path:`.preview/explorer-${width}.png`});
   await page.locator('#route-explorer').screenshot({path:`.preview/routes-${width}.png`});
   await page.locator('#attention-explorer').screenshot({path:`.preview/attention-stock-${width}.png`});
  }
  assert.equal(await page.locator('.live-figure').count(),3);
  assert.deepEqual(await page.locator('.tensor-shape').allTextContents(),['T × 16 × 64','T × 2 × 64','T × 2 × 64']);
  assert.deepEqual(await page.locator('.byte-row').evaluateAll(rows=>rows.map(row=>[...row.children].map(bit=>bit.textContent).join(''))),['00110101','00110110']);
  for(const row of await page.locator('.byte-row').all())assert.equal(await row.locator('.bit-changed').count(),2);
  if(width<=600){
   const positions=await page.locator('#attention-operations li').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return {left:r.left,top:r.top}}));
   assert.equal(positions[0].top,positions[1].top);assert.equal(positions[0].left,positions[2].left);
  }
  await page.locator('[data-routes="16"]').click();assert.equal(await page.locator('#adaptive-row-tile').textContent(),'M = 32');
  assert.equal(await page.locator('#adaptive-row-tiles .useful').count(),16);assert.equal(await page.locator('#adaptive-row-tiles .unused').count(),16);
  await page.locator('[data-routes="17"]').click();assert.equal(await page.locator('#adaptive-row-tile').textContent(),'M = 64');
  await page.locator('#expert-stock-tile').selectOption('48');assert.equal(await page.locator('#adaptive-row-tile').textContent(),'M = 48');
  await page.locator('[data-routes="12"]').click();await page.locator('#expert-eligible').uncheck();assert.equal(await page.locator('#adaptive-row-tile').textContent(),'M = 48');
  await page.locator('#expert-eligible').check();assert.equal(await page.locator('#adaptive-row-tile').textContent(),'M = 32');
  await page.locator('[data-routes="65"]').click();assert.equal(await page.locator('#adaptive-row-tiles .row-tile').count(),2);assert.equal(await page.locator('#adaptive-row-tiles .unused').count(),31);
  await page.locator('[data-routes="0"]').click();assert.equal(await page.locator('#adaptive-row-tiles .row-cell').count(),0);assert.equal(await page.locator('#adaptive-row-tile').textContent(),'No live rows');
  // Native keyboard operation of the range control must update the model too.
  await page.locator('#expert-routes').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#expert-route-count').textContent(),'1');
  await page.locator('#expert-stock-tile').selectOption('64');await page.locator('[data-routes="12"]').click();
  await page.locator('[data-route-view="partition"]').click();
  await page.locator('#partition-next').click();await page.locator('#partition-next').click();
  const partition=await page.locator('.route-array').evaluateAll(rows=>rows.map(row=>[...row.querySelectorAll('button')].map(b=>({id:Number(b.dataset.value),weight:Number(b.dataset.weight),position:Number(b.dataset.position)}))));
  assert.deepEqual(partition.map(row=>row.map(c=>c.id)),[[32,33,7,32,65,33],[0,1,-1,0,-1,1],[0,-1,-1,0,-1,-1],[-1,1,-1,-1,-1,1]]);
  for(let i=0;i<6;i++){assert.equal(Number(partition[2][i].id>=0)+Number(partition[3][i].id>=0),Number(partition[1][i].id>=0));for(const row of partition){assert.equal(row[i].position,i);assert.equal(row[i].weight,partition[0][i].weight)}}
  await page.locator('.route-array').nth(2).locator('button').nth(3).focus();await page.keyboard.press('Enter');assert.match(await page.locator('#route-inspection').textContent(),/r3.*weight 0.4.*small/);assert.equal(await page.locator('.route-array').nth(2).locator('button').nth(3).evaluate(e=>e===document.activeElement),true);
  if(width===390||width===1440)await page.locator('#route-explorer').screenshot({path:`.preview/partition-${width}.png`});
  await page.locator('#partition-reset').click();assert.equal(await page.locator('#partition-count').textContent(),'Step 1 of 3');
  await page.locator('[data-route-view="packing"]').click();await page.locator('#route-reset').click();assert.equal(await page.locator('#expert-route-count').textContent(),'12');
  assert.equal(await page.locator('#attention-operations .operation').count(),6);assert.equal(await page.locator('#attention-cache').textContent(),'Unchanged');
  assert.match(await page.locator('#temp-k').textContent(),/Not created yet/);
  await page.locator('#attention-next').click();assert.match(await page.locator('#temp-k').textContent(),/FP8 copy/);assert.match(await page.locator('#temp-v').textContent(),/Not created yet/);
  await page.locator('#attention-reset').click();
  for(let step=0;step<5;step++)await page.locator('#attention-next').click();
  assert.equal(await page.locator('#attention-writer').textContent(),'FA3 backend');assert.equal(await page.locator('#attention-next').isDisabled(),true);
  await page.locator('[data-version="a"]').click();assert.equal(await page.locator('#attention-operations .operation').count(),3);assert.match(await page.locator('#attention-path').textContent(),/blocked/);assert.equal(await page.locator('#attention-q').textContent(),'BF16');
  await page.locator('#attention-case').selectOption('verify56');assert.match(await page.locator('#attention-path').textContent(),/BLOCK_M = 8, versus stock's 4/);
  await page.locator('#attention-next').click();assert.equal(await page.locator('#attention-q').textContent(),'FP8');
  await page.locator('[data-version="b"]').click();assert.equal(await page.locator('#attention-operations .operation').count(),2);assert.equal(await page.locator('#attention-q').textContent(),'FP8');assert.equal(await page.locator('#attention-writer').textContent(),'Fused preparation');
  assert.match(await page.locator('#temp-q').textContent(),/Direct output above/);
  for(const name of ['k','v'])assert.match(await page.locator('#temp-'+name).textContent(),/Not materialized/);
  await page.locator('#attention-case').selectOption('verify56');assert.match(await page.locator('#attention-path').textContent(),/BLOCK_M = 4/);
  if(width===1440||width===390)await page.locator('#attention-explorer').screenshot({path:`.preview/attention-repaired-${width}.png`});
  await page.locator('#attention-case').selectOption('wide');assert.equal(await page.locator('#attention-operations .operation').count(),6);assert.match(await page.locator('#attention-path').textContent(),/falls back/);
  await page.locator('#attention-operations .operation').nth(3).click();assert.equal(await page.locator('#attention-writer').textContent(),'FA3 backend');
  if(width===1440||width===390)await page.locator('#attention-explorer').screenshot({path:`.preview/attention-fallback-${width}.png`});
  await page.locator('#attention-case').selectOption('prefill');assert.match(await page.locator('#attention-path').textContent(),/ordinary path for prefill/);assert.equal(await page.locator('#attention-operations .operation').count(),6);
  await page.locator('[data-version="a"]').click();assert.match(await page.locator('#attention-path').textContent(),/blocked candidate overall/);
  await page.locator('[data-version="stock"]').click();await page.locator('#attention-case').selectOption('verify8');
  assert.equal(await page.locator('#cache-count').textContent(),'Step 1 of 3');
  await page.locator('#cache-next').click();
  assert.deepEqual(await page.locator('.cache-slot.just-written').evaluateAll(es=>es.map(e=>Number(e.dataset.location))),[17,42]);
  await page.locator('#cache-next').click();
  assert.deepEqual(await page.locator('.cache-slot').evaluateAll(es=>es.map(e=>e.dataset.value)),['reserved','A','C','D','old','old']);
  assert.deepEqual(await page.locator('.cache-slot.just-written').evaluateAll(es=>es.map(e=>Number(e.dataset.location))),[23,42]);
  if(width===390||width===1440)await page.locator('#cache-explorer').screenshot({path:`.preview/cache-${width}.png`});
  await page.locator('#cache-reset').click();assert.deepEqual(await page.locator('.cache-slot').evaluateAll(es=>es.map(e=>e.dataset.value)),['reserved','old','old','old','old','old']);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'No overflow after interactions');
  await page.locator('[data-token="410"]').click();assert.equal(await page.locator('#stock-tile').textContent(),'BLOCK_M = 8');assert.equal(await page.locator('#fused-tile').textContent(),'BLOCK_M = 16');
  await page.locator('[data-token="456"]').click();assert.equal(await page.locator('#stock-tile').textContent(),'BLOCK_M = 16');assert.equal(await page.locator('#fused-tile').textContent(),'BLOCK_M = 16');
  await page.locator('#rounding-next').focus();await page.keyboard.press('Enter');await page.locator('#rounding-next').click();assert.equal(await page.locator('[data-rounding-stage="2"]').getAttribute('class'),'rounding-stage is-current');
  assert.match(await page.locator('.rounding-walkthrough').textContent(),/0.83984375/);assert.match(await page.locator('.rounding-walkthrough').textContent(),/0x36/);
  await page.locator('#rounding-reset').click();assert.equal(await page.locator('#token-output').textContent(),'56');assert.equal(await page.locator('#rounding-count').textContent(),'Step 1 of 3');
  const notes=await page.evaluate(()=>[...document.querySelectorAll('.annotation-row')].map(row=>{
   const note=row.querySelector('.margin-note'),p=row.querySelector('.note-prose');
   const a=note.getBoundingClientRect(),b=p.getBoundingClientRect();
   return {id:note.id,left:a.left,right:a.right,top:a.top,bottom:a.bottom,proseRight:b.right,proseBottom:b.bottom,visible:getComputedStyle(note).display!=='none'};
  }));
  assert.equal(notes.length,16);assert.ok(notes.every(n=>n.visible));
  if(width>=1101){assert.ok(notes.every(n=>n.left>n.proseRight),JSON.stringify(notes));for(let i=1;i<notes.length;i++)assert.ok(notes[i].top>=notes[i-1].bottom-1,'Overlapping margin notes')}
  else assert.ok(notes.every(n=>n.top>=n.proseBottom-1),'Mobile notes must follow their paragraph');
  assert.equal(await page.locator('.note-marker').count(),notes.length);
  assert.ok(await page.locator('.highlight .mi').count()>0,'Python syntax highlighting');
  assert.deepEqual(errors,[]);report.push({...info,marginNotes:notes.length,errors});
  if(width===1440){await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:'.preview/article-desktop-full.png',fullPage:true});await page.emulateMedia({media:'print'});assert.equal(await page.locator('.live-figure:visible').count(),0);assert.equal(await page.locator('.static-figure:visible').count(),3);assert.equal(await page.locator('.rounding-static:visible').count(),2);assert.equal(await page.locator('#tile-explorer').isVisible(),false);await page.pdf({path:'.preview/article-print.pdf',format:'A4',printBackground:true,margin:{top:'18mm',bottom:'18mm',left:'18mm',right:'18mm'}})}
  await page.close();
 }
 const page=await browser.newPage({javaScriptEnabled:false,viewport:{width:390,height:900}});await page.goto(base,{waitUntil:'networkidle'});assert.ok((await page.locator('article').innerText()).length>20000);await page.screenshot({path:'.preview/article-no-js.png'});
 assert.equal(await page.locator('.live-figure').count(),0);assert.equal(await page.locator('img[src$="expert-routing.svg"]').isVisible(),true);assert.equal(await page.locator('img[src$="attention-fusion.svg"]').isVisible(),true);
 assert.equal(await page.locator('img[src$="cache-writes.svg"]').isVisible(),true);assert.equal(await page.locator('img[src$="rounding-midpoint.svg"]').isVisible(),true);
 const links=await page.locator('a[href]').evaluateAll(as=>as.map(a=>a.getAttribute('href')).filter(h=>!h.startsWith('#')&&!/^https?:/.test(h)));
 const refs=await page.evaluate(()=>{
  const ids=[...document.querySelectorAll('ol.references>li')].map(e=>e.id);
  return {ids,citations:[...document.querySelectorAll('a[href^="#ref-"]')].map(a=>({id:a.hash.slice(1),number:Number(a.textContent)}))};
 });
 assert.equal(refs.ids.length,15);assert.equal(new Set(refs.ids).size,refs.ids.length);
 for(const cite of refs.citations)assert.equal(cite.number,refs.ids.indexOf(cite.id)+1,JSON.stringify(cite));
 const ledgerReport=[];
 for(const width of [320,390,768,1440,1720]){
  const ledger=await browser.newPage({javaScriptEnabled:false,viewport:{width,height:1000}});
  await ledger.goto(base+'/sources.html',{waitUntil:'networkidle'});await ledger.evaluate(()=>document.fonts.ready);
  const info=await ledger.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,h1:document.querySelectorAll('h1').length,sections:document.querySelectorAll('article h2').length,missing:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash)}));
  assert.ok(info.scroll<=width+1,JSON.stringify(info));assert.equal(info.h1,1);assert.equal(info.sections,7);assert.deepEqual(info.missing,[]);
  const more=await ledger.locator('a[href]').evaluateAll(as=>as.map(a=>a.getAttribute('href')).filter(h=>!h.startsWith('#')&&!/^https?:/.test(h)));links.push(...more);
  await ledger.screenshot({path:`.preview/sources-${width}-top.png`});
  if(width===390||width===1440){for(const id of ['papers','implementation','prior-art','social-trail','learning']){await ledger.locator('#'+id).scrollIntoViewIfNeeded();await ledger.screenshot({path:`.preview/sources-${width}-${id}.png`})}}
  ledgerReport.push(info);await ledger.close();
 }
 for(const link of new Set(links)){
  const [file,fragment]=link.split('#');const target=path.join(root,file);assert.ok(fs.existsSync(target),link);
  if(fragment&&file.endsWith('.html'))assert.ok(fs.readFileSync(target,'utf8').includes(`id="${fragment}"`),link);
 }
 const scenes=fs.readdirSync('assets/diagrams').filter(x=>x.endsWith('.excalidraw'));
 for(const name of scenes){const s=JSON.parse(fs.readFileSync('assets/diagrams/'+name));assert.equal(s.type,'excalidraw');assert.ok(s.elements.length>10);assert.equal(new Set(s.elements.map(e=>e.id)).size,s.elements.length)}
 const svgPage=await browser.newPage();
 for(const scene of scenes){
  await svgPage.goto(base+'/assets/diagrams/'+scene.replace('.excalidraw','.svg'));
  await svgPage.evaluate(()=>document.fonts.ready);
  const bounds=await svgPage.evaluate(()=>{const v=document.querySelector('svg').viewBox.baseVal;return {font:document.fonts.check('24px Kalam'),outside:[...document.querySelectorAll('svg text')].filter(t=>{const b=t.getBBox();return b.x<0||b.y<0||b.x+b.width>v.width||b.y+b.height>v.height}).map(t=>t.textContent)}});
  assert.ok(bounds.font,'Embedded diagram font: '+scene);assert.deepEqual(bounds.outside,[],scene);
 }
 await svgPage.close();
 const disk=await browser.newPage({viewport:{width:1280,height:900}});await disk.goto(pathToFileURL(path.join(root,'index.html')).href);await disk.evaluate(()=>document.fonts.ready);await disk.locator('[data-token="410"]').click();assert.equal(await disk.locator('#stock-tile').textContent(),'BLOCK_M = 8');assert.equal(await disk.locator('article img').evaluateAll(images=>images.every(i=>i.complete&&i.naturalWidth>0)),true);await disk.locator('[data-routes="17"]').click();assert.equal(await disk.locator('#adaptive-row-tile').textContent(),'M = 64');await disk.locator('[data-version="b"]').click();assert.equal(await disk.locator('#attention-writer').textContent(),'Fused preparation');await disk.close();
 fs.writeFileSync('.preview/qa.json',JSON.stringify({status:'passed',playback:playbackResult,browser:browser.version(),viewports:report,sourceLedgerViewports:ledgerReport,numberedReferences:refs.ids.length,inlineCitations:refs.citations.length,localLinksChecked:new Set(links).size,noJavaScript:'passed',directFileOpen:'passed',excalidrawScenes:scenes.length,interactiveIllustrations:'Four explanations: row packing and exact route partition with weights; stock/A/repaired-B tensor copies and fallback; two-call cache ownership; dispatch and recorded rounding example. Keyboard, reset, print and no-JS fallbacks passed at five widths.'},null,2));console.log('PASS: article and source ledger at five widths; route and attention interactions, keyboard controls, print/static fallbacks, numbered notes and citations, cross-page anchors, images, local links, interactive boundaries, no-JS reading, direct-file opening and Excalidraw structure.');
 }finally{await browser.close();await new Promise(r=>server.close(r))}
})().catch(e=>{console.error(e);process.exitCode=1});
