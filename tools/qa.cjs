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
 for(const width of [320,390,768,1440,1720]){
  const page=await browser.newPage({viewport:{width,height:1000},reducedMotion:'reduce'});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.url().startsWith(base)&&r.status()>=400)errors.push(r.status()+' '+r.url())});
  await page.goto(base,{waitUntil:'networkidle'});await page.evaluate(()=>document.fonts.ready);
  const info=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,broken:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),missing:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash),h1:document.querySelectorAll('h1').length,words:document.querySelector('article').innerText.split(/\s+/).length}));
  assert.ok(info.scroll<=width+1,JSON.stringify(info));assert.deepEqual(info.broken,[]);assert.deepEqual(info.missing,[]);assert.equal(info.h1,1);assert.ok(info.words>3500);
  await page.screenshot({path:`.preview/article-${width}-top.png`});
  for(const id of ['experts','attention','rounding','results','reading','sources']){
   await page.locator('#'+id).scrollIntoViewIfNeeded();await page.screenshot({path:`.preview/article-${width}-${id}.png`});
  }
  if(width===1440||width===390){
   for(let i=0;i<await page.locator('article figure').count();i++)await page.locator('article figure').nth(i).screenshot({path:`.preview/figure-${width}-${i+1}.png`});
   await page.locator('#tile-explorer').screenshot({path:`.preview/explorer-${width}.png`});
  }
  await page.locator('[data-token="410"]').click();assert.equal(await page.locator('#stock-tile').textContent(),'BLOCK_M = 8');assert.equal(await page.locator('#fused-tile').textContent(),'BLOCK_M = 16');
  await page.locator('[data-token="456"]').click();assert.equal(await page.locator('#stock-tile').textContent(),'BLOCK_M = 16');assert.equal(await page.locator('#fused-tile').textContent(),'BLOCK_M = 16');
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
  if(width===1440){await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:'.preview/article-desktop-full.png',fullPage:true});await page.pdf({path:'.preview/article-print.pdf',format:'A4',printBackground:true,margin:{top:'18mm',bottom:'18mm',left:'18mm',right:'18mm'}})}
  await page.close();
 }
 const page=await browser.newPage({javaScriptEnabled:false,viewport:{width:390,height:900}});await page.goto(base,{waitUntil:'networkidle'});assert.ok((await page.locator('article').innerText()).length>20000);await page.screenshot({path:'.preview/article-no-js.png'});
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
 const disk=await browser.newPage({viewport:{width:1280,height:900}});await disk.goto(pathToFileURL(path.join(root,'index.html')).href);await disk.evaluate(()=>document.fonts.ready);await disk.locator('[data-token="410"]').click();assert.equal(await disk.locator('#stock-tile').textContent(),'BLOCK_M = 8');assert.equal(await disk.locator('article img').evaluateAll(images=>images.every(i=>i.complete&&i.naturalWidth>0)),true);await disk.close();
 fs.writeFileSync('.preview/qa.json',JSON.stringify({status:'passed',browser:browser.version(),viewports:report,sourceLedgerViewports:ledgerReport,numberedReferences:refs.ids.length,inlineCitations:refs.citations.length,localLinksChecked:new Set(links).size,noJavaScript:'passed',directFileOpen:'passed',excalidrawScenes:scenes.length},null,2));console.log('PASS: article and source ledger at five widths; numbered margin notes and citations, cross-page anchors, images, local links, interactive boundaries, no-JS reading, direct-file opening and Excalidraw structure.');
 }finally{await browser.close();await new Promise(r=>server.close(r))}
})().catch(e=>{console.error(e);process.exitCode=1});
