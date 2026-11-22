const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{const b=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}); const p=await b.newPage({viewport:{width:1440,height:1000}}); for(const [id,url] of [['humansand','https://humansand.ai/blog/gpu-accelerated-weight-teleportation'],['boehm','https://siboehm.com/articles/22/CUDA-MMM']]){try{await p.goto(url,{waitUntil:'networkidle',timeout:45000}); await p.screenshot({path:'.preview/'+id+'-top.png'});fs.writeFileSync('.preview/'+id+'-text.txt',await p.locator('body').innerText()); await p.evaluate(()=>scrollTo(0,1400));await p.screenshot({path:'.preview/'+id+'-body.png'}); console.log(id,'rendered');}catch(e){console.log(id,e.message)}}await b.close()})();

