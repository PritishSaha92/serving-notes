(() => {
  'use strict';
  const slider = document.querySelector('#tokens');
  // Source-derived launch heuristic. This models dispatch, not correctness or speed.
  function tile(rows) { return rows < 1024 ? 4 : rows < 8192 ? 8 : rows < 65536 ? 16 : 32; }
  function update() {
    const t = Number(slider.value), stock = 18*t, fused = 20*t;
    document.querySelector('#token-output').textContent = t;
    document.querySelector('#stock-rows').textContent = stock.toLocaleString('en-US')+' rows';
    document.querySelector('#fused-rows').textContent = fused.toLocaleString('en-US')+' rows';
    document.querySelector('#stock-tile').textContent = 'BLOCK_M = '+tile(stock);
    document.querySelector('#fused-tile').textContent = 'BLOCK_M = '+tile(fused);
    document.querySelector('#tile-note').textContent = tile(stock) === tile(fused)
      ? 'Same tile family here. This does not prove numerical equivalence; the final repair still required independent tests.'
      : 'Different tile families. This identifies a configuration boundary, not a prediction that every input will fail.';
  }
  if (slider) { slider.addEventListener('input', update); document.querySelectorAll('[data-token]').forEach(b=>b.addEventListener('click',()=>{slider.value=b.dataset.token;update();}));update(); }
  let scheduled=false;
  function progress(){const max=document.documentElement.scrollHeight-innerHeight; document.querySelector('.progress').style.width=(max>0 ? 100*scrollY/max : 0)+'%';scheduled=false;}
  addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(progress)}},{passive:true});progress();
  const headings=[...document.querySelectorAll('.article h2')], links=[...document.querySelectorAll('.toc ol a')];
  if ('IntersectionObserver' in window) {
    const observer=new IntersectionObserver(entries=>{for(const e of entries){if(e.isIntersecting){links.forEach(a=>{if(a.hash==='#'+e.target.id)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current')})}}},{rootMargin:'-5% 0px -65% 0px'});
    headings.forEach(h=>observer.observe(h));
  }
})();
