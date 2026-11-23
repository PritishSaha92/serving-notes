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
  if (slider) {
    slider.addEventListener('input', update);
    document.querySelectorAll('[data-token]').forEach(b=>b.addEventListener('click',()=>{slider.value=b.dataset.token;update();}));
    update();
    const panel = document.querySelector('#rounding-controls');
    panel.className = 'step-controls';
    panel.innerHTML = '<button type="button" id="rounding-prev">Previous</button><span id="rounding-count" role="status"></span><button type="button" id="rounding-next">Next</button><button type="button" id="rounding-reset">Reset</button>';
    let step = 0;
    const render = () => {
      document.querySelectorAll('[data-rounding-stage]').forEach((el, i) => el.classList.toggle('is-current', i === step));
      panel.querySelector('#rounding-count').textContent = `Step ${step + 1} of 3`;
      panel.querySelector('#rounding-prev').disabled = step === 0;
      panel.querySelector('#rounding-next').disabled = step === 2;
    };
    panel.querySelector('#rounding-prev').addEventListener('click', () => { step = Math.max(0, step - 1); render(); });
    panel.querySelector('#rounding-next').addEventListener('click', () => { step = Math.min(2, step + 1); render(); });
    panel.querySelector('#rounding-reset').addEventListener('click', () => { step = 0; slider.value = 56; update(); render(); });
    // Retain the authored overview sketches alongside the interactive explanation.
    ['rounding-boundary', 'rounding-midpoint'].forEach(name => document.querySelector(`img[src="assets/diagrams/${name}.svg"]`)?.closest('figure').classList.add('rounding-static'));
    render();
    window.NotebookPlayback({root: document.querySelector('#tile-explorer'), controls: panel,
      watch: document.querySelector('.rounding-stages'), getIndex: () => step, count: () => 3,
      setIndex: value => { step = value; render(); }, interval: 5000});
  }
  // Desktop margin notes occupy their own column rather than stretching the
  // prose row to the height of a tall note. Stack only notes that would collide.
  const article = document.querySelector('#article');
  const noteRows = [...document.querySelectorAll('.annotation-row')];
  let notesScheduled = false;
  function layoutNotes() {
    notesScheduled = false;
    if (!article) return;
    article.classList.add('sidenotes-flow');
    if (!matchMedia('(min-width:1101px)').matches) return;
    let previousBottom = -Infinity;
    for (const row of noteRows) {
      const note = row.querySelector('.margin-note');
      const anchor = row.getBoundingClientRect().top + scrollY;
      const top = Math.max(anchor, previousBottom + 18);
      note.style.top = `${top - anchor}px`;
      previousBottom = top + note.getBoundingClientRect().height;
    }
  }
  function scheduleNotes() {
    if (!notesScheduled) { notesScheduled = true; requestAnimationFrame(layoutNotes); }
  }
  addEventListener('resize', scheduleNotes);
  document.addEventListener('toggle', scheduleNotes, true);
  document.querySelectorAll('article img').forEach(img => img.addEventListener('load', scheduleNotes));
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(scheduleNotes);
    document.querySelectorAll('.note-prose, .margin-note').forEach(el => observer.observe(el));
  }
  document.fonts.ready.then(scheduleNotes);
  scheduleNotes();
  let scheduled=false;
  function progress(){const max=document.documentElement.scrollHeight-innerHeight; document.querySelector('.progress').style.width=(max>0 ? 100*scrollY/max : 0)+'%';scheduled=false;}
  addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(progress)}},{passive:true});progress();
})();
