/* Teaching models for the article. No timing simulation or GPU execution. */
(() => {
  'use strict';

  function mount(imageName, id, markup, initialize) {
    const image = document.querySelector(`article img[src="assets/diagrams/${imageName}.svg"]`);
    if (!image) return;
    const figure = image.closest('figure');
    const wrapper = document.createElement('div');
    wrapper.className = 'interactive-illustration';
    const panel = document.createElement('section');
    panel.className = 'live-figure';
    panel.id = id;
    panel.setAttribute('aria-labelledby', `${id}-title`);
    panel.innerHTML = markup + `<div class="figure-downloads"><a href="assets/diagrams/${imageName}.svg">Open static diagram</a></div>`;
    // Enhance only after initialization succeeds; the authored figure is the fallback.
    initialize(panel);
    figure.before(wrapper);
    wrapper.append(panel, figure);
    figure.classList.add('static-figure');
    wrapper.classList.add('is-enhanced');
  }

  mount('expert-routing', 'route-explorer', `
    <div class="live-eyebrow">Explore the expert schedule</div>
    <h3 id="route-explorer-title">Same routes. Different row tiles.</h3>
    <p class="live-intro">After mapping routes to this rank, count the rows assigned to one local expert. Try 16 routes, then 17.</p>
    <div class="route-controls">
      <div class="route-slider"><label for="expert-routes">Routes to this expert <output id="expert-route-count" for="expert-routes">12</output></label><input id="expert-routes" type="range" min="0" max="96" step="1" value="12"></div>
      <label class="select-control" for="expert-stock-tile">Stock row tile<select id="expert-stock-tile"><option value="64">M = 64</option><option value="48">M = 48</option></select></label>
    </div>
    <div class="live-presets" role="group" aria-label="Example expert loads"><button type="button" data-routes="0">0</button><button type="button" data-routes="12">12</button><button type="button" data-routes="16">16</button><button type="button" data-routes="17">17</button><button type="button" data-routes="50">50</button><button type="button" data-routes="65">65</button></div>
    <label class="eligibility-control"><input id="expert-eligible" type="checkbox" checked> This forward is eligible for adaptive scheduling</label>
    <div class="route-comparison">
      <div class="route-column"><div class="route-column-title">Stock <span id="stock-row-tile">M = 64</span></div><div class="row-tiles" id="stock-row-tiles" aria-hidden="true"></div><p class="row-stat" id="stock-row-stat"></p></div>
      <div class="route-column adaptive-column"><div class="route-column-title">Adaptive <span id="adaptive-row-tile">M = 32</span></div><div class="row-tiles" id="adaptive-row-tiles" aria-hidden="true"></div><p class="row-stat" id="adaptive-row-stat"></p></div>
    </div>
    <div class="row-legend"><span><i class="row-key useful"></i>Useful row</span><span><i class="row-key unused"></i>Unused row slot</span></div>
    <p class="live-explanation" id="route-decision" role="status" aria-live="polite"></p>
    <p class="live-footnote">Each square is one row slot; each outlined group is one tile. This illustrates row packing, not memory traffic or speed. The real dispatch also checks the forward shape and expert map; its eligible range here is 116–512 tokens. Outside that range, the stock schedule stays in use.</p>
  `, panel => {
    const slider = panel.querySelector('#expert-routes');
    const stockSelect = panel.querySelector('#expert-stock-tile');
    const eligible = panel.querySelector('#expert-eligible');
    function draw(target, count, size) {
      target.replaceChildren();
      const tiles = Math.ceil(count / size);
      for (let t = 0; t < tiles; t++) {
        const group = document.createElement('div');
        group.className = 'row-tile';
        for (let i = 0; i < size; i++) {
          const cell = document.createElement('span');
          cell.className = 'row-cell ' + (t * size + i < count ? 'useful' : 'unused');
          group.append(cell);
        }
        target.append(group);
      }
      if (!tiles) { const empty = document.createElement('span'); empty.className = 'empty-expert'; empty.textContent = 'No routed rows'; target.append(empty); }
      return {tiles, unused: tiles * size - count};
    }
    function render() {
      const count = Number(slider.value), stock = Number(stockSelect.value);
      const small = eligible.checked && count > 0 && count <= 16;
      const adaptive = small ? 32 : stock;
      panel.querySelector('#expert-route-count').textContent = count;
      for (const [prefix, size] of [['stock', stock], ['adaptive', adaptive]]) {
        const result = draw(panel.querySelector(`#${prefix}-row-tiles`), count, size);
        panel.querySelector(`#${prefix}-row-tile`).textContent = count ? `M = ${size}` : 'No live rows';
        panel.querySelector(`#${prefix}-row-stat`).textContent = `${count} useful · ${result.unused} unused · ${result.tiles} ${result.tiles === 1 ? 'tile' : 'tiles'}`;
      }
      panel.querySelector('#route-decision').textContent = !count
        ? 'No routes reach this expert, so there are no live rows to pack.'
        : !eligible.checked
          ? `This forward uses the stock schedule. Both paths keep M = ${stock}, even for a lightly loaded expert.`
          : small
            ? `${count} ≤ 16: this expert takes the M = 32 path. The route threshold is 16; the tile still has 32 row slots.`
            : `${count} > 16: this expert keeps the stock M = ${stock} tile. Other experts in the same forward can still take the small class.`;
      panel.querySelectorAll('[data-routes]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.routes) === count)));
    }
    slider.addEventListener('input', render);
    stockSelect.addEventListener('change', render);
    eligible.addEventListener('change', render);
    panel.querySelectorAll('[data-routes]').forEach(button => button.addEventListener('click', () => { slider.value = button.dataset.routes; render(); }));
    render();
    installRoutePartition(panel);
  });

  mount('attention-fusion', 'attention-explorer', `
    <div class="live-eyebrow">Step through attention preparation</div>
    <h3 id="attention-explorer-title">Where do the separate operations go?</h3>
    <p class="live-intro">The walkthrough plays when its tensors come into view. Pause to inspect a step, or choose a version and press Play. Follow Q, K and V as temporary tensors appear—or disappear.</p>
    <div class="version-controls" role="group" aria-label="Attention preparation version"><button type="button" data-version="stock" aria-pressed="true">Stock</button><button type="button" data-version="a" aria-pressed="false">A · historical</button><button type="button" data-version="b" aria-pressed="false">Repaired B</button></div>
    <label class="select-control attention-case" for="attention-case">Example call<select id="attention-case"><option value="verify8">Verify · T = 8</option><option value="verify56">Verify boundary · T = 56</option><option value="wide">Wide kernel case · T = 3641</option><option value="prefill">Prefill · T = 4096</option></select></label>
    <p class="path-notice" id="attention-path"></p>
    <div class="attention-layout">
      <div><div class="diagram-label">Operation sequence</div><ol class="operation-flow" id="attention-operations" aria-label="Preparation and attention operations"></ol></div>
      <div class="tensor-state"><div class="diagram-label">After the selected operation</div>
        <div class="tensor-board">
          <div class="tensor-card query"><strong>Q</strong><div class="tensor-shape">T × 16 × 64</div><div class="tensor-strips" aria-hidden="true"></div><span id="attention-q"></span><small>Query to next stage</small></div>
          <div class="tensor-card key"><strong>K</strong><div class="tensor-shape">T × 2 × 64</div><div class="tensor-strips" aria-hidden="true"></div><span id="attention-k"></span><small>Key output</small></div>
          <div class="tensor-card value"><strong>V</strong><div class="tensor-shape">T × 2 × 64</div><div class="tensor-strips" aria-hidden="true"></div><span>BF16</span><small>Value input</small></div>
          <div class="tensor-temp query" id="temp-q"></div><div class="tensor-temp key" id="temp-k"></div><div class="tensor-temp value" id="temp-v"></div>
          <div class="cache-bank"><span class="diagram-label">Persistent KV cache</span><div class="cache-strips" aria-hidden="true"><i>K</i><i>V</i><i>K</i><i>V</i><i>…</i></div><span id="attention-cache"></span></div>
        </div>
        <p class="rounding-contract">Logical views on one rank: T tokens × heads × 64 components. Physical strides are omitted. Q/K keep BF16 rounding even when no BF16 query tensor is stored.</p>
        <dl><dt>Cache writer</dt><dd id="attention-writer"></dd></dl><div class="consumer-state" id="attention-consumer"></div></div>
    </div>
    <div class="step-controls"><button type="button" id="attention-prev">Previous</button><span id="attention-step-count"></span><button type="button" id="attention-next">Next</button><button type="button" id="attention-reset">Reset</button></div>
    <p class="live-explanation" id="attention-detail" role="status" aria-live="polite"></p>
    <p class="live-footnote">Conceptual order, not a timing trace. Blocks show formats and ownership, not measured storage sizes. These examples use FA3 with an FP8 cache; padding slot 0 is skipped. Historical A failed its numerical tests.</p>
  `, panel => {
    let version = 'stock', index = 0, steps = [];
    const scenarios = {
      verify8: {tokens: 8, eligible: true},
      verify56: {tokens: 56, eligible: true},
      wide: {tokens: 3641, eligible: true},
      prefill: {tokens: 4096, eligible: false}
    };
    const tile = rows => rows < 1024 ? 4 : rows < 8192 ? 8 : rows < 65536 ? 16 : 32;
    const state = (q, cache = 'Unchanged', writer = 'No write yet', ready = false) => ({q, k: 'BF16', cache, writer, ready});
    const fa3 = writer => ({title: 'FA3 attention', sub: 'Same attention implementation', detail: 'FA3 consumes the FP8 query and cache. Fusion changes their preparation; the attention algorithm remains the same.', ...state('FP8', 'K and V stored as FP8', writer, true), consumed: true});
    const stockSteps = () => [
      {title: 'QK normalization + RoPE', sub: 'Produce BF16 Q and K', detail: 'Normalize Q/K and apply RoPE where needed. The BF16 outputs preserve the rounding boundary used by the later FP8 conversions.', ...state('BF16')},
      {title: 'Convert K to FP8', sub: 'Create a temporary K tensor', detail: 'The BF16 key output remains available. A separate cast produces FP8 keys for the cache; no cache slot has been written yet.', ...state('BF16')},
      {title: 'Convert V to FP8', sub: 'Create a temporary V tensor', detail: 'Cast V to FP8. Both cache inputs are now prepared, but they still need to be written to the assigned locations.', ...state('BF16')},
      {title: 'Write the KV cache', sub: 'Ordinary backend store', detail: 'The backend writes the prepared FP8 keys and values. Reserved padding slot 0 and all unrelated slots remain untouched.', ...state('BF16', 'K and V stored as FP8', 'FA3 backend')},
      {title: 'Convert Q to FP8', sub: 'Prepare the query for FA3', detail: 'Cast the BF16 query to FP8. Query and cache inputs are now ready for attention.', ...state('FP8', 'K and V stored as FP8', 'FA3 backend', true)},
      fa3('FA3 backend')
    ];
    function selectPath() {
      const scenario = scenarios[panel.querySelector('#attention-case').value];
      const stockTile = tile(18 * scenario.tokens);
      const fallback = !scenario.eligible || (version === 'b' && stockTile === 32);
      const notice = panel.querySelector('#attention-path');
      notice.classList.toggle('historical-warning', version === 'a');
      if (version === 'stock' || fallback) {
        steps = stockSteps();
        notice.textContent = version === 'stock'
          ? `Stock preparation: five separate preparation operations, then FA3. Q/K launch tile: BLOCK_M = ${stockTile}.`
          : !scenario.eligible
            ? `${version === 'a' ? 'Historical A' : 'Repaired B'} takes the ordinary path for prefill. The backend retains the cache write and the separate query conversion.`
            : `Repaired B falls back: 18 × ${scenario.tokens} = ${(18 * scenario.tokens).toLocaleString('en-US')} Q/K rows selects BLOCK_M = 32. That whole family uses stock preparation and backend cache writing.`;
      } else {
        const isB = version === 'b';
        steps = [{title: isB ? 'Fused Q / K / V preparation' : 'Fused QK + KV-store preparation', sub: isB ? 'FP8 query and cache in one operation' : 'BF16 query; FP8 cache written here', detail: isB
          ? 'Normalize Q/K, apply RoPE, preserve BF16 rounding, emit FP8 Q and write FP8 K/V to the cache. The key output stays BF16. The backend skips its store, and its same-dtype query cast becomes a no-op.'
          : 'A combines normalization, RoPE, FP8 K/V conversion and cache writing, but keeps a BF16 query. This is its intended dataflow; later boundary tests found numerical failures.', ...state(isB ? 'FP8' : 'BF16', 'K and V stored as FP8', 'Fused preparation', isB)}];
        if (!isB) steps.push({title: 'Convert Q to FP8', sub: 'Separate query cast remains', detail: 'The backend still casts the BF16 query before FA3. It skips its ordinary cache write because the fused operation already wrote K/V.', ...state('FP8', 'K and V stored as FP8', 'Fused preparation', true)});
        steps.push(fa3('Fused preparation'));
        notice.textContent = isB
          ? `Repaired B: choose BLOCK_M = ${stockTile} from stock's 18 × ${scenario.tokens} Q/K rows; still cover all 20 × ${scenario.tokens} Q/K/V rows. One preparation operation, then FA3.`
          : `Historical A is blocked by numerical failures. Its original tile uses 20 × ${scenario.tokens} rows: BLOCK_M = ${tile(20 * scenario.tokens)}, versus stock's ${stockTile}. The diagram shows the intended sequence, not validated numerical equivalence.`;
      }
      if (version === 'a' && fallback) notice.textContent += ' A remains a blocked candidate overall.';
      index = 0;
      const list = panel.querySelector('#attention-operations');
      list.replaceChildren();
      steps.forEach((step, i) => {
        const li = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'operation';
        if (step.consumed) button.classList.add('unchanged-operation');
        const title = document.createElement('span'); title.className = 'operation-title'; title.textContent = step.title;
        const number = document.createElement('span'); number.className = 'operation-number'; number.textContent = `${i + 1}. `; number.setAttribute('aria-hidden', 'true'); title.prepend(number);
        const sub = document.createElement('span'); sub.className = 'operation-sub'; sub.textContent = step.sub;
        button.append(title, sub); button.addEventListener('click', () => {index = i; renderStep();});
        li.append(button); list.append(li);
      });
      panel.querySelectorAll('[data-version]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.version === version)));
      renderStep();
    }
    function renderStep() {
      const step = steps[index];
      const fused = steps[0].writer === 'Fused preparation';
      const tempStates = fused
        ? [version === 'b' ? 'Direct output above' : index >= 1 ? 'FP8 copy' : 'Not created yet', 'Not materialized', 'Not materialized']
        : [index >= 4 ? 'FP8 copy' : 'Not created yet', index >= 1 ? 'FP8 copy' : 'Not created yet', index >= 2 ? 'FP8 copy' : 'Not created yet'];
      ['q', 'k', 'v'].forEach((name, i) => {
        const box = panel.querySelector(`#temp-${name}`);
        box.replaceChildren();
        const label = document.createElement('strong'); label.textContent = name === 'q' ? 'Extra FP8 Q copy' : `FP8 ${name.toUpperCase()} temporary`;
        const value = document.createElement('span'); value.textContent = tempStates[i];
        box.append(label, value); box.classList.toggle('is-materialized', tempStates[i] === 'FP8 copy');
      });
      panel.querySelector('.cache-bank').classList.toggle('is-written', step.cache !== 'Unchanged');
      panel.querySelectorAll('.operation').forEach((button, i) => {
        button.setAttribute('aria-current', i === index ? 'step' : 'false');
        button.classList.toggle('is-complete', i < index);
      });
      for (const key of ['q', 'k', 'cache', 'writer']) panel.querySelector(`#attention-${key}`).textContent = step[key];
      const consumer = panel.querySelector('#attention-consumer');
      consumer.textContent = step.consumed ? 'FA3 has consumed the prepared inputs.' : step.ready ? 'Query and cache are ready for FA3.' : 'Preparation continues before FA3.';
      consumer.classList.toggle('is-ready', step.ready);
      panel.querySelector('#attention-detail').textContent = step.detail;
      panel.querySelector('#attention-step-count').textContent = `Step ${index + 1} of ${steps.length}`;
      panel.querySelector('#attention-prev').disabled = index === 0;
      panel.querySelector('#attention-next').disabled = index === steps.length - 1;
    }
    panel.querySelectorAll('[data-version]').forEach(button => button.addEventListener('click', () => {version = button.dataset.version; selectPath();}));
    panel.querySelector('#attention-case').addEventListener('change', selectPath);
    panel.querySelector('#attention-prev').addEventListener('click', () => {if (index > 0) {index--; renderStep();}});
    panel.querySelector('#attention-next').addEventListener('click', () => {if (index < steps.length - 1) {index++; renderStep();}});
    panel.querySelector('#attention-reset').addEventListener('click', () => { index = 0; renderStep(); });
    panel.querySelectorAll('.tensor-strips').forEach(strip => { for (let i = 0; i < 6; i++) strip.append(document.createElement('i')); });
    selectPath();
    window.NotebookPlayback({root: panel, controls: panel.querySelector('.step-controls'),
      watch: panel.querySelector('.tensor-board'), getIndex: () => index, count: () => steps.length,
      setIndex: value => { index = value; renderStep(); }});
  });

  function installRoutePartition(panel) {
    const packing = document.createElement('div'); packing.id = 'route-packing-view';
    const start = panel.querySelector('.route-controls');
    const nodes = []; for (let node = start; node && !node.classList.contains('figure-downloads'); node = node.nextElementSibling) nodes.push(node);
    start.before(packing); nodes.forEach(node => packing.append(node));
    const switches = document.createElement('div'); switches.className = 'version-controls';
    switches.setAttribute('role', 'group'); switches.setAttribute('aria-label', 'Expert diagram view');
    switches.innerHTML = '<button type="button" data-route-view="packing" aria-pressed="true">Row packing</button><button type="button" data-route-view="partition" aria-pressed="false">Follow the routes</button>';
    packing.before(switches);
    const partition = document.createElement('div'); partition.id = 'route-partition-view'; partition.hidden = true;
    partition.innerHTML = `<p class="live-intro">This rank owns global experts 32–63. Follow six route positions from a larger forward. Their weights stay attached to the same positions.</p>
      <div class="route-histogram"><span>Full-forward counts</span><strong>Local expert 0: 12 routes → M32</strong><strong>Local expert 1: 50 routes → stock tile</strong></div>
      <div id="route-arrays"></div>
      <p class="route-inspection" id="route-inspection">Select a route position to inspect its weight and destination.</p>
      <div class="step-controls"><button type="button" id="partition-prev">Previous</button><span id="partition-count"></span><button type="button" id="partition-next">Next</button><button type="button" id="partition-reset">Reset</button></div>
      <p class="live-explanation" id="partition-detail" role="status"></p>
      <p class="live-footnote">Illustrative excerpt, not six routes from a captured request. Counts include the other routes in the forward. −1 marks an inactive entry; splitting the IDs does not compact or reorder the associated weights.</p>`;
    packing.after(partition);
    const globalIds = [32, 33, 7, 32, 65, 33], weights = [0.2, 0.3, 0.1, 0.4, 0.2, 0.3];
    const local = globalIds.map(id => id >= 32 && id < 64 ? id - 32 : -1);
    const counts = [12, 50];
    const small = local.map(id => id >= 0 && counts[id] <= 16 ? id : -1);
    const large = local.map(id => id >= 0 && counts[id] > 16 ? id : -1);
    let step = 0, selected = 0;
    const explanations = [
      'Start with global expert IDs. Each column keeps its route position throughout the example.',
      'Subtract this rank’s expert offset for local experts. Global experts 7 and 65 belong elsewhere, so their entries become −1.',
      'Use the full-forward histogram: expert 0 takes the small class, expert 1 keeps the stock class. Every local route is active in exactly one output array.'
    ];
    function renderPartition() {
      const arrays = partition.querySelector('#route-arrays'); arrays.replaceChildren();
      [['Global expert', globalIds, 0], ['Local expert', local, 1], ['Small class · M32', small, 2], ['Large class · stock', large, 2]].forEach(([label, values, visibleAt]) => {
        const row = document.createElement('div'); row.className = 'route-array';
        const heading = document.createElement('div'); heading.className = 'diagram-label'; heading.textContent = label;
        const cells = document.createElement('div'); cells.className = 'route-cells';
        values.forEach((value, position) => {
          const cell = document.createElement('button'); cell.type = 'button';
          const known = step >= visibleAt;
          cell.dataset.position = position; cell.dataset.value = known ? value : '';
          cell.dataset.weight = weights[position];
          cell.className = 'route-position' + (known && value < 0 ? ' is-inactive' : '');
          cell.setAttribute('aria-pressed', String(position === selected));
          cell.setAttribute('aria-label', `${label}, position ${position}, ${known ? 'expert ' + value : 'not computed yet'}, weight ${weights[position]}`);
          cell.innerHTML = `<small>r${position}</small><strong>${known ? value : '·'}</strong>`;
          cell.addEventListener('click', () => {
            const rowIndex = [...arrays.children].indexOf(row);
            selected = position; renderPartition();
            arrays.children[rowIndex].querySelectorAll('button')[position].focus({preventScroll: true});
          }); cells.append(cell);
        });
        row.append(heading, cells); arrays.append(row);
      });
      partition.querySelector('#route-inspection').textContent = `Position r${selected} · weight ${weights[selected].toFixed(1)} · ` + (step === 0 ? `global expert ${globalIds[selected]}` : local[selected] < 0 ? 'inactive on this rank' : step === 1 ? `local expert ${local[selected]}` : `${small[selected] >= 0 ? 'small' : 'large'} class, local expert ${local[selected]}. Active once; weight unchanged.`);
      partition.querySelector('#partition-detail').textContent = explanations[step];
      partition.querySelector('#partition-count').textContent = `Step ${step + 1} of 3`;
      partition.querySelector('#partition-prev').disabled = step === 0; partition.querySelector('#partition-next').disabled = step === 2;
    }
    switches.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
      const isPacking = button.dataset.routeView === 'packing'; packing.hidden = !isPacking; partition.hidden = isPacking;
      switches.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    }));
    partition.querySelector('#partition-prev').addEventListener('click', () => { step = Math.max(0, step - 1); renderPartition(); });
    partition.querySelector('#partition-next').addEventListener('click', () => { step = Math.min(2, step + 1); renderPartition(); });
    partition.querySelector('#partition-reset').addEventListener('click', () => { step = 0; selected = 0; renderPartition(); });
    const reset = document.createElement('button'); reset.type = 'button'; reset.textContent = 'Reset'; reset.id = 'route-reset';
    reset.addEventListener('click', () => { panel.querySelector('#expert-routes').value = 12; panel.querySelector('#expert-stock-tile').value = 64; panel.querySelector('#expert-eligible').checked = true; panel.querySelector('#expert-routes').dispatchEvent(new Event('input')); });
    packing.querySelector('.live-presets').append(reset); renderPartition();
    const playbackControls = document.createElement('div'); playbackControls.className = 'step-controls packing-playback';
    playbackControls.innerHTML = '<button type="button" id="packing-prev">Previous</button><span id="packing-count"></span><button type="button" id="packing-next">Next</button>';
    packing.querySelector('.row-legend').after(playbackControls);
    playbackControls.append(reset);
    const examples = [12, 16, 17, 50, 65, 0]; let packingIndex = 0;
    function showPacking(value) {
      packingIndex = value; panel.querySelector('#expert-routes').value = examples[value];
      panel.querySelector('#expert-routes').dispatchEvent(new Event('input'));
      refreshPacking();
    }
    function refreshPacking() {
      const found = examples.indexOf(Number(panel.querySelector('#expert-routes').value));
      if (found >= 0) packingIndex = found;
      playbackControls.querySelector('#packing-count').textContent = found < 0 ? 'Custom load' : `Example ${found + 1} of ${examples.length}`;
      playbackControls.querySelector('#packing-prev').disabled = packingIndex === 0;
      playbackControls.querySelector('#packing-next').disabled = packingIndex === examples.length - 1;
    }
    panel.querySelector('#expert-routes').addEventListener('input', refreshPacking);
    panel.querySelectorAll('[data-routes]').forEach(b => b.addEventListener('click', refreshPacking));
    playbackControls.querySelector('#packing-prev').addEventListener('click', () => showPacking(Math.max(0, packingIndex - 1)));
    playbackControls.querySelector('#packing-next').addEventListener('click', () => showPacking(Math.min(examples.length - 1, packingIndex + 1)));
    refreshPacking();
    window.NotebookPlayback({root: packing, controls: playbackControls, watch: packing.querySelector('.route-comparison'),
      getIndex: () => packingIndex, count: () => examples.length, setIndex: showPacking});
    window.NotebookPlayback({root: partition, controls: partition.querySelector('.step-controls'), watch: partition.querySelector('#route-arrays'),
      getIndex: () => step, count: () => 3, setIndex: value => { step = value; renderPartition(); }});
  }

  mount('cache-writes', 'cache-explorer', `
    <div class="live-eyebrow">Two calls, the same cache</div><h3 id="cache-explorer-title">New locations on the next replay</h3>
    <p class="live-intro">Step through two illustrative calls. Watch slot 42 change twice, slot 17 retain its first write, and reserved slot 0 remain untouched.</p>
    <div class="cache-replay-input" id="cache-replay-input"></div>
    <div class="diagram-label">Selected slots in a persistent cache · other slots omitted</div><div class="cache-replay-slots" id="cache-replay-slots"></div>
    <div class="step-controls"><button type="button" id="cache-prev">Previous</button><span id="cache-count"></span><button type="button" id="cache-next">Next</button><button type="button" id="cache-reset">Reset</button></div>
    <p class="live-explanation" id="cache-detail" role="status"></p>
    <p class="live-footnote">Addresses and letters are illustrative; each letter stands for one K/V pair. This models changing inputs, not GPU execution or timing. Ordinary and fused writers must both skip padding and preserve unrelated slots.</p>
  `, panel => {
    let step = 0;
    const locations = [0, 17, 23, 42, 58, 73];
    const snapshots = [ ['reserved', 'old', 'old', 'old', 'old', 'old'], ['reserved', 'A', 'old', 'B', 'old', 'old'], ['reserved', 'A', 'C', 'D', 'old', 'old'] ];
    function render() {
      panel.querySelector('#cache-replay-input').textContent = ['Before either call: all existing slots are preserved.', 'Call 1: A → 17, B → 42, padding → 0 (skip).', 'Call 2: C → 23, D → 42, padding → 0 (skip).'][step];
      const slots = panel.querySelector('#cache-replay-slots'); slots.replaceChildren();
      locations.forEach((location, i) => {
        const changed = step > 0 && snapshots[step][i] !== snapshots[step - 1][i];
        const cell = document.createElement('div'); cell.className = 'cache-slot' + (changed ? ' just-written' : '') + (location === 0 ? ' reserved-slot' : '');
        cell.dataset.location = location; cell.dataset.value = snapshots[step][i];
        cell.innerHTML = `<small>Slot ${location}</small><strong>${snapshots[step][i]}</strong><span>${changed ? 'written now' : 'unchanged'}</span>`; slots.append(cell);
      });
      panel.querySelector('#cache-detail').textContent = ['The cache already contains data. A new call must not clear unrelated slots.', 'Only slots 17 and 42 changed. Slot 0 is not a valid destination for a live cache write.', 'The second call uses fresh locations. Slot 42 now holds D; slot 17 still holds A. Reusing the first call’s addresses would corrupt this state.'][step];
      panel.querySelector('#cache-count').textContent = `Step ${step + 1} of 3`;
      panel.querySelector('#cache-prev').disabled = step === 0; panel.querySelector('#cache-next').disabled = step === 2;
    }
    panel.querySelector('#cache-prev').addEventListener('click', () => { step = Math.max(0, step - 1); render(); });
    panel.querySelector('#cache-next').addEventListener('click', () => { step = Math.min(2, step + 1); render(); });
    panel.querySelector('#cache-reset').addEventListener('click', () => { step = 0; render(); });
    render();
    window.NotebookPlayback({root: panel, controls: panel.querySelector('.step-controls'), watch: panel.querySelector('#cache-replay-slots'),
      getIndex: () => step, count: () => 3, setIndex: value => { step = value; render(); }});
  });
})();
