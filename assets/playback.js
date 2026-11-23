/* Viewport-aware playback shared by the article's teaching diagrams. */
(() => {
  'use strict';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  window.NotebookPlayback = function ({root, controls, watch, getIndex, count, setIndex, interval = 3500}) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'playback-toggle';
    button.setAttribute('aria-label', 'Play diagram');
    const previous = controls.querySelector('button');
    previous ? previous.after(button) : controls.append(button);
    let requested = !reduced.matches, visible = false, timer = null, complete = false, advancing = false;
    const announcements = [...root.querySelectorAll('[aria-live], [role="status"]')];
    const liveValues = announcements.map(el => el.getAttribute('aria-live'));
    function clear() { if (timer !== null) clearTimeout(timer); timer = null; }
    function sync() {
      clear();
      const running = requested && visible && !document.hidden && !complete;
      root.dataset.playback = running ? 'playing' : complete ? 'complete' : requested ? 'offscreen' : 'paused';
      button.textContent = requested && !complete ? 'Pause' : 'Play';
      button.setAttribute('aria-label', requested && !complete ? 'Pause diagram' : 'Play diagram');
      announcements.forEach((el, i) => {
        if (running) el.setAttribute('aria-live', 'off');
        else if (liveValues[i] === null) el.removeAttribute('aria-live');
        else el.setAttribute('aria-live', liveValues[i]);
      });
      if (running) timer = setTimeout(() => {
        timer = null;
        if (getIndex() >= count() - 1) { complete = true; requested = false; sync(); return; }
        advancing = true;
        try { setIndex(getIndex() + 1); } finally { advancing = false; }
        sync();
      }, interval);
    }
    function pause() { requested = false; complete = false; sync(); }
    // Manual inspection wins over playback, including changing a variant or slider.
    root.addEventListener('click', event => {
      if (event.target.closest('button,input,select,a') && !button.contains(event.target)) pause();
    }, true);
    root.addEventListener('input', () => { if (!advancing) pause(); }, true);
    root.addEventListener('change', () => { if (!advancing) pause(); }, true);
    button.addEventListener('click', () => {
      if (requested && !complete) pause();
      else {
        if (getIndex() >= count() - 1) setIndex(0);
        complete = false; requested = true; sync();
      }
    });
    const observer = new IntersectionObserver(entries => {
      visible = entries[0].isIntersecting && entries[0].intersectionRatio >= 0.35;
      sync();
    }, {threshold: [0, 0.35]});
    observer.observe(watch || root);
    document.addEventListener('visibilitychange', sync);
    reduced.addEventListener('change', () => { if (reduced.matches) pause(); });
    sync();
    return {pause};
  };
})();
