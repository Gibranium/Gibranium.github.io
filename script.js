// davidegualano.com — scrollytelling for the Second opinion story, the two data graphics, and the brief form.
(function () {
  const SVG = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent, text) => {
    const n = document.createElementNS(SVG, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };

  // frame 3: 100 squares, Championship strikers' first season after the move (240 moves)
  const waffle = document.querySelector('.waffle');
  if (waffle) {
    const parts = [['met', 45], ['out', 35], ['low', 20]];
    let i = 0;
    for (const [cls, n] of parts) {
      for (let j = 0; j < n; j++, i++) {
        el('rect', { x: (i % 10) * 26, y: Math.floor(i / 10) * 26, width: 22, height: 22, rx: 3, class: 'sq sq-' + cls }, waffle);
      }
    }
  }

  // frame 4: fair price and worth on a log scale, €2m to €50m
  const range = document.querySelector('.range');
  if (range) {
    const lo = Math.log(2), hi = Math.log(50);
    const x = v => 40 + 320 * (Math.log(v) - lo) / (hi - lo);
    // an illustration of the shape, not a player: centre, 80% range, and where a market value might sit
    el('text', { x: 40, y: 0, class: 'lbl' }, range, 'Fair price and its 80% range');
    el('line', { x1: x(4), x2: x(30), y1: 20, y2: 20, class: 'bar bar-price' }, range);
    el('circle', { cx: x(11), cy: 20, r: 9, class: 'dot' }, range);
    el('text', { x: 40, y: 64, class: 'lbl muted' }, range, 'Transfermarkt');
    el('circle', { cx: x(20), cy: 84, r: 7, class: 'dot dot-worth' }, range);
    el('line', { x1: 40, x2: 360, y1: 116, y2: 116, class: 'axis' }, range);
    el('text', { x: 40, y: 140, class: 'tick' }, range, 'cheaper');
    el('text', { x: 360, y: 140, class: 'tick', 'text-anchor': 'end' }, range, 'dearer');
  }

  // scrollytelling: the step in the middle of the viewport sets the visible frame
  const frames = document.querySelector('.frames');
  const steps = document.querySelectorAll('.step');
  if (frames && steps.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      for (const e of entries) {
        if (e.isIntersecting) {
          frames.dataset.active = e.target.dataset.step;
          steps.forEach(s => s.classList.toggle('is-active', s === e.target));
        }
      }
    }, { rootMargin: '-45% 0px -45% 0px' });
    steps.forEach(s => io.observe(s));
  } else if (frames) {
    frames.classList.add('no-scrolly');
  }

  // the brief: builds an email to Davide with the answers filled in
  const form = document.getElementById('brief');
  if (form) {
    form.addEventListener('submit', ev => {
      ev.preventDefault();
      const msg = form.querySelector('.form-msg');
      if (!form.checkValidity()) {
        const first = form.querySelector(':invalid');
        msg.textContent = 'Fill in the fields marked as missing, then send again.';
        form.classList.add('tried');
        if (first) first.focus();
        return;
      }
      const f = new FormData(form);
      const body = [
        'Name and role: ' + f.get('name'),
        'Club / agency / outlet: ' + f.get('org'),
        'What I need: ' + f.get('need'),
        'Position or player, and league: ' + f.get('target'),
        'When: ' + f.get('when'),
        'Data or video access: ' + f.get('access'),
        '',
        'Anything else: ' + (f.get('notes') || '-'),
      ].join('\n');
      const subject = 'Brief: ' + f.get('need') + ' (' + f.get('org') + ')';
      window.location.href = 'mailto:davide@davidegualano.com?subject=' + encodeURIComponent(subject) + '&body=' + encodeURIComponent(body);
      msg.textContent = 'Your email app should be open with the brief filled in. If it isn\'t, write to davide@davidegualano.com.';
    });
  }
})();
