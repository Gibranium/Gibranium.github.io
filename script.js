// davidegualano.com — hero dots, scrollytelling for the Second opinion story, the live price lookup and the brief form.
(function () {
  const SVG = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent, text) => {
    const n = document.createElementNS(SVG, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };
  // "Reduce motion" (a system setting) makes the page stand still on purpose. ?motion=1 in the address overrides it, to check.
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches && !/[?&]motion=1/.test(location.search);

  // movement on scroll: the elements already on the page rise into place once, one after the other, when their
  // section comes into view. Only with IntersectionObserver and without "reduce motion"; otherwise nothing is hidden.
  if (!still && 'IntersectionObserver' in window) {
    const groups = [
      ['.story-head > *, .story > h2', ''], ['.lookup-sec .price-text > *', ''], ['.lookup-sec .price-card', 'rv-scale'],
      ['.services > h2, .services > .intro, .services .offer, .services > *:not(h2):not(.intro):not(.offer)', ''], ['.proof > h2, .proof article', ''],
      ['.how > h2', ''], ['.how-list dt, .how-list dd', 'rv-left'], ['.how > *:not(h2):not(.how-list)', ''], ['.method-band > *', ''], ['.media-text > *', ''], ['.reel', 'rv-scale'],
      ['.cutback > *, .cutback li', ''], ['.about > *', ''], ['.book > *, .book form > *', '']];
    const io = new IntersectionObserver(es => {
      es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    const seen = new Set();
    groups.forEach(([sel, extra]) => {
      let i = 0, parent = null;
      document.querySelectorAll(sel).forEach(n => {
        if (seen.has(n) || n.closest('.hero')) return;
        for (let p = n.parentElement; p; p = p.parentElement) if (seen.has(p)) return;     // a parent already moves
        seen.add(n);
        if (n.parentElement !== parent) { parent = n.parentElement; i = 0; }                // the sequence restarts per block
        n.classList.add('rv'); if (extra) n.classList.add(extra);
        n.style.setProperty('--i', Math.min(i++, 6));
        io.observe(n);
      });
    });
  }

  // the reel of graphics (For editors and brands): it turns by itself; a sideways swipe on the trackpad (or a drag,
  // or the arrow keys) pushes it, stops it or sends it back, then it drifts to its own pace again.
  document.querySelectorAll('.reel').forEach(reel => {
    const view = reel.querySelector('.reel-view'), track = reel.querySelector('.reel-track');
    if (!view || !track || still) return;                       // reduced motion: the plain scrollable row stays
    const items = Array.from(track.children);
    if (items.length < 2) return;
    items.forEach(li => { const c = li.cloneNode(true); c.setAttribute('aria-hidden', 'true'); track.appendChild(c); });   // endless loop
    reel.classList.add('is-live');
    const BASE = 36;                                            // px per second, its own pace
    let x = 0, v = BASE, target = BASE, half = 0, last = 0, on = false, paused = false, hover = false, drag = null, idle = 0;
    const measure = () => { half = track.scrollWidth / 2 + parseFloat(getComputedStyle(track).columnGap || 0) / 2; };
    const frame = t => {
      if (!on) return;
      const dt = Math.min(0.05, (t - last) / 1000 || 0); last = t;
      if (!drag) {
        const want = (paused || hover) && performance.now() > idle ? 0 : target;
        v += (want - v) * Math.min(1, dt * 3);                 // ease to the pace it should have
        if (performance.now() > idle) target += ((paused ? 0 : BASE) - target) * Math.min(1, dt * 0.9);   // a push fades back to its own pace
        x += v * dt;
      }
      if (half) x = ((x % half) + half) % half;
      track.style.transform = 'translate3d(' + (-x).toFixed(1) + 'px,0,0)';
      requestAnimationFrame(frame);
    };
    const start = () => { if (!on) { on = true; last = performance.now(); measure(); requestAnimationFrame(frame); } };
    const push = d => { target = Math.max(-700, Math.min(700, target + d)); v += d * 0.5; idle = performance.now() + 900; };
    view.addEventListener('wheel', e => {
      if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;     // vertical scroll goes to the page
      e.preventDefault();
      x += e.deltaX; push(e.deltaX * 1.2);                   // the swipe moves it directly and leaves it some speed
    }, { passive: false });
    view.addEventListener('pointerdown', e => { drag = { x0: e.clientX, x, t: performance.now(), lx: e.clientX, v: 0 }; view.classList.add('is-drag'); view.setPointerCapture(e.pointerId); });
    view.addEventListener('pointermove', e => {
      if (!drag) return;
      const now = performance.now(); drag.v = (drag.lx - e.clientX) / Math.max(1, now - drag.t) * 1000; drag.lx = e.clientX; drag.t = now;
      x = drag.x + (drag.x0 - e.clientX);
    });
    const drop = () => { if (!drag) return; v = drag.v; target = Math.max(-900, Math.min(900, drag.v)); idle = performance.now() + 700; drag = null; view.classList.remove('is-drag'); };
    view.addEventListener('pointerup', drop); view.addEventListener('pointercancel', drop);
    view.addEventListener('mouseenter', () => { hover = true; }); view.addEventListener('mouseleave', () => { hover = false; });
    view.addEventListener('keydown', e => { if (e.key === 'ArrowRight') { push(500); e.preventDefault(); } else if (e.key === 'ArrowLeft') { push(-500); e.preventDefault(); } });
    reel.querySelectorAll('[data-reel]').forEach(b => b.addEventListener('click', () => {
      const k = b.dataset.reel;
      if (k === 'toggle') { paused = !paused; b.textContent = paused ? 'Play' : 'Pause'; b.setAttribute('aria-label', paused ? 'Play the carousel' : 'Pause the carousel'); if (!paused) target = BASE; }
      else push(k === 'next' ? 700 : -700);
    }));
    window.addEventListener('resize', measure);
    track.querySelectorAll('img').forEach(img => img.addEventListener('load', measure));
    if ('IntersectionObserver' in window) new IntersectionObserver(es => { if (es[0].isIntersecting) start(); else on = false; }, { rootMargin: '200px 0px' }).observe(reel);
    else start();
  });

  // hero: 13,908 transfers, one dot each. They start as one block (every signing looks the same on the day)
  // and sort into the three outcomes. Shares (run of 12 Nov 2026): 43.4 met / 33.6 not selected or too few minutes / the rest played and fell short.
  const canvas = document.querySelector('.dots');
  if (canvas && canvas.getContext) {
    const N = 13908;
    const met = Math.round(N * 0.434), out = Math.round(N * 0.336);
    const groups = [met, out, N - met - out];
    const colours = ['#2A52B0', '#A8231C', '#8A8380'];
    const NEUTRAL = '#3A3533', GAP = 10, DURATION = 2200;
    const ctx = canvas.getContext('2d');
    const split = document.querySelector('.hero .split');
    let dots = [], cell = 4, cols = 1, t = still ? 1 : 0, started = false, t0 = 0;

    function layout() {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!w || !h) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const room = w - 2 * GAP;
      cell = Math.sqrt(room * h / N);
      let rows, gcols;
      for (;;) {                                   // shrink the cell until every group's columns fit the width
        rows = Math.max(1, Math.floor(h / cell));
        gcols = groups.map(n => Math.ceil(n / rows));
        if (gcols.reduce((a, b) => a + b, 0) * cell <= room) break;
        cell *= 0.995;
      }
      cols = gcols.reduce((a, b) => a + b, 0);
      dots = [];
      let col0 = 0;
      groups.forEach((n, g) => {
        for (let i = 0; i < n; i++) {
          const c = col0 + Math.floor(i / rows), r = i % rows;
          dots.push({ x: c * cell + cell / 2, y: r * cell + cell / 2, g, d: 0.5 * (c / cols) + 0.12 * Math.random() });
        }
        col0 += gcols[g];
      });
      draw();
    }

    function draw() {
      const w = canvas.clientWidth, h = canvas.clientHeight, r = cell * 0.36;
      const ease = t < 1 ? 1 - Math.pow(1 - t, 3) : 1;
      ctx.clearRect(0, 0, w, h);
      const paths = [new Path2D(), new Path2D(), new Path2D(), new Path2D()];   // neutral, then the three outcomes
      for (const p of dots) {
        const x = p.x + p.g * GAP * ease;
        const path = paths[t >= p.d + 0.25 ? p.g + 1 : 0];
        path.moveTo(x + r, p.y);
        path.arc(x, p.y, r, 0, 6.2832);
      }
      ctx.fillStyle = NEUTRAL; ctx.fill(paths[0]);
      colours.forEach((c, i) => { ctx.fillStyle = c; ctx.fill(paths[i + 1]); });
    }

    function frame(now) {
      if (!t0) t0 = now;
      t = Math.min(1, (now - t0) / DURATION);
      draw();
      if (t > 0.55 && split) split.classList.add('is-on');
      if (t < 1) requestAnimationFrame(frame);
    }

    layout();
    if (still) { if (split) split.classList.add('is-on'); }
    else if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver(es => {
        if (es[0].isIntersecting && !started) { started = true; io.disconnect(); setTimeout(() => requestAnimationFrame(frame), 500); }
      }, { threshold: 0.35 });
      io.observe(canvas);
    } else { t = 1; draw(); if (split) split.classList.add('is-on'); }
    let rz;
    window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(layout, 120); });
  } else {
    const split = document.querySelector('.hero .split');
    if (split) split.classList.add('is-on');
  }

  // frame 3: 100 squares, Championship strikers' first season after the move (strikers leaving it, 241 moves)
  const waffle = document.querySelector('.waffle');
  if (waffle) {
    const parts = [['met', 47], ['out', 33], ['low', 20]];
    let i = 0;
    for (const [cls, n] of parts) {
      for (let j = 0; j < n; j++, i++) {
        el('rect', { x: (i % 10) * 26, y: Math.floor(i / 10) * 26, width: 22, height: 22, rx: 3, class: 'sq sq-' + cls }, waffle);
      }
    }
  }

  // frame 5: fair price and worth on a log scale, €2m to €50m
  const range = document.querySelector('.range');
  if (range) {
    const lo = Math.log(2), hi = Math.log(50);
    const x = v => 40 + 320 * (Math.log(v) - lo) / (hi - lo);
    // an illustration of the shape, not a player: centre, 80% range, and where a market value might sit
    el('text', { x: 40, y: 0, class: 'lbl' }, range, 'Fair price and its 80% range');
    el('line', { x1: x(4), x2: x(30), y1: 20, y2: 20, class: 'bar bar-price' }, range);
    el('circle', { cx: x(11), cy: 20, r: 9, class: 'dot' }, range);
    el('text', { x: 40, y: 64, class: 'lbl muted' }, range, 'Transfermarkt');
    el('circle', { cx: x(20), cy: 84, r: 7, class: 'dot-worth' }, range);
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

  // price lookup: reads the price app's own published file, so a data refresh pushed to the app shows up here too.
  // Columns used: player, team, league, role, age, minutes, season, worth(_low/_high), price(_low/_high), in € million.
  const DATA_URL = 'https://raw.githubusercontent.com/Gibranium/what-would-he-cost/main/data/players.csv';
  const SEASON = '2025/26';
  const section = document.getElementById('price');
  if (section) {
    const $ = id => document.getElementById(id);
    const q = $('lk-q'), list = $('lk-list'), meta = $('lk-meta'), chart = $('lk-chart'), brief = $('lk-brief');
    let rows = null, loading = false, shown = [], active = -1, current = null;

    const eur = x => x >= 100 ? '€' + Math.round(x).toLocaleString('en') + 'm' : x >= 10 ? '€' + x.toFixed(0) + 'm'
      : x >= 1 ? '€' + x.toFixed(1) + 'm' : '€' + Math.round(x * 1000).toLocaleString('en') + 'k';
    const norm = s => s.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/\u0131/g, 'i').replace(/\u00f8/g, 'o').replace(/\u0142/g, 'l').replace(/\u0111/g, 'd').replace(/\u00e6/g, 'ae').replace(/\u00df/g, 'ss');

    function parseCSV(text) {
      const out = []; let row = [], cur = '', inq = false;
      for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (inq) {
          if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else inq = false; }
          else cur += c;
        } else if (c === '"') inq = true;
        else if (c === ',') { row.push(cur); cur = ''; }
        else if (c === '\n') { row.push(cur); out.push(row); row = []; cur = ''; }
        else if (c !== '\r') cur += c;
      }
      if (cur || row.length) { row.push(cur); out.push(row); }
      return out;
    }

    function fail() {
      meta.innerHTML = 'The prices didn\'t load. <a href="https://what-would-he-cost.streamlit.app">Open the price app</a> instead.';
      $('lk-card').hidden = true;
    }

    function load() {
      if (rows || loading) return;
      loading = true;
      fetch(DATA_URL).then(r => { if (!r.ok) throw new Error(r.status); return r.text(); }).then(text => {
        const t = parseCSV(text), head = t[0], ix = {};
        head.forEach((h, i) => ix[h] = i);
        const need = ['player', 'team', 'league', 'role', 'age', 'minutes', 'season', 'worth', 'worth_low', 'worth_high', 'price', 'price_low', 'price_high'];
        if (need.some(k => !(k in ix))) throw new Error('columns');
        const num = ['age', 'minutes', 'worth', 'worth_low', 'worth_high', 'price', 'price_low', 'price_high'];
        rows = [];
        for (let i = 1; i < t.length; i++) {
          const a = t[i];
          if (a.length < head.length || a[ix.season] !== SEASON) continue;
          const r = {};
          for (const k of need) r[k] = a[ix[k]];
          for (const k of num) r[k] = +r[k];
          for (const k of ['price_pl', 'price_other', 'pl_chance']) r[k] = k in ix ? +a[ix[k]] : NaN;   // the two prices behind the fair one (from 12 Nov 2026)
          if (!(r.price > 0) || !(r.price_low > 0) || !(r.worth_low > 0)) continue;
          r.key = norm(r.player + ' ' + r.team);
          rows.push(r);
        }
        if (!rows.length) throw new Error('empty');
        rows.sort((a, b) => b.price - a.price);
        const leagues = new Set(rows.map(r => r.league)).size;
        meta.textContent = rows.length.toLocaleString('en') + ' players in ' + leagues + ' leagues, ' + SEASON + ' season, priced at the summer 2026 market. The card shows the dearest player in the model until you pick one.';
        show(rows.find(r => r.minutes >= 900) || rows[0]);
        if (q.value.trim()) suggest();
      }).catch(fail);
    }

    function show(r) {
      current = r;
      $('lk-who').textContent = [r.player, r.team, r.league, r.role, Math.round(r.age), r.minutes.toLocaleString('en') + ' min (' + r.season + ')'].join(' · ');
      $('lk-big').textContent = eur(r.price);
      $('lk-rng').textContent = '80% range: ' + eur(r.price_low) + ' – ' + eur(r.price_high);
      const two = $('lk-two');
      if (two) {
        const ok = r.price_pl > 0 && r.price_other > 0 && Math.abs(r.price_pl / r.price_other - 1) > 0.005;
        two.hidden = !ok;
        if (ok) two.textContent = 'If a Premier League club buys: ' + eur(r.price_pl) + ' · anyone else: ' + eur(r.price_other)
          + ' · players like him go to a Premier League club ' + Math.round(r.pl_chance * 100) + '% of the time';
      }
      brief.textContent = 'Get the full read on ' + r.player + ', €200';
      while (chart.firstChild) chart.removeChild(chart.firstChild);
      const W = Math.max(280, Math.round(chart.clientWidth || 520)), L = 88, R = W - 14;
      chart.setAttribute('viewBox', '0 0 ' + W + ' 150');
      const lo = Math.max(0.01, Math.min(r.price_low, r.worth_low) * 0.8), hi = Math.max(r.price_high, r.worth_high) * 1.2;
      const x = v => L + (R - L) * (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
      [['Fair price', 'r-price', r.price_low, r.price, r.price_high, 34], ['Worth', 'r-worth', r.worth_low, r.worth, r.worth_high, 78]].forEach(([name, cls, a, m, b, y]) => {
        el('text', { x: 0, y: y + 5, class: 't' }, chart, name);
        el('line', { x1: x(a), x2: x(b), y1: y, y2: y, class: cls }, chart);
        el('circle', { cx: x(m), cy: y, r: 8, class: 'c' }, chart);
      });
      el('line', { x1: L, x2: R, y1: 112, y2: 112, class: 'ax' }, chart);
      let last = -99;
      for (const v of [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000]) {
        if (v < lo || v > hi || x(v) - last < 52) continue;
        last = x(v);
        el('line', { x1: x(v), x2: x(v), y1: 112, y2: 117, class: 'ax' }, chart);
        el('text', { x: x(v), y: 134, class: 'tk', 'text-anchor': 'middle' }, chart, v >= 1 ? '€' + v + 'm' : '€' + v * 1000 + 'k');
      }
    }

    function close() { list.hidden = true; q.setAttribute('aria-expanded', 'false'); active = -1; }

    function suggest() {
      const words = norm(q.value).split(/\s+/).filter(Boolean);
      list.textContent = '';
      if (!rows || !words.length) { close(); return; }
      shown = [];
      for (const r of rows) {
        if (words.every(w => r.key.includes(w))) { shown.push(r); if (shown.length === 8) break; }
      }
      if (!shown.length) {
        const li = document.createElement('li'); li.textContent = 'No player or team matches. Check the spelling.'; li.setAttribute('aria-disabled', 'true'); list.appendChild(li);
      }
      shown.forEach((r, i) => {
        const li = document.createElement('li'), s = document.createElement('span');
        li.setAttribute('role', 'option'); li.id = 'lk-o' + i; li.dataset.i = i;
        li.textContent = r.player; s.textContent = r.team + ' · ' + eur(r.price); li.appendChild(s);
        list.appendChild(li);
      });
      active = -1; list.hidden = false; q.setAttribute('aria-expanded', 'true');
    }

    function pick(i) {
      const r = shown[i]; if (!r) return;
      show(r); q.value = r.player; close();
    }
    function mark(i) {
      const items = list.querySelectorAll('[role=option]'); if (!items.length) return;
      active = (i + items.length) % items.length;
      items.forEach((n, k) => n.setAttribute('aria-selected', k === active ? 'true' : 'false'));
      q.setAttribute('aria-activedescendant', 'lk-o' + active);
      items[active].scrollIntoView({ block: 'nearest' });
    }

    q.addEventListener('focus', load);
    q.addEventListener('input', () => { load(); suggest(); });
    q.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown') { e.preventDefault(); if (list.hidden) suggest(); else mark(active + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); mark(active - 1); }
      else if (e.key === 'Enter') { e.preventDefault(); pick(active >= 0 ? active : 0); }
      else if (e.key === 'Escape') close();
    });
    list.addEventListener('mousedown', e => { const li = e.target.closest('[role=option]'); if (li) { e.preventDefault(); pick(+li.dataset.i); } });
    q.addEventListener('blur', () => setTimeout(close, 150));

    // the card's button carries the player into the brief form
    brief.addEventListener('click', () => {
      const form = document.getElementById('brief'); if (!form || !current) return;
      form.elements.need.value = 'Second opinion';
      form.elements.target.value = current.player + ' (' + current.team + ', ' + current.league + ')';
    });

    let lkrz;
    window.addEventListener('resize', () => { clearTimeout(lkrz); lkrz = setTimeout(() => { if (current) show(current); }, 150); });

    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); load(); } }, { rootMargin: '900px 0px' });
      io.observe(section);
    } else load();
  }

  // the brief. With a form endpoint in data-endpoint (Formspree or similar) it is sent from the page;
  // without one, or if sending fails, it opens the visitor's email app with the answers filled in.
  const form = document.getElementById('brief');
  if (form) {
    const endpoint = (form.dataset.endpoint || '').trim();
    const how = document.getElementById('brief-how');
    if (endpoint && how) how.textContent = 'Your brief comes straight to my inbox.';
    const msg = form.querySelector('.form-msg');

    const mail = f => {
      const body = [
        'Name and role: ' + f.get('name'),
        'Club / agency / outlet: ' + f.get('org'),
        'Email: ' + f.get('email'),
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
    };

    form.addEventListener('submit', ev => {
      ev.preventDefault();
      if (!form.checkValidity()) {
        const first = form.querySelector(':invalid');
        msg.textContent = 'Fill in the fields marked as missing, then send again.';
        form.classList.add('tried');
        if (first) first.focus();
        return;
      }
      const f = new FormData(form);
      if (!endpoint) { mail(f); return; }
      f.append('_subject', 'Brief: ' + f.get('need') + ' (' + f.get('org') + ')');
      const button = form.querySelector('button[type=submit]');
      button.disabled = true; msg.textContent = 'Sending…';
      fetch(endpoint, { method: 'POST', body: f, headers: { Accept: 'application/json' } })
        .then(r => { if (!r.ok) throw new Error(r.status); msg.textContent = 'Sent. I\'ll reply personally, usually with a couple of questions before we talk.'; form.reset(); form.classList.remove('tried'); })
        .catch(() => mail(f))
        .then(() => { button.disabled = false; });
    });
  }
})();
