/* The page displays saved local model outputs; manual labels are never predictions. */
(() => {
  const data = window.EMBEDDING_PILOT;
  if (!data) { document.querySelector('#load-error').hidden = false; return; }
  const $ = id => document.getElementById(id);
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const base = 'generated/embedding-pilot/';
  const labels = {normal:'Норма по разметке', violation:'Нарушение по разметке', uncertain:'Пограничное окно'};
  const range = s => `${s.start.toFixed(2)}–${s.end.toFixed(2)} с`;
  const option = (value, title) => `<option value="${escape(value)}">${escape(title)}</option>`;
  function player(segment, id, title) {
    return `<article><h3>${escape(title)} · ${escape(segment.video)}</h3><p class="small">Исходное время ${range(segment)} <span class="state-label ${segment.label}">${labels[segment.label]}</span></p>
      <video id="${id}" controls playsinline preload="metadata" src="${base + segment.clip}"></video>
      <details><summary class="small">Что получает модель</summary><img class="input-image" src="${base + segment.input_image}" alt="Центральный из 32 кадров после обрезки: ${escape(segment.video)}, ${range(segment)}"><p class="small muted">Один из 32 кадров. Центральная обрезка 224 × 224.</p></details></article>`;
  }
  function wirePlayback(prefix) {
    $(`${prefix}-play`).addEventListener('click', async () => {
      const players = [$(`${prefix}-a`), $(`${prefix}-b`)];
      for (const video of players) video.currentTime = 0;
      const results = await Promise.allSettled(players.map(video => video.play()));
      if (results.some(result => result.status === 'rejected')) {
        $(`${prefix}-distance`).textContent = 'Не удалось начать оба видео. Используйте кнопки плееров.';
      }
    });
    $(`${prefix}-pause`).addEventListener('click', () => [$(`${prefix}-a`), $(`${prefix}-b`)].forEach(video => video.pause()));
  }
  $('metrics').innerHTML = data.metrics.map(m => `<article class="metric"><p class="eyebrow">${escape(m.query)} → ${escape(m.reference)}</p><h2>${escape(m.title)}</h2><strong>AUC ${m.auc.toFixed(2)}</strong><p>${m.normal} норм. / ${m.violation} наруш. окон; ${m.uncertain} исключены</p><p class="muted">Тот же ролик, банк других пар: ${m.other_pair_auc.toFixed(2)}</p></article>`).join('');
  $('match-select').innerHTML = data.matches.map((m, i) => option(i, `${data.config.pairs.find(p => p.id === m.pair).query} / ${data.config.pairs.find(p => p.id === m.pair).reference} · ${m.title}`)).join('');
  function showMatch() {
    const match = data.matches[Number($('match-select').value)];
    $('match-note').textContent = match.note + (match.kind === 'normal_control' ? ' Контроль: оба фрагмента нормальные.' : ' Слева — состояние с нарушением, справа — норма.');
    $('matched-players').innerHTML = player(data.segments[match.a], 'match-a', 'Первый фрагмент') + player(data.segments[match.b], 'match-b', 'Чистая пара');
    $('match-distance').textContent = `Cosine similarity ${match.cosine.toFixed(3)} · расстояние ${match.distance.toFixed(3)}`;
  }
  $('match-select').addEventListener('change', showMatch);
  wirePlayback('match');
  $('query-pair').innerHTML = data.config.pairs.map(p => option(p.id, `${p.query} / ${p.reference} · ${p.title}`)).join('');
  let current;
  function showRetrieval() {
    const own = $('bank-select').value === 'own';
    const neighbor = data.segments[own ? current.nearest : current.other_nearest];
    const score = own ? current.score : current.other_score;
    $('retrieval-players').innerHTML = player(current, 'retrieval-a', 'Проверяемое окно') + player(neighbor, 'retrieval-b', 'Ближайший эталон');
    $('retrieval-distance').textContent = `Расстояние до нормы ${score.toFixed(3)} · cosine ${(1-score).toFixed(3)}`;
    $('retrieval-note').textContent = 'Выбран максимум сходства среди окон банка. Это поиск похожего фрагмента, без выравнивания фаз и без знания того, что происходило раньше.';
    document.querySelectorAll('[data-window]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.window) === current.index)));
  }
  function showPair() {
    const pair = data.config.pairs.find(p => p.id === $('query-pair').value);
    const windows = data.segments.filter(s => s.purpose === 'grid' && s.video === pair.query);
    $('query-windows').innerHTML = windows.map(s => `<button class="${s.label}" type="button" data-window="${s.index}" aria-label="${escape(s.video)}, ${range(s)}, ${labels[s.label]}" aria-pressed="false">${s.start.toFixed(0)} с</button>`).join('');
    current = windows.find(s => s.label === 'violation') || windows[0];
    showRetrieval();
  }
  $('query-windows').addEventListener('click', event => {
    const button = event.target.closest('[data-window]');
    if (button) { current = data.segments[Number(button.dataset.window)]; showRetrieval(); }
  });
  $('query-pair').addEventListener('change', showPair);
  $('bank-select').addEventListener('change', showRetrieval);
  wirePlayback('retrieval');
  $('run-info').textContent = JSON.stringify({model:data.run.model, device:data.run.device, windows:data.segments.length,
    dimension:data.run.dimension, inference_seconds:data.run.inference_seconds, total_seconds:data.run.total_seconds,
    checkpoint_sha256:data.run.checkpoint_sha256, pca_variance:data.pca_variance}, null, 2);
  $('match-select').value = '1';
  showMatch(); showPair();
})();
