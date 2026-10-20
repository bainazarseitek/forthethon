(() => {
  const data=window.ACTION_RESULTS,$=id=>document.getElementById(id),base='generated/action-pilot/';
  if(!data){$('load-error').hidden=false;return;}
  const keys=['videomae','clip','clip_text'],names={videomae:'VideoMAE + обучение',clip:'CLIP + обучение',clip_text:'CLIP + текст'},pct=x=>`${(100*x).toFixed(1)}%`,f=x=>x.toFixed(3);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const classes=data.classes,truth=w=>classes.findIndex(c=>c.id===w.label),test=data.summary.test;
  const norm=window.ACTION_NORMALIZATION;
  if(norm){
    $('normalization').hidden=false;
    const encoders=['videomae','clip'];
    $('normalization-metrics').innerHTML=encoders.map(k=>{const {before:b,after:a}=norm.models[k].variants;return `<article class="metric"><h2>${k==='videomae'?'VideoMAE':'CLIP'}</h2><strong>${f(b.validation.macro_f1)} → ${f(a.validation.macro_f1)}</strong><p>Macro F1 · только validation</p><p>Верных окон: ${Math.round(b.validation.accuracy*37)}/37 → ${Math.round(a.validation.accuracy*37)}/37</p><p>Выбранный C: ${b.C} → ${a.C}</p></article>`;}).join('');
    $('normalization-fixed').textContent='При неизменном C: '+encoders.map(k=>{const {before:b,after:a}=norm.models[k].variants;return `${k==='videomae'?'VideoMAE':'CLIP'} (C=${b.C}): ${f(b.validation.macro_f1)} → ${f(a.at_original_C.validation.macro_f1)}`;}).join('; ')+'.';
    $('normalization-classes').querySelector('tbody').innerHTML=norm.classes.map((c,i)=>`<tr><td>${esc(c.title)}</td><td>${norm.models.videomae.variants.before.validation.support[i]}</td>${encoders.map(k=>['before','after'].map(v=>`<td>${f(norm.models[k].variants[v].validation.f1[i])}</td>`).join('')).join('')}</tr>`).join('');
    $('normalization-grid').querySelector('tbody').innerHTML=encoders.map(k=>norm.C_grid.map((c,i)=>`<tr><td>${k==='videomae'?'VideoMAE':'CLIP'}</td><td>${c}</td><td>${f(norm.models[k].variants.before.grid[i].validation.macro_f1)}</td><td>${f(norm.models[k].variants.after.grid[i].validation.macro_f1)}</td></tr>`).join('')).join('');
    $('normalization-train').textContent=`После стандартизации F1 на train: VideoMAE ${f(norm.models.videomae.variants.after.train.macro_f1)}, CLIP ${f(norm.models.clip.variants.after.train.macro_f1)}. Разрыв с validation остаётся. Выходы голов не калиброваны; высокая уверенность не гарантирует правильность.`;
    $('normalization-audit').textContent=`Validation: ${norm.splits.validation.videos.join(', ')}. Исходные результаты всех C воспроизведены. Проверены статистики только по train и неизменность исходных файлов; изменение validation не меняет параметры стандартизации. Новые предсказания test отсутствуют. Расчёт на CPU по сохранённым векторам, sklearn ${norm.sklearn}.`;
  }
  $('metrics').innerHTML=keys.map(k=>{const m=data.models[k].metrics.test;return `<article class="metric"><h2>${names[k]}</h2><strong>${f(m.macro_f1)}</strong><p>Macro F1 · ${test.windows} окон / ${test.videos.length} видео</p><p>Верных ответов: ${pct(m.accuracy)}</p></article>`;}).join('');
  const delta=data.models.videomae.metrics.test.macro_f1-data.models.clip.metrics.test.macro_f1;
  $('conclusion').textContent=`VideoMAE ${delta>=0?'выше':'ниже'} обученного CLIP на ${Math.abs(delta).toFixed(3)} macro F1 в этом разделении. Для обеих моделей одинаковы кадры, метки, размер головы и процедура подбора. Это один небольшой эксперимент; устойчивость разницы на новых съёмках ещё не проверена.`;
  $('class-results').querySelector('tbody').innerHTML=classes.map((c,i)=>`<tr><td>${esc(c.title)}</td><td>${data.summary.train.classes[c.id]}</td><td>${data.models.videomae.metrics.test.support[i]}</td>${keys.map(k=>`<td>${f(data.models[k].metrics.test.f1[i])}</td>`).join('')}</tr>`).join('');
  $('video-results').querySelector('tbody').innerHTML=test.videos.map(v=>`<tr><td><button data-video="${v}">${v}</button></td><td>${data.models.videomae.test_per_video[v].windows}</td>${keys.map(k=>`<td>${pct(data.models[k].test_per_video[v].accuracy)}</td>`).join('')}</tr>`).join('');
  const splitNames={train:'Обучение',validation:'Выбор C',test:'Финальная проверка'};
  $('splits').querySelector('tbody').innerHTML=Object.entries(data.summary).map(([s,v])=>`<tr><td>${splitNames[s]}<br><small>${v.videos.length} видео</small></td><td class="split-videos">${v.videos.map(id=>`<span>${id}</span>`).join('')}</td><td>${v.windows}</td><td>${v.intervals}</td></tr>`).join('');
  $('selection').querySelector('tbody').innerHTML=keys.map(k=>{const m=data.models[k];return `<tr><td>${names[k]}</td><td>${m.C??'—'}</td>${['train','validation','test'].map(s=>`<td>${f(m.metrics[s].macro_f1)}</td>`).join('')}<td>${pct(m.test_video_mean_accuracy)}</td></tr>`;}).join('');
  $('run-info').textContent=`GPU: ${data.inference.gpu}\nДва замороженных энкодера; обучаемых параметров энкодера: 0\nДве головы: 768 × 6 весов + 6 смещений каждая\nОкно 2 с, 16 кадров по исходным PTS, шаг 1 с, максимум 8 окон из одного интервала\nСетка C: ${data.config.C_grid.join(', ')}\n\n`+['videomae','clip'].map(k=>{const m=data.inference.models[k];return `${names[k]}\n${m.id}@${m.revision}\nРасчёт ${data.windows.length} окон: ${m.forward_seconds.toFixed(1)} с; медиана окна: ${(1000*m.median_window_seconds).toFixed(0)} мс\nОбучение и выбор головы: ${data.models[k].fit_seconds.toFixed(2)} с\n`;}).join('\n')+`\nСкорость — GPU forward, без декодирования, сети и браузера. Онлайн-поток не тестировался.\nСнимок выборки SHA256: ${data.prepared_sha256}`;
  $('source').innerHTML=test.videos.map(v=>`<option>${v}</option>`).join('');
  $('source').value='X-26';
  let windows=[],selected=null,playingWindow=false;
  function render(w,seek=true){
    selected=w;playingWindow=false;
    const label=truth(w);
    $('window-title').textContent=`${w.video} · ${w.start.toFixed(2)}–${w.end.toFixed(2)} с · ${classes[label].title}`;
    $('source-note').textContent=`Ручная разметка ${w.annotation_start}–${w.annotation_end} с: ${w.seen}`;
    $('input').src=base+w.thumbnail;
    $('predictions').innerHTML=keys.map(k=>{const pred=data.models[k].predictions[w.index];return `<div class="prediction ${pred===label?'':'wrong'}">${names[k]}<strong>${esc(classes[pred].title)}</strong>${pred===label?'Совпало с разметкой':'Ошибка'}</div>`;}).join('');
    $('scores').innerHTML=classes.map((c,i)=>`<div class="action-score"><span>${esc(c.title)}</span>${keys.map(k=>{const s=data.models[k].scores[w.index][i],width=k==='clip_text'?Math.max(0,s)*200:s*100;return `<div><small>${s.toFixed(3)}</small><span class="bar" style="width:${Math.min(100,width)}%"></span></div>`;}).join('')}</div>`).join('');
    $('windows').querySelectorAll('button').forEach(b=>{const active=Number(b.dataset.index)===w.index;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
    if(seek){$('player').pause();$('player').currentTime=w.start;}
  }
  function choose(){
    $('player').pause();playingWindow=false;
    windows=data.windows.filter(w=>w.split==='test'&&w.video===$('source').value).sort((a,b)=>a.start-b.start);
    $('windows').innerHTML=windows.map(w=>`<button data-index="${w.index}" class="${data.models.videomae.predictions[w.index]===truth(w)?'right':'wrong'}" title="${esc(classes[truth(w)].title)} · ${w.start.toFixed(2)}–${w.end.toFixed(2)} с">${w.end.toFixed(1)} с</button>`).join('');
    $('player').src=data.sources[$('source').value].preview;
    render(windows[0],false);
    $('player').addEventListener('loadedmetadata',()=>{if(selected)$('player').currentTime=selected.start;},{once:true});
  }
  $('windows').addEventListener('click',e=>{const b=e.target.closest('button');if(b)render(data.windows[Number(b.dataset.index)]);});
  $('source').addEventListener('change',choose);
  $('video-results').addEventListener('click',e=>{const b=e.target.closest('button');if(b){$('source').value=b.dataset.video;choose();$('inspect-heading').scrollIntoView({behavior:'smooth'});}});
  $('play-window').addEventListener('click',async()=>{
    if(!selected)return;
    $('play-error').hidden=true;$('player').currentTime=selected.start;playingWindow=true;
    try{await $('player').play();}catch{playingWindow=false;$('play-error').hidden=false;}
  });
  $('player').addEventListener('timeupdate',()=>{
    const t=$('player').currentTime;$('clock').textContent=`${t.toFixed(2)} с`;
    if(playingWindow&&selected&&t>=selected.end){$('player').pause();playingWindow=false;}
  });
  choose();
})();
