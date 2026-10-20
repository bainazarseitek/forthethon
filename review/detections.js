(() => {
  const all=window.DETECTOR_PILOTS,$=id=>document.getElementById(id);
  if(!all?.wide||!all?.patch){$('load-error').hidden=false;return;}
  const names={count:'Пересчёт купюр',put_table:'Положить на стол',carry:'Нести деньги',store:'Положить в сейф',close:'Закрыть сейф',leave:'Уйти от стола',idle:'Пустое рабочее место',cash:'Деньги обнаружены',clear:'Предполагается пустота',occluded:'Закрыто руками',unknown:'Неясно',attending:'Кассир в рабочей зоне',away:'Кассир вне рабочей зоны'};
  const stages={cash_on_table:'Сохранено свидетельство денег',closing_seen:'Наблюдалось предполагаемое закрытие',storage_seen:'Наблюдалось предполагаемое помещение в сейф',counting_seen:'Наблюдался предполагаемый пересчёт',unconfirmed:'Этап не подтверждён'};
  const first=x=>x===null?'нет':`${x.toFixed(1)} с`;
  const comparisons=[['Прежний CLIP',all.wide.metrics.clip_without_memory],['Детекторы: весь стол',all.wide.metrics.without_memory],['Весь стол + память',all.wide.metrics.with_memory],['Детекторы: крупный участок',all.patch.metrics.without_memory],['Крупный участок + память',all.patch.metrics.with_memory]];
  document.querySelector('#comparison tbody').innerHTML=comparisons.map(([label,m])=>`<tr><td>${label}</td><td>${m.auc.toFixed(3)}</td><td>${m.true_positive}</td><td>${m.false_negative}</td><td>${m.false_positive}</td><td>${first(m.first_alarm['X-26'])}</td></tr>`).join('');
  let data,base,rows=[],index=-1;
  function variant(){
    const view=$('variant').value;data=all[view];base=`generated/detector-pilot${view==='patch'?'-patch':''}/`;
    const a=data.audits.presence_clear,c=data.audits.cash;
    $('summary').innerHTML=[['Присутствие кассира',`${a.detector_binary_correct}/${a.total}`,`Прежний CLIP: ${a.clip_binary_correct}/${a.total} на тех же явных моментах`],['Деньги на поздних кадрах',`${c.detector_binary_correct}/${c.total}`,`Прежний CLIP: ${c.clip_binary_correct}/${c.total}; бинарная проверка наличия`],['Память изменила тревог',data.changed_alarms,`Численная оценка изменилась в ${data.changed_scores} моментах`]].map(([t,v,n])=>`<article class="metric"><h2>${t}</h2><strong>${v}</strong><p>${n}</p></article>`).join('');
    const m=data.metrics.with_memory,n=data.metrics.without_memory;
    $('conclusion').textContent=`Без памяти найдено ${n.true_positive}/12 моментов нарушения, с памятью — ${m.true_positive}/12. Ложных тревог: ${n.false_positive} и ${m.false_positive}. ${data.memory_active_points===0?'Память денег ни разу не активировалась: двух уверенных подтверждений подряд нет. Этот прогон не проверил её пользу при надёжном распознавании денег.':data.changed_alarms?'Изменение нужно оценивать вместе с пропусками и ложными тревогами.':'Преимущество памяти в этом прогоне не показано.'} Первая тревога X-27: ${first(m.first_alarm['X-27'])}.`;
    $('plot').src=base+'timelines.png';$('plot-link').href=base+'timelines.png';
    $('json-link').href=base+'results.json';$('csv-link').href=base+'scores.csv';$('raw-link').href=base+'detections.json';
    $('audit').textContent=`Деньги: бинарно ${c.detector_binary_correct}/${c.total}; точное состояние подтверждено на ${c.detector_state_correct}/${c.total}, неопределённых — ${c.detector_unknown}. Присутствие: ${data.audits.presence.detector_binary_correct}/${data.audits.presence.total} по всей прежней разметке, ${a.detector_binary_correct}/${a.total} на явных моментах. Это проверка состояния, не оценка точности локализации рамок (mAP).`;
    const run=data.run,l=run.per_update_seconds.slice(1).sort((a,b)=>a-b);
    $('run-info').textContent=`Детектор: ${run.model}\nРевизия: ${run.revision}\nУстройство: ${run.gpu}, ${run.dtype}\nОбласть денег: ${data.config.table_crop.join(', ')}\nЗагрузка: ${run.model_load_seconds.toFixed(2)} с; 47 обновлений: ${run.detector_seconds.toFixed(2)} с\nМедиана после первого обновления: ${(l[Math.floor(l.length/2)]*1000).toFixed(0)} мс (две области, подготовка и обработка рамок включены)\nCLIP: прежние сохранённые признаки, без повторного расчёта.\nПередача файлов, декодирование видео и браузер не включены; живой видеопоток не тестировался.`;
    const savedTime=$('player').currentTime;rows=data.rows.filter(r=>r.video===$('source').value);
    if(index>=0)render(Math.max(0,rows.findLastIndex(r=>r.t<=savedTime+1e-6)));
  }
  function render(i,seek=false){
    index=i;
    if(i<0){$('evidence').hidden=true;$('timestamp').textContent='Ждём 1 с';$('boxes-image').hidden=true;$('manual').textContent='Начало записи';return;}
    const r=rows[i];$('evidence').hidden=false;$('boxes-image').hidden=false;$('time').value=i;
    if(seek)$('player').currentTime=r.t;
    $('timestamp').textContent=`${r.t.toFixed(2)} с`;
    $('boxes-image').src=base+r.annotated_image;$('table-image').src=base+r.table_image;
    $('manual').textContent=`Ручная разметка: ${r.label==='violation'?'нарушение':'норма'} · ${r.manual_note}`;
    $('table-state').textContent=`Стол: ${names[r.table_observation]}. Оценка рамки денег: ${r.cash_raw.toFixed(3)}.`;
    $('presence-state').textContent=`${names[r.presence_observation]}. Оценка человека в зоне: ${r.person_raw.toFixed(3)}.`;
    $('risk').innerHTML=[['without_memory','Без памяти'],['with_memory','С памятью']].map(([key,label])=>`<div class="${r[key]>=.5?'alarm':''}">${label}<strong>${r[key].toFixed(3)}</strong>${r[key]>=.5?'Тревога':'Ниже порога'}</div>`).join('');
    $('reason').textContent=`Кадр: ${r.pts.toFixed(3)} с. ${r.memory_evidence_t===null?'Сохранённого подтверждения денег нет.':`Последнее подтверждение денег: ${r.memory_evidence_t.toFixed(2)} с.`} ${r.memory_used?'Использована память о деньгах.':''}`;
    $('memory-state').textContent=stages[r.stage];
    const best=Object.keys(r.actions).reduce((a,b)=>r.actions[a]>r.actions[b]?a:b);
    $('action-scores').innerHTML=Object.entries(r.actions).map(([k,v])=>`<div class="score ${k===best?'best':''}"><span>${names[k]}</span><div class="score-track"><span style="width:${Math.max(0,Math.min(100,v*200))}%"></span></div><span>${v.toFixed(3)}</span></div>`).join('');
    $('history').textContent=Object.entries(r.seen).map(([k,t])=>`${names[k]} (${t.toFixed(1)} с)`).join(' → ')||'Нет подтверждённых действий';
    $('expected').textContent=`Возможные следующие действия: ${r.expected.map(k=>names[k].toLowerCase()).join(', ')}. Все действия по-прежнему оцениваются.`;
    $('box-list').textContent=r.boxes.map(b=>`${b.kind} ${b.score.toFixed(3)} [${b.box.map(x=>x.toFixed(1)).join(', ')}]${b.kind==='cash'?` · на столешнице: ${b.on_surface}; перекрытие рукой: ${b.hand_overlap.toFixed(2)}`:b.kind==='person'?` · в зоне: ${b.in_zone}`:''}`).join('\n')||'Кандидатов с оценкой ≥ 0.1 нет';
    $('prev').disabled=i===0;$('next').disabled=i===rows.length-1;
  }
  function choose(){
    rows=data.rows.filter(r=>r.video===$('source').value);$('time').max=rows.length-1;index=-1;
    $('player').src=data.sources[$('source').value].preview;
    $('player').addEventListener('loadedmetadata',()=>render(0,true),{once:true});
  }
  $('variant').addEventListener('change',variant);$('source').addEventListener('change',choose);
  $('play').addEventListener('click',async()=>{ $('play-error').hidden=true;if(!$('player').paused){$('player').pause();return;}try{await $('player').play();}catch{$('play-error').hidden=false;} });
  $('player').addEventListener('play',()=>{$('play').textContent='Ⅱ Пауза';});$('player').addEventListener('pause',()=>{$('play').textContent='▶ Воспроизвести';});
  $('time').addEventListener('input',()=>{$('player').pause();render(Number($('time').value),true);});
  $('prev').addEventListener('click',()=>{$('player').pause();render(Math.max(0,index-1),true);});
  $('next').addEventListener('click',()=>{$('player').pause();render(Math.min(rows.length-1,index+1),true);});
  $('player').addEventListener('timeupdate',()=>{const i=rows.findLastIndex(r=>r.t<=$('player').currentTime+1e-6);if(i!==index)render(i);});
  variant();choose();
})();
