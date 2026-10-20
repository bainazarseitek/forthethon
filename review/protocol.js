(() => {
  const data = window.PROTOCOL_PILOT, $ = id => document.getElementById(id), base = 'generated/protocol-pilot/';
  if (!data) { $('load-error').hidden = false; return; }
  const names = {count:'Пересчёт купюр',put_table:'Положить на стол',carry:'Нести деньги',store:'Положить в сейф',close:'Закрыть сейф',leave:'Уйти от стола',idle:'Пустое рабочее место',cash:'Купюры',clear:'Пустой участок',occluded:'Закрыто руками',paper:'Бумага / чек',unknown:'Неясно'};
  const stages = {cash_on_table:'Помнит деньги на столе',closing_seen:'Наблюдалось действие, похожее на закрытие',storage_seen:'Наблюдалось действие, похожее на помещение в сейф',counting_seen:'Наблюдался пересчёт',unconfirmed:'Этап пока не подтверждён'};
  const methods = {without_memory:'Без памяти',with_memory:'С памятью'};
  const first = x => x === null ? 'нет' : `${x.toFixed(1)} с`;
  const cards = [
    ['Без памяти',data.metrics.without_memory.auc.toFixed(3),'AUC на 47 моментах обеих записей'],
    ['С памятью',data.metrics.with_memory.auc.toFixed(3),'AUC на тех же визуальных наблюдениях'],
    ['Изменилось решений',String(data.memory_changed_alarms),`Память изменила численную оценку в ${data.memory_changed_decisions} моментах`]
  ];
  $('summary').innerHTML = cards.map(([title,value,note])=>`<article class="metric"><h2>${title}</h2><strong>${value}</strong><p>${note}</p></article>`).join('');
  $('conclusion').textContent = data.memory_changed_alarms === 0 ? `Преимущество памяти не показано: решения одинаковы. Несмотря на высокий AUC, при пороге 0.5 пропущено ${data.metrics.with_memory.false_negative} из ${data.metrics.with_memory.true_positive+data.metrics.with_memory.false_negative} моментов нарушения. CLIP путает действия и присутствие кассира; память не исправляет эти ошибки.` : 'Память изменила часть решений. Сравните пропуски и ложные тревоги ниже: изменение само по себе ещё не означает улучшение.';
  document.querySelector('#comparison tbody').innerHTML = Object.entries(methods).map(([key,label])=>{
    const m=data.metrics[key]; return `<tr><td>${label}</td><td>${m.auc.toFixed(3)}</td><td>${m.true_positive}</td><td>${m.false_negative}</td><td>${m.false_positive}</td><td>${first(m.first_alarm['X-26'])}</td><td>${first(m.first_alarm['X-27'])}</td></tr>`;
  }).join('');
  const audit=data.object_metrics;
  $('object-audit').textContent=`Проверка денег на явно размеченных поздних кадрах: общий план ${audit.table.table_scene.correct_top1}/${audit.table.table_scene.total}, участок стола ${audit.table.table.correct_top1}/${audit.table.table.total}. Присутствие у стола: ${audit.presence.correct_top1}/${audit.presence.total}. Участок выбран после визуального знакомства с этой парой.`;
  const run=data.run, latencies=run.per_frame_forward_seconds.slice(1).sort((a,b)=>a-b);
  $('run-info').textContent=`Модель: ${run.model}\nРевизия: ${run.revision}\nУстройство: ${run.gpu}\nРазмер признака: ${run.dimension}\nЗагрузка: ${run.model_load_seconds.toFixed(2)} с; кодирование текстов: ${run.text_seconds.toFixed(2)} с\nВычисление признаков 95 кадров × 2 области: ${run.image_forward_seconds.toFixed(2)} с\nМедиана одного обновления после первого: ${(latencies[Math.floor(latencies.length/2)]*1000).toFixed(0)} мс (две области, только forward)\nЭто повтор записей с причинным доступом к кадрам, не проверка живой камеры. Время передачи по SSH и декодирования сюда не входит.\nПороги эвристические; тестовой выборки для калибровки нет.`;
  let rows=[], index=-1, imageRevision=0;
  function bars(id, values) {
    const best=Object.keys(values).reduce((a,b)=>values[a]>values[b]?a:b);
    $(id).innerHTML=Object.entries(values).map(([key,value])=>`<div class="score ${key===best?'best':''}"><span>${names[key]}</span><div class="score-track"><span style="width:${Math.max(0,Math.min(100,value*200))}%"></span></div><span>${value.toFixed(3)}</span></div>`).join('');
  }
  function render(i, seek=false) {
    if (i < 0) { $('evidence').hidden=true; $('timestamp').textContent='Ждём 1 с'; $('history').textContent='Оценок ещё нет'; $('expected').textContent=''; $('manual').textContent='Начало записи'; index=-1; return; }
    const r=rows[i]; index=i; $('evidence').hidden=false; $('time').value=i;
    if(seek) $('player').currentTime=r.t;
    $('timestamp').textContent=`${r.t.toFixed(2)} с`;
    $('manual').textContent=`${r.label==='violation'?'Нарушение':r.label==='normal'?'Норма':'Не размечено'} · ${r.manual_note}`;
    $('table-state').textContent=`Сейчас: ${names[r.table_observation]}`;
    $('memory-state').textContent=stages[r.stage];
    $('risk').innerHTML=Object.entries(methods).map(([key,label])=>`<div class="${r[key]>=data.config.alarm_threshold?'alarm':''}">${label}<strong>${r[key].toFixed(3)}</strong>${r[key]>=data.config.alarm_threshold?'Тревога':'Ниже порога'}</div>`).join('');
    $('reason').textContent=`Кадр: ${r.pts.toFixed(3)} с. Поддержка «кассир отошёл»: ${r.away_support.toFixed(3)}. ${r.memory_evidence_t===null?'Сохранённого свидетельства денег нет.':`Последнее подтверждение денег: ${r.memory_evidence_t.toFixed(2)} с.`} ${r.memory_used?'Использовано сохранённое свидетельство.':''}`;
    bars('action-scores',r.actions); bars('table-scores',r.table);
    $('history').textContent=Object.entries(r.seen).map(([k,t])=>`${names[k]} (${t.toFixed(1)} с)`).join(' → ') || 'Уверенно подтверждённых действий пока нет';
    $('expected').textContent=`Следующие возможные действия: ${r.expected.map(k=>names[k].toLowerCase()).join(', ')}. Это подсказка: остальные действия тоже оцениваются, обязательный порядок не навязывается.`;
    $('prev').disabled=i===0; $('next').disabled=i===rows.length-1;
    const revision=++imageRevision, img=new Image();
    img.onload=()=>{ if(revision!==imageRevision)return; const [x,y,x2,y2]=data.config.table_box; const ctx=$('patch').getContext('2d'); ctx.drawImage(img,x,y,x2-x,y2-y,0,0,90,70); };
    img.src=base+r.image;
  }
  function choose() {
    rows=data.rows.filter(r=>r.video===$('source').value); $('time').max=rows.length-1;
    $('player').src=data.sources[$('source').value].preview; index=-1;
    $('player').addEventListener('loadedmetadata',()=>render(0,true),{once:true});
  }
  $('source').addEventListener('change',choose);
  $('play').addEventListener('click',async()=>{
    $('play-error').hidden=true;
    if(!$('player').paused) { $('player').pause(); return; }
    try { await $('player').play(); } catch { $('play-error').hidden=false; }
  });
  $('player').addEventListener('play',()=>{ $('play').textContent='Ⅱ Пауза'; });
  $('player').addEventListener('pause',()=>{ $('play').textContent='▶ Воспроизвести'; });
  $('time').addEventListener('input',()=>{ $('player').pause(); render(Number($('time').value),true); });
  $('prev').addEventListener('click',()=>{ $('player').pause(); render(Math.max(0,index-1),true); });
  $('next').addEventListener('click',()=>{ $('player').pause(); render(Math.min(rows.length-1,index+1),true); });
  $('player').addEventListener('timeupdate',()=>{
    let i=-1; for(let j=0;j<rows.length;j++) if(rows[j].t <= $('player').currentTime+1e-6) i=j;
    if(i!==index)render(i);
  });
  choose();
})();
