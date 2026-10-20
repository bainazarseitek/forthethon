'use strict';
const D = window.ACTION_DIAGNOSIS;
const $ = id => document.getElementById(id);
const titles = Object.fromEntries(D.classes.map(c => [c.id, c.title]));
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const base = 'generated/action-diagnosis/';
const file = i => String(i).padStart(4, '0');
const desc = w => `${w.video} · ${w.start.toFixed(3)}–${w.end.toFixed(3)} с`;
function card(index, heading, extra='') {
  const w = D.windows[index];
  return `<article><h3>${esc(heading)}</h3><p>${esc(desc(w))}<br><strong>${esc(titles[w.label])}</strong> · ${esc(w.split)}</p><video controls muted playsinline preload="metadata" src="${base}${file(index)}.mp4"></video>${extra}<p class="muted">Разметка: ${esc(w.seen)}</p>${D.notes.notes[index] ? `<p class="note">Наблюдение: ${esc(D.notes.notes[index])}</p>` : ''}<details><summary>Шесть исходных кадров с увеличением</summary><img loading="lazy" src="${base}${file(index)}.jpg" alt="Кадры ${esc(desc(w))}"></details><details><summary>Фактический вход encoder</summary><img loading="lazy" src="generated/action-pilot/${esc(w.thumbnail)}" alt="Полный кадр на входе encoder"></details></article>`;
}
for (const q of D.queries) {
  const opt = document.createElement('option');opt.value=q.index;opt.textContent=`${desc(D.windows[q.index])} → ${titles[q.prediction]}`;$('query').append(opt);
}
$('query').value='97';
$('hands').innerHTML = D.train_hands.map(i=>card(i,'Train: ручной пересчёт', `<button data-hand="${i}">Сравнить с выбранным validation</button>`)).join('');
$('coverage').textContent = `Ручной пересчёт: ${D.coverage.hand_count.windows} окна / ${D.coverage.hand_count.videos} видео / ${D.coverage.hand_count.scenario_groups} группы сцен. Машинка: ${D.coverage.machine.windows} окон / ${D.coverage.machine.videos} видео / ${D.coverage.machine.intervals} интервалов. Cosine X-2 ↔ X-3 = ${D.train_hand_cosine[0][1].toFixed(3)}; X-2 ↔ X-8 = ${D.train_hand_cosine[0][2].toFixed(3)}; X-3 ↔ X-8 = ${D.train_hand_cosine[1][2].toFixed(3)}.`;
$('binary').textContent = 'Дополнительная бинарная проверка: только train «пересчёт / машинка» (3 / 46 окон), новый scaler только по этим 49 train-окнам, C=1 зафиксирован из предыдущего v2-опыта. На validation: пересчёт 0/5, машинка 6/6 — все 11 окон названы машинкой. Значит, конкуренция с документом/сейфом не является единственным объяснением. Это диагностика, а не новый подобранный benchmark.';
let selected = null;
function render(reference=null, handReference=null) {
  document.querySelectorAll('#comparison video').forEach(v=>v.pause());
  selected=D.queries.find(q=>q.index===Number($('query').value));
  const s=selected.spaces[$('space').value];
  const probs=D.classes.map((c,i)=>`${c.title}: ${(selected.scores[i]*100).toFixed(1)}%`).join(' · ');
  const hand=handReference || s.best_hand;const machine=reference || s.best_machine;
  $('comparison').innerHTML = card(selected.index,'Validation: ошибка',`<p><strong>Предсказание: ${esc(titles[selected.prediction])}</strong></p><p class="muted">Оценки головы, не калиброванные вероятности: ${esc(probs)}</p>`)+card(hand.index,handReference?'Выбранный train-пересчёт':'Ближайший train-пересчёт',`<p>Cosine ${hand.cosine.toFixed(3)} · ранг ${hand.rank}/123</p>`)+card(machine.index,reference?'Выбранный train-сосед':'Ближайшая train-машинка',`<p>Cosine ${machine.cosine.toFixed(3)} · ранг ${machine.rank}/123</p>`);
  $('neighbors').innerHTML=s.top.map(n=>`<tr><td>${n.rank}</td><td>${esc(desc(D.windows[n.index]))}</td><td>${esc(titles[D.windows[n.index].label])}</td><td>${n.cosine.toFixed(3)}</td><td><button data-index="${n.index}">Сравнить</button></td></tr>`).join('');
  $('neighbors').querySelectorAll('button').forEach(b=>b.onclick=()=>render(s.top.find(n=>n.index===Number(b.dataset.index))));
  $('scrub').value=0;$('status').textContent='';
}
$('hands').querySelectorAll('[data-hand]').forEach(b=>b.onclick=()=>render(null,selected.spaces[$('space').value].all_hand.find(n=>n.index===Number(b.dataset.hand))));
$('query').onchange=()=>render();$('space').onchange=()=>render();
$('play').onclick=async()=>{const videos=[...document.querySelectorAll('#comparison video')];videos.forEach(v=>{v.currentTime=0;v.playbackRate=Number($('speed').value);});const r=await Promise.allSettled(videos.map(v=>v.play()));$('status').textContent=r.some(x=>x.status==='rejected')?'Нажмите воспроизведение на нужном видео.':'Совместный старт';};
$('pause').onclick=()=>document.querySelectorAll('#comparison video').forEach(v=>v.pause());
$('speed').onchange=()=>document.querySelectorAll('#comparison video').forEach(v=>v.playbackRate=Number($('speed').value));
$('scrub').oninput=()=>document.querySelectorAll('#comparison video').forEach(v=>{v.pause();v.currentTime=Number($('scrub').value);});
render();
