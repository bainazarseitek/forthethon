"use strict";
const data = window.REVIEW_DATA;
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const clean = text => String(text ?? "").replaceAll("**", "");
const fmt = value => `${Number(value).toFixed(2)} s`;
const byId = new Map(data.clips.map(c => [c.id, c]));
const rules = new Map(data.rules.types.map(r => [r.type_id, r]));
let current = data.clips[0];
let mode = "review";
let activeKey = "";

function label(c) { return c.official_time === null ? "No violation · official label" : `Violation · official ${fmt(c.official_time)}`; }
function seek(video, t) {
  const target = Math.max(0, Math.min(t, Number.isFinite(video.duration) ? video.duration - .001 : t));
  if (video.readyState) video.currentTime = target;
  else video.onloadedmetadata = () => { video.currentTime = target; video.onloadedmetadata = null; };
}
function switchView(view) {
  mode = view;
  document.querySelectorAll("video").forEach(v => v.pause());
  for (const id of ["review", "compare", "overview"]) $(id).hidden = view !== id;
  document.querySelectorAll("nav button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  if (view === "overview") overview();
}
function cautions(c) {
  const messages = [];
  if (c.same_content.length > 1) messages.push(`Identical file contents: ${c.same_content.join(" / ")}. Keep together in evaluation. ${c.id === "X-4" || c.id === "X-47" ? "Official labels refer to 57 s and 68 s; inspect both events." : "Count once when measuring independent examples."}`);
  for (const type of new Set(c.ground_truth.events.map(e => e.type_id))) {
    const rule = rules.get(type);
    if (rule?.caution) messages.push(clean(rule.caution));
    if (rule && rule.basis !== "direct") messages.push(`${type}: source catalogue marks legal basis as “${rule.basis}”. This is a research note, not a verified legal conclusion.`);
  }
  if (["X-11", "X-12", "X-28", "X-29"].includes(c.id)) messages.push("An open or closed laptop lid alone does not establish its lock state. Inspect the screen and the action before departure.");
  if (c.id === "X-1") messages.push("Returning notes is visible; their authenticity and the reason for returning them are not established by the video.");
  return messages.map(m => `<p class="warning">${esc(m)}</p>`).join("");
}
function selectClip(id, t = 0) {
  current = byId.get(id); activeKey = "";
  $("clip-select").value = id;
  const index = data.clips.indexOf(current);
  $("previous").disabled = index === 0; $("next").disabled = index === data.clips.length - 1;
  $("clip-heading").innerHTML = `<div class="clip-title"><h2>${esc(id)}</h2><span class="tag ${current.label}">${label(current)}</span><span class="muted">${fmt(current.duration)} · ${index + 1} of ${data.clips.length}</span></div>`;
  const player = $("player"); player.pause(); player.onloadedmetadata = null;
  player.src = current.video; player.poster = current.frames[0].file; player.playbackRate = Number($("speed").value);
  $("media-error").hidden = true;
  $("manual-note").textContent = current.manual_note || "No additional spreadsheet comment.";
  $("operation").textContent = clean(current.annotation.operation);
  $("cautions").innerHTML = cautions(current);
  $("jump-label").disabled = current.official_time === null;
  $("source-text").textContent = `Sources: docs/annotations.md; assets/team_labels.xlsx; assets/open_labels.xlsx\nOriginal filename: ${current.source}\nSHA-256: ${current.sha256}\nAnnotation status in source: ${current.annotation.status}\n${current.annotation.inherited_from ? "Timeline inherited from identical " + current.annotation.inherited_from + "\n" : ""}\n${current.annotation.source_text}`;
  $("event-list").innerHTML = current.annotation.rows.map((r, i) => `<button type="button" class="event ${esc(r.kind)}" data-event="${i}"><small>${fmt(r.start)}${r.end !== r.start ? " – " + fmt(r.end) : ""} · ${r.kind === "ok" ? "Correct step" : r.kind === "viol" ? "Flagged in notes" : "Context / uncertain"}</small><strong>${esc(r.types || "Observation")}</strong><p lang="ru">${esc(clean(r.seen))}</p></button>`).join("");
  $("filmstrip").innerHTML = current.frames.map(f => `<button type="button" class="frame ${current.official_time !== null && Math.abs(f.t - current.official_time) < .1 ? "near-label" : ""}" data-time="${f.t}" aria-label="Seek to ${fmt(f.t)}"><img src="${f.file}" loading="lazy" alt="${id} at ${fmt(f.t)}"><span>${fmt(f.t)}</span></button>`).join("");
  const related = data.pairs.map((p, i) => ({p,i})).filter(({p}) => p.includes(id));
  $("pair-links").innerHTML = related.map(({p,i}) => `<button type="button" data-pair="${i}">Compare ${p[0]} / ${p[1]}</button>`).join("");
  seek(player, t); updateTime();
  history.replaceState(null, "", `#${id}`);
}
function updateTime() {
  const t = $("player").currentTime;
  $("clock").textContent = `${t.toFixed(3)} s`;
  const active = current.annotation.rows.map((r,i) => ({r,i})).filter(({r}) => t >= r.start - .03 && t <= Math.max(r.end, r.start + .25));
  const key = current.id + ":" + active.map(x => x.i).join(",");
  if (key === activeKey) return;
  activeKey = key;
  $("active-events").innerHTML = active.length ? active.map(({r}) => `<div class="observation"><small>${esc(r.types || "Context")} · ${fmt(r.start)}–${fmt(r.end)}</small><p lang="ru">${esc(clean(r.seen))}</p>${r.rule ? `<small>Source note: ${esc(r.rule)}</small>` : ""}</div>`).join("") : '<p class="muted">No manual observation at this time. Unannotated time is not evidence of normal behavior.</p>';
  document.querySelectorAll(".event").forEach(b => b.classList.toggle("active", active.some(x => x.i === Number(b.dataset.event))));
}
function compare(index) {
  document.querySelectorAll("#comparison video").forEach(v => v.pause());
  $("pair-select").value = index;
  const pair = data.pairs[index];
  $("comparison").innerHTML = pair.slice(0,2).map((id,i) => {
    const c = byId.get(id);
    return `<article><h2>${id} <span class="tag ${c.label}">${label(c)}</span></h2><video id="compare-${i}" src="${c.video}" poster="${c.frames[0].file}" controls playsinline preload="metadata"></video><div class="toolbar"><button type="button" data-compare-start="${i}">Start</button>${c.official_time === null ? "" : `<button type="button" data-compare-label="${i}">Official ${fmt(c.official_time)}</button>`}<button type="button" data-open="${id}">Open full review</button><label>Jump to step <select aria-label="${id} step" data-compare-step="${i}"><option value="">Choose an observation</option>${c.annotation.rows.map(r=>`<option value="${r.start}">${fmt(r.start)} · ${esc(r.types || "Context")}</option>`).join("")}</select></label></div><p lang="ru">${esc(c.manual_note || clean(c.annotation.operation))}</p></article>`;
  }).join("");
}
function overview() {
  const unique = $("unique").checked, filter = $("filter").value;
  const clips = data.clips.filter(c => (!unique || !c.duplicate_of) && (filter === "all" || c.label === filter));
  const max = Math.ceil(Math.max(...data.clips.map(c => c.duration)) / 10) * 10;
  let html = `<div class="axis">${Array.from({length:6}, (_,i) => `<span style="left:${i*20}%">${Math.round(i*max/5)}</span>`).join("")}</div>`;
  html += clips.map(c => {
    const family = unique ? c.same_content.map(id => byId.get(id)) : [c];
    const events = family.flatMap(v => v.ground_truth.events);
    const labels = [...new Set(family.map(v => v.official_time).filter(t => t !== null))];
    const marks = events.map(e => `<span class="event-mark" style="left:${e.t_start_sec/max*100}%;width:${Math.max(.3,(e.t_end_sec-e.t_start_sec)/max*100)}%"></span>`).join("");
    return `<div class="timeline-row"><button type="button" class="timeline-label" data-open="${c.id}">${esc(unique ? c.same_content.join(" / ") : c.id)}<br><small>${c.label === "clean" ? "No violation label" : "Violation label"}</small></button><button type="button" class="track" data-open="${c.id}" aria-label="Review ${c.id}; ${fmt(c.duration)}; ${labels.length ? "official timestamps " + labels.join(", ") : "no violation label"}"><span class="length" style="width:${c.duration/max*100}%"></span>${marks}${labels.map(t => `<span class="label-mark" style="left:${t/max*100}%"></span>`).join("")}</button></div>`;
  }).join("");
  $("timeline-chart").innerHTML = html;
  const types = new Map();
  for (const c of data.clips) for (const e of c.ground_truth.events) {
    if (!types.has(e.type_id)) types.set(e.type_id,new Set());
    types.get(e.type_id).add(c.same_content[0]);
  }
  const counts = [...types].sort((a,b) => b[1].size-a[1].size || a[0].localeCompare(b[0]));
  const maximum = Math.max(...counts.map(x=>x[1].size));
  $("pattern-list").innerHTML = counts.map(([type,ids]) => `<div class="pattern"><span>${esc(type)}<br><small>${[...ids].map(id => `<button type="button" data-open="${id}">${id}</button>`).join(" · ")}</small></span><span class="pattern-bar" style="width:${ids.size/maximum*100}%"></span><span class="pattern-count">${ids.size}</span></div>`).join("");
}
$("summary").textContent = `${data.summary.clips} clips · ${data.summary.unique} unique videos · ${(data.summary.unique_duration/60).toFixed(1)} minutes`;
$("clip-select").innerHTML = data.clips.map(c => `<option value="${c.id}">${c.id} · ${c.label === "clean" ? "No violation" : "Violation at " + fmt(c.official_time)}</option>`).join("");
$("pair-select").innerHTML = data.pairs.map((p,i) => `<option value="${i}">${p[0]} / ${p[1]} · ${esc(p[2])}</option>`).join("");
document.querySelector("nav").addEventListener("click", e => { if (e.target.dataset.view) switchView(e.target.dataset.view); });
$("clip-select").onchange = e => selectClip(e.target.value);
$("previous").onclick = () => selectClip(data.clips[data.clips.indexOf(current)-1].id);
$("next").onclick = () => selectClip(data.clips[data.clips.indexOf(current)+1].id);
$("back").onclick = () => seek($("player"), $("player").currentTime-1);
$("forward").onclick = () => seek($("player"), $("player").currentTime+1);
$("speed").onchange = e => { $("player").playbackRate = Number(e.target.value); };
$("jump-label").onclick = () => seek($("player"),current.official_time);
$("player").ontimeupdate = updateTime;
$("player").onerror = () => { $("media-error").hidden = false; };
$("player").onended = () => {
  const i = data.clips.indexOf(current);
  if ($("autonext").checked && i < data.clips.length-1) { selectClip(data.clips[i+1].id); $("player").play().catch(() => {}); }
};
$("pair-select").onchange = e => compare(Number(e.target.value));
$("play-both").onclick = () => document.querySelectorAll("#comparison video").forEach(v => v.play().catch(() => {}));
$("pause-both").onclick = () => document.querySelectorAll("#comparison video").forEach(v => v.pause());
$("filter").onchange = overview; $("unique").onchange = overview;
$("comparison").addEventListener("change", e => {
  if (e.target.dataset.compareStep !== undefined && e.target.value !== "") seek($("compare-"+e.target.dataset.compareStep),Number(e.target.value));
});
document.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.event !== undefined) seek($("player"),current.annotation.rows[Number(b.dataset.event)].start);
  if (b.dataset.time !== undefined) seek($("player"),Number(b.dataset.time));
  if (b.dataset.pair !== undefined) { compare(Number(b.dataset.pair)); switchView("compare"); }
  if (b.dataset.open) { selectClip(b.dataset.open); switchView("review"); window.scrollTo({top:0,behavior:"smooth"}); }
  if (b.dataset.compareStart !== undefined) seek($("compare-"+b.dataset.compareStart),0);
  if (b.dataset.compareLabel !== undefined) seek($("compare-"+b.dataset.compareLabel),byId.get(data.pairs[Number($("pair-select").value)][Number(b.dataset.compareLabel)]).official_time);
});
compare(0);
selectClip(byId.has(location.hash.slice(1)) ? location.hash.slice(1) : data.clips[0].id);
