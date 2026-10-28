import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const $ = (s) => document.querySelector(s);
const GAP = 0.42;            // distance between layer planes
const UP = new THREE.Color("#f2a93b");
const DOWN = new THREE.Color("#ee5a9f");

const state = {
  info: null,
  layout: "index",
  selected: null,           // {layer, neuron}
  neuron: null,             // last neuron payload
  steers: [],               // [{layer, neuron, dose}]
  heat: null,               // Uint8Array L*N from last run
  lastRun: null,
  chat: [],                 // finished turns: [{user, reply, steers}]
  presets: [],
  searchBy: "lens",
  session: null,
  chatBusy: false,
};

// ------------------------------------------------------------------ helpers

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const tokText = (t) => esc(t === "\n" ? "↵" : t.replace(/\n/g, "↵"));
const nid = (l, n) => `L${l} N${n}`;
const fmt = (v) => (Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2));

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

function chip(t, cls = "") {
  const ws = /^\s+$/.test(t) ? " ws" : "";
  return `<span class="chip ${cls}${ws}">${ws ? esc(JSON.stringify(t).slice(1, -1)) : esc(t)}</span>`;
}

function refBtn(r) {
  const hint = r.hint !== undefined ? `<span class="h">${esc(r.hint.trim() || JSON.stringify(r.hint))}</span>` : "";
  return `<button class="ref" data-l="${r.layer}" data-n="${r.neuron}" title="weight ${r.w ?? r.score ?? r.rel ?? ""}">${nid(r.layer, r.neuron)}${hint}</button>`;
}

// shade a token by activation relative to a max: amber for positive, pink for negative
function shade(v, max) {
  const a = Math.max(-1, Math.min(1, v / (max || 1)));
  if (a > 0.04) return `background: rgba(242,169,59,${(a * 0.75).toFixed(2)})`;
  if (a < -0.04) return `background: rgba(238,90,159,${(-a * 0.6).toFixed(2)})`;
  return "";
}

function tokens(toks, acts, max, peak = -1) {
  return toks.map((t, i) => `<span class="tok${i === peak ? " peak" : ""}" style="${shade(acts[i], max)}" title="${fmt(acts[i])}">${esc(t)}</span>`).join("");
}

// ------------------------------------------------------------------ 3D scene

const stage = $("#stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x0a0b0d);
stage.prepend(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.cssText = "position:absolute;inset:0;pointer-events:none";
stage.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0a0b0d, 14, 40);
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.6;

let points, heatAttr, markAttr, flashAttr, posAttr, planes = [], layerLabels = [];
const selGroup = new THREE.Group();
scene.add(selGroup);

const pointMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  uniforms: { uScale: { value: 1 }, uFocus: { value: -1 } },
  vertexShader: /* glsl */ `
    attribute float heat;
    attribute float mark;
    attribute float layer;
    attribute float flash;
    uniform float uScale;
    uniform float uFocus;
    varying float vHeat;
    varying float vMark;
    varying float vDim;
    varying float vFlash;
    void main() {
      vMark = mark;
      vFlash = flash;
      vDim = (uFocus < 0.0 || abs(layer - uFocus) < 0.5) ? 1.0 : 0.35;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float h = smoothstep(0.35, 1.0, heat);
      vHeat = h;
      float size = 2.4 + h * 4.0 + mark * 4.0 + flash * 9.0;
      gl_PointSize = min(size * uScale / -mv.z, 3.0 + h * 9.0 + mark * 7.0 + flash * 16.0);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */ `
    varying float vHeat;
    varying float vMark;
    varying float vDim;
    varying float vFlash;
    void main() {
      float d = length(gl_PointCoord - 0.5);
      if (d > 0.5) discard;
      float soft = smoothstep(0.5, 0.15, d);
      vec3 base = vec3(0.45, 0.66, 0.78);
      vec3 hot = vec3(1.0, 0.95, 0.86);
      vec3 col = mix(base, hot, smoothstep(0.05, 0.6, vHeat));
      col = mix(col, vec3(0.95, 0.66, 0.23), vMark);
      col = mix(col, vec3(1.0, 0.78, 0.38), vFlash);
      float a = (0.2 + vHeat * 0.85 + vMark * 0.9 + vFlash * 1.4) * soft * max(vDim, vFlash);
      gl_FragColor = vec4(col * a, a);
    }`,
});

const planeMat = new THREE.MeshBasicMaterial({ color: 0x8fd3f4, transparent: true, opacity: 0.025, side: THREE.DoubleSide, depthWrite: false });
const planeFocusMat = planeMat.clone();
planeFocusMat.opacity = 0.07;
const edgeMat = new THREE.LineBasicMaterial({ color: 0x8fd3f4, transparent: true, opacity: 0.22 });

const layerX = (l) => (l - (state.info.layers - 1) / 2) * GAP;

function buildScene(layout) {
  const { layers: L, neurons: N, types } = state.info;
  const pos = new Float32Array(L * N * 3);
  const layerIdx = new Float32Array(L * N);
  for (let l = 0; l < L; l++) {
    for (let n = 0; n < N; n++) {
      const i = l * N + n;
      pos[i * 3] = layerX(l);
      pos[i * 3 + 1] = layout[i * 2 + 1];
      pos[i * 3 + 2] = layout[i * 2];
      layerIdx[i] = l;
    }
  }
  const geo = new THREE.BufferGeometry();
  posAttr = new THREE.BufferAttribute(pos, 3);
  heatAttr = new THREE.BufferAttribute(new Float32Array(L * N), 1);
  markAttr = new THREE.BufferAttribute(new Float32Array(L * N), 1);
  flashAttr = new THREE.BufferAttribute(new Float32Array(L * N), 1);
  flashAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("position", posAttr);
  geo.setAttribute("heat", heatAttr);
  geo.setAttribute("mark", markAttr);
  geo.setAttribute("flash", flashAttr);
  geo.setAttribute("layer", new THREE.BufferAttribute(layerIdx, 1));
  points = new THREE.Points(geo, pointMat);
  scene.add(points);

  const pgeo = new THREE.PlaneGeometry(2.12, 2.12);
  const egeo = new THREE.EdgesGeometry(pgeo);
  for (let l = 0; l < L; l++) {
    const plane = new THREE.Mesh(pgeo, planeMat.clone());
    plane.userData.edge = edgeMat.clone();
    plane.userData.pulse = 0;
    plane.rotation.y = Math.PI / 2;
    plane.position.x = layerX(l);
    plane.add(new THREE.LineSegments(egeo, plane.userData.edge));
    scene.add(plane);
    planes.push(plane);
    const div = document.createElement("div");
    const attn = types[l] !== "conv";
    div.className = `layer-label${attn ? " attn" : ""}`;
    div.textContent = attn ? `L${l} attn` : `L${l}`;
    const lab = new CSS2DObject(div);
    lab.position.set(layerX(l), 1.18, -1.06);
    scene.add(lab);
    layerLabels.push(div);
  }
  if (state.heat) applyHeat(state.heat);
}

function setLayout(layout) {
  const { layers: L, neurons: N } = state.info;
  for (let i = 0; i < L * N; i++) {
    posAttr.array[i * 3 + 1] = layout[i * 2 + 1];
    posAttr.array[i * 3 + 2] = layout[i * 2];
  }
  posAttr.needsUpdate = true;
  points.geometry.computeBoundingSphere();
  if (state.neuron) drawWeb(state.neuron);
}

function neuronPos(l, n) {
  const i = (l * state.info.neurons + n) * 3;
  return new THREE.Vector3(posAttr.array[i], posAttr.array[i + 1], posAttr.array[i + 2]);
}

function applyHeat(heat) {
  for (let i = 0; i < heat.length; i++) heatAttr.array[i] = heat[i] / 255;
  heatAttr.needsUpdate = true;
}

function applyMarks(results) {
  markAttr.array.fill(0);
  for (const r of results) markAttr.array[r.layer * state.info.neurons + r.neuron] = 1;
  markAttr.needsUpdate = true;
}

function focusLayer(l) {
  pointMat.uniforms.uFocus.value = l;
  planes.forEach((p, i) => (p.userData.focus = i === l));
  layerLabels.forEach((d, i) => d.classList.toggle("focus", i === l));
}

// camera glide
let glide = null;
function flyTo(target, offset) {
  glide = { t0: performance.now(), fromT: controls.target.clone(), toT: target, fromC: camera.position.clone(), toC: target.clone().add(offset) };
}
function stepGlide(now) {
  if (!glide) return;
  const k = Math.min(1, (now - glide.t0) / 700);
  const e = 1 - Math.pow(1 - k, 4);
  controls.target.lerpVectors(glide.fromT, glide.toT, e);
  camera.position.lerpVectors(glide.fromC, glide.toC, e);
  if (k === 1) glide = null;
}

function drawWeb(d) {
  selGroup.clear();
  const p0 = neuronPos(d.layer, d.neuron);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.028, 0.036, 32), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthTest: false }));
  ring.rotation.y = Math.PI / 2;
  ring.position.copy(p0);
  selGroup.add(ring);
  const verts = [], cols = [], ends = [], endCols = [];
  const groups = [["fed_by", UP], ["held_back_by", DOWN], ["pushes_toward", UP], ["pushes_away", DOWN]];
  for (const [key, col] of groups) {
    for (const r of (d.web?.[key] || []).slice(0, 8)) {
      const p = neuronPos(r.layer, r.neuron);
      verts.push(p0.x, p0.y, p0.z, p.x, p.y, p.z);
      cols.push(col.r, col.g, col.b, col.r * 0.6, col.g * 0.6, col.b * 0.6);
      ends.push(p.x, p.y, p.z);
      endCols.push(col.r, col.g, col.b);
    }
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  lg.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
  selGroup.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75, depthWrite: false })));
  const eg = new THREE.BufferGeometry();
  eg.setAttribute("position", new THREE.Float32BufferAttribute(ends, 3));
  eg.setAttribute("color", new THREE.Float32BufferAttribute(endCols, 3));
  selGroup.add(new THREE.Points(eg, new THREE.PointsMaterial({ size: 0.035, vertexColors: true, transparent: true, depthWrite: false })));
}

function resize() {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h);
  labels.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  pointMat.uniforms.uScale.value = h * renderer.getPixelRatio() * 0.012;
}
new ResizeObserver(resize).observe(stage);

function loop(now) {
  stepGlide(now);
  stepFlash(now);
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ live flashes
// Every generated token sends the neurons it lit up; they flash as a wave sweeping L0 -> L29.

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const WAVE_MS = reduceMotion ? 0 : 32;     // delay per layer: a full L0 -> L29 sweep takes ~1 s
const active = new Set();                  // indices with flash > 0
const pending = [];                        // { at, idx: Int32Array, val: Uint8Array }
let lastFlash = performance.now();

function queueFire(ix, v) {
  const N = state.info.neurons;
  const byLayer = new Map();
  for (let k = 0; k < ix.length; k++) {
    const l = Math.floor(ix[k] / N);
    if (!byLayer.has(l)) byLayer.set(l, []);
    byLayer.get(l).push(k);
  }
  const t0 = performance.now();
  addTrail(byLayer, ix, v, t0);
  for (const [l, ks] of byLayer) {
    pending.push({ at: t0 + l * WAVE_MS, layer: l, idx: ks.map((k) => ix[k]), val: ks.map((k) => v[k] / 255), n: ks.length });
  }
}

function stepFlash(now) {
  if (!flashAttr) return;
  const dt = Math.min(0.1, (now - lastFlash) / 1000);
  lastFlash = now;
  let dirty = false;
  for (let p = pending.length - 1; p >= 0; p--) {
    const f = pending[p];
    if (f.at > now) continue;
    for (let k = 0; k < f.idx.length; k++) {
      const i = f.idx[k];
      if (f.val[k] > flashAttr.array[i]) flashAttr.array[i] = f.val[k];
      active.add(i);
    }
    const pl = planes[f.layer];
    if (pl) pl.userData.pulse = Math.min(1, pl.userData.pulse + 0.25 + f.n / 400);
    pending.splice(p, 1);
    dirty = true;
  }
  const decay = Math.exp(-dt * 3.2);
  for (const i of active) {
    const v = flashAttr.array[i] * decay;
    if (v < 0.02) { flashAttr.array[i] = 0; active.delete(i); } else flashAttr.array[i] = v;
    dirty = true;
  }
  if (dirty) flashAttr.needsUpdate = true;
  stepTrails(now);
  const pd = Math.exp(-dt * 4);
  for (const pl of planes) {
    pl.userData.pulse *= pd;
    const base = pl.userData.focus ? 0.07 : 0.025;
    pl.material.opacity = base + pl.userData.pulse * 0.16;
    pl.userData.edge.opacity = 0.22 + pl.userData.pulse * 0.6;
    pl.material.color.setHex(pl.userData.pulse > 0.05 ? 0xf2c27a : 0x8fd3f4);
  }
  if (!$("#live").hidden) {
    // the bar follows the newest word only: cells fill as its signal reaches each layer
    const t = trails.find((x) => now - x.t0 < x.dur) || trails[trails.length - 1];
    const L = barCells.length;
    const at = t ? ((now - t.t0) / t.dur) * (L - 1) : -1;
    barCells.forEach((c, l) => {
      const d = at - l;
      const on = d >= 0 && at <= L - 1 + 4;
      const head = on && d < 1;
      c.style.background = head ? "#ffe2a8" : on ? `rgba(242, 169, 59, ${Math.max(0.25, 0.9 - d * 0.03)})` : "";
      c.style.boxShadow = head ? "0 0 10px rgba(255, 210, 140, 0.95)" : "";
    });
  }
}

// One glowing trail per word: it runs through the strongest neuron of every layer, L0 -> L29,
// so you can watch the signal travel through the network.
const trails = [];
const MAX_TRAILS = 6;
const IN_FLIGHT = 2;             // words arrive ~13/s, a sweep takes ~1 s: only follow a couple at once
const trailMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
const headMat = new THREE.PointsMaterial({ color: 0xffe2a8, size: 0.09, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });

function addTrail(byLayer, ix, v, t0) {
  const L = state.info.layers;
  const now = performance.now();
  if (trails.filter((t) => now - t.t0 < t.dur).length >= IN_FLIGHT) return;
  const pts = [];
  for (let l = 0; l < L; l++) {
    const ks = byLayer.get(l);
    if (!ks) continue;
    let best = ks[0];
    for (const k of ks) if (v[k] > v[best]) best = k;
    const i = ix[best];
    pts.push(neuronPos(l, i % state.info.neurons));
  }
  if (pts.length < 2) return;
  const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
  const n = pts.length * 8;
  const samples = curve.getPoints(n);
  const geo = new THREE.BufferGeometry().setFromPoints(samples);
  geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(samples.length * 3), 3));
  const line = new THREE.Line(geo, trailMat.clone());
  const head = new THREE.Points(new THREE.BufferGeometry().setFromPoints([samples[0]]), headMat.clone());
  scene.add(line, head);
  trails.push({ line, head, samples, t0, dur: Math.max(1, (L - 1) * WAVE_MS), word: live.lastTok });
  while (trails.length > MAX_TRAILS) killTrail(trails.shift());
}

function killTrail(t) {
  scene.remove(t.line, t.head);
  t.line.geometry.dispose(); t.head.geometry.dispose();
  t.line.material.dispose(); t.head.material.dispose();
}

function stepTrails(now) {
  for (let k = trails.length - 1; k >= 0; k--) {
    const t = trails[k];
    const prog = (now - t.t0) / t.dur;                 // 0..1 while travelling
    const fade = prog <= 1 ? 1 : Math.max(0, 1 - (prog - 1) * 1.4);
    if (fade <= 0) { killTrail(t); trails.splice(k, 1); continue; }
    const S = t.samples.length;
    const headAt = Math.min(S - 1, Math.max(0, Math.floor(prog * (S - 1))));
    const col = t.line.geometry.attributes.color.array;
    for (let i = 0; i < S; i++) {
      const behind = headAt - i;                       // how far behind the head this sample is
      const a = i > headAt ? 0 : Math.exp(-behind / (S * 0.25)) * fade;
      col[i * 3] = a; col[i * 3 + 1] = a * 0.72; col[i * 3 + 2] = a * 0.35;
    }
    t.line.geometry.attributes.color.needsUpdate = true;
    t.head.geometry.attributes.position.array.set([t.samples[headAt].x, t.samples[headAt].y, t.samples[headAt].z]);
    t.head.geometry.attributes.position.needsUpdate = true;
    t.head.material.opacity = prog <= 1 ? 1 : fade * 0.5;
  }
}

const barCells = [];
function buildLiveBar() {
  const bar = $("#liveBar");
  state.info.types.forEach((t) => {
    const i = document.createElement("i");
    if (t !== "conv") i.className = "attn";
    bar.appendChild(i);
    barCells.push(i);
  });
}

const b64bytes = (b) => Uint8Array.from(atob(b), (c) => c.charCodeAt(0));
const live = { who: "", steered: "", baseline: "", lane: "steered", tokens: 0, fired: 0 };

function renderLive(done = false) {
  const el = $("#live");
  el.hidden = false;
  el.classList.toggle("done", done);
  $("#liveWho").textContent = done ? `${live.who} · done` : live.lane === "baseline" ? `${live.who} · unsteered pass` : live.who;
  $("#liveCount").textContent = live.tokens ? `${live.fired.toLocaleString()} neurons lit · word ${live.tokens}` : "";
  const txt = live.lane === "baseline" ? live.baseline : live.steered;
  const tail = txt.slice(-220);
  const cut = tail.length - (live.lastTok || "").length;
  $("#liveText").innerHTML = `<span class="${done ? "" : "caret"}">${esc(tail.slice(0, cut))}<b>${esc(tail.slice(cut))}</b></span>`;
  if (live.source === "you" && !done) {
    const base = live.baseline ? `<p class="label">Unsteered</p><p class="reply" style="color:var(--dim)">${esc(live.baseline)}</p>` : "";
    const older = state.chat.map((t) => `<div class="turn old"><p class="you">${esc(t.user)}</p><p class="reply">${esc(t.reply)}</p></div>`).join("");
    $("#runOutput").innerHTML = `${older}<div class="turn"><p class="you">${esc(live.prompt)}</p><p class="label">Reply · writing live</p><p class="reply caret">${esc(live.steered)}</p>${base}</div>`;
    $("#runOutput").scrollTop = $("#runOutput").scrollHeight;
  }
}

let hideLive;
function onLiveEvent(ev) {
  if (!state.info) return;
  if (ev.kind === "start") {
    clearTimeout(hideLive);
    Object.assign(live, { who: ev.source === "claude" ? "Claude is testing" : "Writing", source: ev.source, prompt: ev.prompt, steered: "", baseline: "", lane: ev.baseline ? "baseline" : "steered", tokens: 0, fired: 0, lastTok: "" });
    if (ev.steers?.length) live.who += " · steered " + ev.steers.map((s) => `${nid(s.layer, s.neuron)} ${s.dose > 0 ? "+" : ""}${s.dose}`).join(", ");
    renderLive();
  } else if (ev.kind === "token") {
    live.lane = ev.lane;
    live[ev.lane] += ev.text;
    live.lastTok = ev.text;
    live.tokens++;
    renderLive();
  } else if (ev.kind === "fire") {
    const ix = new Int32Array(b64bytes(ev.ix).buffer);
    live.fired = ix.length;
    queueFire(ix, b64bytes(ev.v));
  } else if (ev.kind === "end") {
    if (live.source === "claude") {
      state.heat = b64bytes(ev.heat);
      applyHeat(state.heat);
    }
    renderLive(true);
    hideLive = setTimeout(() => ($("#live").hidden = true), 6000);
  }
}

function connectLive() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => onLiveEvent(JSON.parse(m.data));
  es.onerror = () => { es.close(); setTimeout(connectLive, 2000); };
}

// picking: click (not drag) selects the nearest dot under the cursor
const ray = new THREE.Raycaster();
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => (downAt = [e.clientX, e.clientY]));
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4 || !points) return;
  const r = renderer.domElement.getBoundingClientRect();
  const m = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(m, camera);
  ray.params.Points.threshold = 0.014 * camera.position.distanceTo(controls.target) / 4;
  const focus = pointMat.uniforms.uFocus.value;
  const hits = ray.intersectObject(points).filter((h) => focus < 0 || Math.floor(h.index / state.info.neurons) === focus);
  if (!hits.length) return;
  // prefer bright (fired / matched) dots among the closest hits
  hits.sort((a, b) => (b.distanceToRay < 0.02 ? heatAttr.array[b.index] + markAttr.array[b.index] : 0) - (a.distanceToRay < 0.02 ? heatAttr.array[a.index] + markAttr.array[a.index] : 0) || a.distance - b.distance);
  const i = hits[0].index;
  selectNeuron(Math.floor(i / state.info.neurons), i % state.info.neurons, { fly: false });
});

// ------------------------------------------------------------------ neuron panel

const panel = $("#neuronPanel");

function welcome() {
  panel.innerHTML = `
    <div class="welcome">
      <h2>Explore the model's neurons</h2>
      <p>Each sheet is one of the model's 30 layers. Each dot is one of its 10,752 neurons: a single number that rises for some kinds of text.</p>
      <ol>
        <li><b>Run a prompt</b> on the right. Neurons that fire on the reply glow.</li>
        <li><b>Click a dot</b> to see what that neuron does and which neurons it talks to.</li>
        <li><b>Steer</b> it up or down and run again to see the reply change.</li>
        <li>Or <b>ask Claude</b> to go hunting for you.</li>
      </ol>
      <p>Try opening <button class="ref" data-l="16" data-n="1399">L16 N1399</button>, the neuron that made replies angry in the first look.</p>
    </div>`;
}

async function selectNeuron(l, n, { fly = true } = {}) {
  if (!(l >= 0 && l < state.info.layers && n >= 0 && n < state.info.neurons)) return;
  state.selected = { layer: l, neuron: n };
  panel.innerHTML = `<div class="nhead"><h2>${nid(l, n)}</h2></div>${'<div class="skeleton"></div>'.repeat(6)}`;
  if (fly) {
    const p = neuronPos(l, n);
    flyTo(p, new THREE.Vector3(3.4, 1.2, 3.1));
  }
  try {
    const d = await api(`/api/neuron?layer=${l}&neuron=${n}`);
    if (state.selected?.layer !== l || state.selected?.neuron !== n) return;
    state.neuron = d;
    renderNeuron(d);
    drawWeb(d);
    renderRunOutput();
    renderSuggest();
    loadSteerLens(l, n);
  } catch (err) {
    panel.innerHTML = `<p class="err">Couldn't load ${nid(l, n)}: ${esc(err.message)}</p>`;
  }
}

function renderNeuron(d) {
  const key = `name.${d.layer}.${d.neuron}`;
  const attn = d.type !== "conv";
  const ex = (d.examples || []).map((e) => {
    const max = Math.max(...e.acts.map(Math.abs), 1e-6);
    return `<p class="snip"><span class="v">${fmt(e.act)}</span>${tokens(e.tokens, e.acts, max, e.at)}</p>`;
  }).join("");
  const web = [
    ["fed_by", "fed by", "upstream neurons that excite it", "var(--up)"],
    ["held_back_by", "held back by", "upstream neurons that inhibit it", "var(--down)"],
    ["pushes_toward", "feeds", "downstream neurons it excites", "var(--up)"],
    ["pushes_away", "pushes away", "downstream neurons it inhibits", "var(--down)"],
  ].filter(([k]) => d.web[k].length).map(([k, t, sub, c]) => `
    <div class="web-group"><p><i style="background:${c}"></i>${t} · ${sub}</p>
    <div class="chips">${d.web[k].slice(0, 10).map(refBtn).join("")}</div></div>`).join("");
  panel.innerHTML = `
    <div class="nhead">
      <h2>${nid(d.layer, d.neuron)}</h2>
      <span class="tag ${attn ? "attn" : ""}">${attn ? "attention" : "conv"} layer</span>
      <button class="iconbtn" id="closeNeuron" aria-label="Close neuron">
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      </button>
    </div>
    <input class="name" id="nameInput" placeholder="name this neuron" value="${esc(localStorage.getItem(key) || "")}" aria-label="Neuron name">
    ${d.max !== undefined ? `<p class="stats">max ${fmt(d.max)} · mean ${fmt(d.mean)} · std ${fmt(d.std)}</p>` : ""}

    <div class="sec">
      <h3>Steer</h3>
      <div class="steer-ctl">
        <input type="range" id="doseRange" min="-3" max="3" step="0.1" value="1" aria-label="Dose">
        <output id="doseOut">+1.0</output>
        <button class="btn" id="addSteer">Add</button>
      </div>
    </div>

    ${d.trace ? `<div class="sec"><h3>On the last reply<small>shaded by this neuron</small></h3>
      <p class="snip">${tokens(d.trace.tokens, d.trace.acts, d.max || Math.max(...d.trace.acts.map(Math.abs), 1e-6))}</p></div>` : ""}

    <div class="sec" id="steerLens"><h3>Steer lens<small>what steering does to the next word</small></h3><div class="skeleton"></div><div class="skeleton"></div></div>

    ${d.lens_up ? `<div class="sec"><h3>Logit lens<small>words its output pushes toward / away</small></h3>
      <div class="pole"><span class="sign up">+</span><div class="chips">${d.lens_up.map((t) => chip(t, "up")).join("")}</div></div>
      <div class="pole"><span class="sign down">−</span><div class="chips">${d.lens_down.map((t) => chip(t, "down")).join("")}</div></div></div>` : ""}

    ${ex ? `<div class="sec"><h3>Fires hardest on<small>top snippets from 3,000 web texts</small></h3>${ex}</div>` : ""}

    <div class="sec"><h3>Web<small>direct links through the weights</small></h3>${web || '<p class="empty">No links.</p>'}</div>`;

  $("#closeNeuron").onclick = clearNeuron;
  $("#nameInput").oninput = (e) => localStorage.setItem(key, e.target.value);
  const range = $("#doseRange");
  range.oninput = () => ($("#doseOut").textContent = (range.value > 0 ? "+" : "") + Number(range.value).toFixed(1));
  $("#addSteer").onclick = () => addSteer(d.layer, d.neuron, Number(range.value));
}

async function loadSteerLens(l, n) {
  try {
    const s = await api(`/api/steerlens?layer=${l}&neuron=${n}`);
    const box = $("#steerLens");
    if (!box || state.selected?.layer !== l || state.selected?.neuron !== n) return;
    const pole = (p, sign, cls, label) => `
      <div class="pole"><span class="sign ${cls}">${sign}</span><div>
        <p class="stats" style="margin:2px 0 4px">${label} · KL ${p.kl}</p>
        <div class="chips">${p.up.slice(0, 8).map((t) => chip(t, cls)).join("")}</div></div></div>`;
    box.innerHTML = `<h3>Steer lens<small>words that become likelier when steered</small></h3>
      ${pole(s.plus, "+", "up", "dose +1")}${pole(s.minus, "−", "down", "dose −1")}`;
  } catch (err) {
    const box = $("#steerLens");
    if (box) box.innerHTML = `<h3>Steer lens</h3><p class="err">${esc(err.message)}</p>`;
  }
}

function clearNeuron() {
  state.selected = null;
  state.neuron = null;
  selGroup.clear();
  welcome();
  renderRunOutput();
  renderSuggest();
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("button.ref, button.nref");
  if (b) selectNeuron(Number(b.dataset.l), Number(b.dataset.n));
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.selected && !e.target.closest("input, textarea")) clearNeuron();
});

// ------------------------------------------------------------------ steering + runs

function addSteer(layer, neuron, dose) {
  const ex = state.steers.find((s) => s.layer === layer && s.neuron === neuron);
  if (ex) ex.dose = dose;
  else state.steers.push({ layer, neuron, dose });
  renderSteers();
}

function renderSteers() {
  const box = $("#steerList");
  box.innerHTML = state.steers.map((s, i) => `
    <span class="steer ${s.dose < 0 ? "neg" : ""}">
      <button type="button" class="go" data-i="${i}" title="Open this neuron">${nid(s.layer, s.neuron)}</button>
      <input type="number" step="0.1" value="${s.dose}" data-i="${i}" aria-label="Dose for ${nid(s.layer, s.neuron)}">
      <button type="button" class="x" data-i="${i}" aria-label="Remove steer">
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 2l6 6M8 2l-6 6" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
      </button>
    </span>`).join("");
  box.querySelectorAll("input").forEach((inp) => (inp.onchange = () => {
    state.steers[inp.dataset.i].dose = Number(inp.value) || 0;
    renderSteers();
  }));
  box.querySelectorAll(".x").forEach((b) => (b.onclick = () => {
    state.steers.splice(Number(b.dataset.i), 1);
    renderSteers();
  }));
  box.querySelectorAll(".go").forEach((b) => (b.onclick = () => {
    const s = state.steers[b.dataset.i];
    selectNeuron(s.layer, s.neuron);
  }));
}

$("#runForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = $("#promptInput").value.trim();
  if (!prompt) return;
  const btn = $("#runBtn");
  btn.disabled = true;
  btn.textContent = "Running…";
  $("#runProgress").hidden = false;
  try {
    const out = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, steers: state.steers, baseline: $("#compareBox").checked, history: chatHistory() }),
    });
    state.heat = Uint8Array.from(atob(out.heat), (c) => c.charCodeAt(0));
    state.lastRun = { ...out, prompt, steers: state.steers.map((s) => ({ ...s })) };
    state.chat.push({ user: prompt, reply: out.reply, steers: state.lastRun.steers });
    $("#promptInput").value = "";
    $("#newChat").hidden = false;
    applyHeat(state.heat);
    if (state.selected) {
      const { layer, neuron } = state.selected;
      state.neuron = await api(`/api/neuron?layer=${layer}&neuron=${neuron}`);
      renderNeuron(state.neuron);
      loadSteerLens(layer, neuron);
    }
    renderRunOutput();
  } catch (err) {
    $("#runOutput").innerHTML = `<p class="err">Run failed: ${esc(err.message)}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Send";
    $("#promptInput").focus();
    $("#runProgress").hidden = true;
  }
});

function topFromHeat(k = 16) {
  const h = state.heat;
  const N = state.info.neurons;
  const idx = [];
  for (let i = 0; i < h.length; i++) if (h[i] > 120) idx.push(i);
  // strongest neuron per layer; many saturate, so ties go to deeper (more abstract) layers
  const best = {};
  for (const i of idx) {
    const l = Math.floor(i / N);
    if (best[l] === undefined || h[i] > h[best[l]]) best[l] = i;
  }
  return Object.values(best).sort((a, b) => h[b] - h[a] || b - a).slice(0, k).map((i) => ({ layer: Math.floor(i / N), neuron: i % N, rel: (h[i] / 255).toFixed(2) }));
}

function renderRunOutput() {
  const r = state.lastRun;
  if (!r) return;
  const tr = state.neuron?.trace;
  const body = tr
    ? `<p class="label">Reply · shaded by ${nid(state.neuron.layer, state.neuron.neuron)}</p><p class="reply toks">${tokens(tr.tokens, tr.acts, state.neuron.max || Math.max(...tr.acts.map(Math.abs), 1e-6))}</p>`
    : `<p class="label">Reply${r.steers.length ? " · steered " + r.steers.map((s) => `${nid(s.layer, s.neuron)} ${s.dose > 0 ? "+" : ""}${s.dose}`).join(", ") : ""}</p><p class="reply">${esc(r.reply)}</p>`;
  const base = r.baseline !== undefined ? `<p class="label">Unsteered</p><p class="reply" style="color:var(--dim)">${esc(r.baseline)}</p>` : "";
  const top = topFromHeat();
  const steerTag = (st) => (st.length ? `<p class="who-steer">steered: ${st.map((s) => `${nid(s.layer, s.neuron)} ${s.dose > 0 ? "+" : ""}${s.dose}`).join(", ")}</p>` : "");
  const older = state.chat.slice(0, -1).map((t) => `<div class="turn old"><p class="you">${esc(t.user)}</p>${steerTag(t.steers)}<p class="reply">${esc(t.reply)}</p></div>`).join("");
  $("#runOutput").innerHTML = `${older}<div class="turn"><p class="you">${esc(r.prompt || "")}</p>${body}${base}</div>`;
  $("#runOutput").insertAdjacentHTML("beforeend", `
    <p class="label">Firing hardest on this reply<small style="font-weight:400"> · relative to each neuron's max</small></p>
    <div class="toplist">${top.map(refBtn).join("") || '<span class="empty">Nothing above half its usual max.</span>'}</div>`);
  $("#runOutput").scrollTop = $("#runOutput").scrollHeight;
}

// ------------------------------------------------------------------ chat memory + saved characters

function chatHistory() {
  return state.chat.slice(-3).flatMap((t) => [{ role: "user", content: t.user }, { role: "assistant", content: t.reply }]);
}

$("#promptInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("#runForm").requestSubmit();
  }
});

$("#newChat").onclick = () => {
  state.chat = [];
  state.lastRun = null;
  $("#newChat").hidden = true;
  $("#runOutput").innerHTML = '<p class="empty">New chat. Your steers are still on.</p>';
  $("#promptInput").focus();
};

function renderPresets(selected = $("#charSelect").value) {
  const sel = $("#charSelect");
  sel.innerHTML = '<option value="">Saved characters…</option>' + state.presets.map((p) =>
    `<option value="${esc(p.name)}">${esc(p.name)} · ${p.steers.map((s) => nid(s.layer, s.neuron)).join(", ")}</option>`).join("");
  sel.value = state.presets.some((p) => p.name === selected) ? selected : "";
  $("#charDelete").hidden = !sel.value;
}

async function loadPresets() {
  state.presets = await api("/api/presets");
  renderPresets();
}

$("#charSelect").onchange = () => {
  const p = state.presets.find((x) => x.name === $("#charSelect").value);
  $("#charDelete").hidden = !p;
  if (!p) return;
  state.steers = p.steers.map((s) => ({ ...s }));
  $("#charName").value = p.name;
  renderSteers();
};

$("#charSave").onclick = async () => {
  const name = $("#charName").value.trim();
  if (!name) { $("#charName").focus(); return; }
  if (!state.steers.length) { $("#charName").placeholder = "Add a steer first"; $("#charName").value = ""; return; }
  state.presets = await api("/api/presets", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, steers: state.steers }),
  });
  renderPresets(name);
};

$("#charDelete").onclick = async () => {
  const name = $("#charSelect").value;
  if (!name) return;
  state.presets = await api("/api/presets", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, steers: null }),
  });
  $("#charName").value = "";
  renderPresets("");
};

// ------------------------------------------------------------------ search + jump

let searchTimer;
$("#searchInput").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => runSearch(e.target.value.trim()), 250);
});
$("#searchInput").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { e.target.value = ""; runSearch(""); }
});
document.querySelectorAll("[data-by]").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("[data-by]").forEach((x) => x.classList.toggle("on", x === b));
  state.searchBy = b.dataset.by;
  runSearch($("#searchInput").value.trim());
}));

async function runSearch(q) {
  const box = $("#searchResults");
  if (!q) { box.hidden = true; applyMarks([]); return; }
  try {
    const res = await api(`/api/search?q=${encodeURIComponent(q)}&by=${state.searchBy}`);
    applyMarks(res);
    box.innerHTML = res.length
      ? res.map((r) => `<button data-l="${r.layer}" data-n="${r.neuron}"><span class="chip">${nid(r.layer, r.neuron)}</span><span style="color:var(--dim)">${esc(r.hint ?? "")}</span><span style="margin-left:auto;color:var(--faint)">${r.score}</span></button>`).join("")
      : `<p class="none">No neuron's ${state.searchBy === "lens" ? "logit lens" : "top-firing words"} match “${esc(q)}”.</p>`;
    box.hidden = false;
    box.querySelectorAll("button").forEach((b) => (b.onclick = () => {
      box.hidden = true;
      selectNeuron(Number(b.dataset.l), Number(b.dataset.n));
    }));
  } catch (err) {
    box.innerHTML = `<p class="none err">${esc(err.message)}</p>`;
    box.hidden = false;
  }
}
document.addEventListener("pointerdown", (e) => {
  if (!e.target.closest(".search")) $("#searchResults").hidden = true;
});

$("#jumpForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const m = $("#jumpInput").value.match(/L?\s*(\d+)\D+(\d+)/i);
  $("#jumpInput").classList.toggle("bad", !m);
  if (m) selectNeuron(Number(m[1]), Number(m[2]));
});

$("#layerSelect").addEventListener("change", (e) => {
  const v = e.target.value;
  if (v === "all") {
    focusLayer(-1);
    flyTo(new THREE.Vector3(0, 0, 0), new THREE.Vector3(8.5, 5.5, 14.5));
  } else {
    const l = Number(v);
    focusLayer(l);
    flyTo(new THREE.Vector3(layerX(l), 0, 0), new THREE.Vector3(3.8, 1.3, 3.3));
  }
});

document.querySelectorAll("[data-layout]").forEach((b) => (b.onclick = async () => {
  if (b.classList.contains("on")) return;
  document.querySelectorAll("[data-layout]").forEach((x) => x.classList.toggle("on", x === b));
  state.layout = b.dataset.layout;
  setLayout(await fetchLayout(state.layout));
}));

async function fetchLayout(mode) {
  const res = await fetch(`/api/layout?mode=${mode}`);
  return new Float32Array(await res.arrayBuffer());
}

// ------------------------------------------------------------------ tabs

document.querySelectorAll("[data-tab]").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("[data-tab]").forEach((x) => {
    x.classList.toggle("on", x === b);
    x.setAttribute("aria-selected", x === b);
  });
  $("#tab-prompt").hidden = b.dataset.tab !== "prompt";
  $("#tab-claude").hidden = b.dataset.tab !== "claude";
  if (b.dataset.tab === "claude") $("#askInput").focus();
}));

// ------------------------------------------------------------------ Claude

const chatLog = $("#chatLog");

function md(text) {
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\bL(\d{1,2})[\s:]*N(\d{1,5})\b/g, (m, l, n) => `<button class="nref" data-l="${l}" data-n="${n}">L${l} N${n}</button>`);
  const out = [];
  let list = null;
  for (const line of text.split("\n")) {
    const li = line.match(/^\s*[-*]\s+(.*)/);
    if (li) { (list ||= []).push(`<li>${inline(li[1])}</li>`); continue; }
    if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
    if (line.trim()) out.push(`<p>${inline(line.replace(/^#+\s*/, ""))}</p>`);
  }
  if (list) out.push(`<ul>${list.join("")}</ul>`);
  return out.join("");
}

function toolLabel(cmd) {
  const path = (cmd.match(/\/api\/(\w+)/) || [])[1];
  const params = {};
  for (const m of cmd.matchAll(/[?&]([a-z_]+)=([^&\s"']+)/g)) params[m[1]] = decodeURIComponent(m[2]);
  for (const m of cmd.matchAll(/--data-urlencode\s+(?:"([a-z_]+)=([^"]*)"|'([a-z_]+)=([^']*)'|([a-z_]+)=(\S*))/g)) {
    params[m[1] || m[3] || m[5]] = m[2] ?? m[4] ?? m[6];
  }
  const n = params.layer !== undefined ? nid(params.layer, params.neuron) : "";
  switch (path) {
    case "search": return { text: `search ${params.by || "lens"} “${params.q ?? ""}”` };
    case "neuron": return { text: `neuron ${n}`, open: params.layer !== undefined && [Number(params.layer), Number(params.neuron)] };
    case "steerlens": return { text: `steer lens ${n}` };
    case "top_firing": return { text: `top firing “${(params.prompt || "").slice(0, 40)}”` };
    case "compare": return { text: `compare ${params.steer || ""}` };
    default: return { text: cmd.length > 70 ? cmd.slice(0, 70) + "…" : cmd };
  }
}

function addMsg(cls, html) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.innerHTML = html;
  chatLog.querySelector(".empty")?.remove();
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return el;
}

let chatAbort = null;
async function ask(message) {
  if (state.chatBusy || !message.trim()) return;
  state.chatBusy = true;
  addMsg("user", esc(message));
  const send = $("#askBtn");
  send.classList.add("stop");
  send.setAttribute("aria-label", "Stop");
  send.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="2" width="8" height="8" rx="1.5" fill="currentColor"/></svg>';
  let tools = null;
  const thinking = addMsg("bot", '<p class="thinking">thinking…</p>');
  chatAbort = new AbortController();
  try {
    const res = await fetch("/api/claude", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session: state.session }),
      signal: chatAbort.signal,
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, cut).replace(/^data: /, "");
        buf = buf.slice(cut + 2);
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.kind === "session") state.session = ev.id;
        else if (ev.kind === "tool") {
          if (!tools) { tools = addMsg("bot tools", ""); chatLog.appendChild(thinking); }
          const t = toolLabel(ev.command);
          const div = document.createElement("div");
          div.className = "toolline";
          div.textContent = t.text;
          div.title = ev.command;
          tools.appendChild(div);
          if (t.open) selectNeuron(t.open[0], t.open[1]);
        } else if (ev.kind === "text") {
          tools = null;
          const el = addMsg("bot", md(ev.text));
          chatLog.insertBefore(el, thinking);
          chatLog.appendChild(thinking);
        } else if (ev.kind === "done") {
          if (ev.cost) $("#chatCost").textContent = `last answer ≈ $${ev.cost.toFixed(3)} of plan usage`;
        } else if (ev.kind === "error") {
          addMsg("bot", `<p class="err">Claude stopped: ${esc(ev.text)}</p>`);
        }
        chatLog.scrollTop = chatLog.scrollHeight;
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") addMsg("bot", `<p class="err">Couldn't reach Claude: ${esc(err.message)}</p>`);
    else addMsg("bot", '<p class="stats">stopped</p>');
  } finally {
    thinking.remove();
    state.chatBusy = false;
    chatAbort = null;
    send.classList.remove("stop");
    send.setAttribute("aria-label", "Send");
    send.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 13V3M3.5 7.5L8 3l4.5 4.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  }
}

$("#askForm").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.chatBusy) {
    fetch("/api/claude/stop", { method: "POST" });
    chatAbort?.abort();
    return;
  }
  const v = $("#askInput").value;
  $("#askInput").value = "";
  autoGrow();
  ask(v);
});
const autoGrow = () => {
  const t = $("#askInput");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 140) + "px";
};
$("#askInput").addEventListener("input", autoGrow);
$("#askInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#askForm").requestSubmit();
  }
});
$("#newChat").onclick = () => {
  state.session = null;
  chatLog.innerHTML = '<p class="empty">New chat. Claude starts fresh.</p>';
  $("#chatCost").textContent = "";
};

function renderSuggest() {
  const s = state.selected;
  const items = s
    ? [`explain ${nid(s.layer, s.neuron)}`, `what does its web do?`, `steer it and compare`]
    : ["find a neuron that makes replies angry", "find a pirate-speak neuron", "what fires on “I failed my test”?"];
  $("#suggest").innerHTML = items.map((t) => `<button type="button">${esc(t)}</button>`).join("");
  $("#suggest").querySelectorAll("button").forEach((b) => (b.onclick = () => {
    const t = b.textContent;
    ask(s && !t.includes("L") ? `${t} (${nid(s.layer, s.neuron)})` : t);
  }));
}

// ------------------------------------------------------------------ boot

async function boot() {
  state.info = await api("/api/info");
  const sel = $("#layerSelect");
  state.info.types.forEach((t, l) => sel.insertAdjacentHTML("beforeend", `<option value="${l}">L${l} · ${t === "conv" ? "conv" : "attention"}</option>`));
  if (!state.info.indexed) document.querySelector('[data-layout="writes"]').disabled = true;
  buildScene(await fetchLayout("index"));
  camera.position.set(8.5, 5.5, 14.5);
  controls.target.set(0, 0, 0);
  resize();
  requestAnimationFrame(loop);
  buildLiveBar();
  loadPresets();
  connectLive();
  welcome();
  renderSuggest();
  $("#boot").classList.add("done");
}

boot().catch((err) => {
  $("#boot").innerHTML = `<p class="err">Couldn't start: ${esc(err.message)}. Is <code>python explorer.py</code> running?</p>`;
});
