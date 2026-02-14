const BASE = "./viz_export_umap";

const W = () => document.getElementById("plot").clientWidth;
const H = () => document.getElementById("plot").clientHeight;

let allPoints = [];
let nnMap = {};
let selectedId = null;

const typeFilter = document.getElementById("typeFilter");
const labelFilter = document.getElementById("labelFilter");
const statsEl = document.getElementById("stats");
const detailEl = document.getElementById("detail");

const nnChartEl = document.getElementById("nnChart");
const nnSummaryEl = document.getElementById("nnSummary");

const tip = document.getElementById("tip");
const modal = document.getElementById("modal");
const modalImg = document.getElementById("modalImg");
const modalCap = document.getElementById("modalCap");
const modalClose = document.getElementById("modalClose");

/* ---------------- Utils ---------------- */
function applyFilters() {
  const t = typeFilter.value;
  const l = labelFilter.value;

  return allPoints.filter(p => {
    if (t !== "all" && p.type !== t) return false;
    if (l !== "all" && String(p.label) !== l) return false;
    return true;
  });
}

function fmt(n, k = 4) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toFixed(k);
}

function fillColor(d) {
  return d.label === 0 ? "#60a5fa" : "#fb7185";
}

/* ---------------- Modal ---------------- */
function openModal(imgSrc, caption) {
  modalImg.src = imgSrc;
  modalCap.textContent = caption || "";
  modal.classList.add("show");
}
function closeModal() {
  modal.classList.remove("show");
  modalImg.src = "";
  modalCap.textContent = "";
}
modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
modalClose.addEventListener("click", closeModal);

/* ---------------- Tooltip ---------------- */
function positionTip(evt) {
  const plot = document.getElementById("plot");
  const rect = plot.getBoundingClientRect();

  let x = evt.clientX - rect.left;
  let y = evt.clientY - rect.top;

  const pad = 18;
  x = Math.max(pad, Math.min(rect.width - pad, x));
  y = Math.max(pad, Math.min(rect.height - pad, y));

  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function showTip(evt, d) {
  const lines = [
    `<div class="trow"><span>${d.type}</span><span>label ${d.label}</span></div>`,
    `<div class="tmuted">id: ${d.id}</div>`,
    `<div class="tmuted">sharpness: ${fmt(d.sharpness, 4)}</div>`,
  ];
  if (d.type === "synthetic" && d.nn_real_dist != null) {
    lines.push(`<div class="tmuted">nn real dist: ${fmt(d.nn_real_dist, 4)}</div>`);
  }
  tip.innerHTML = lines.join("");
  tip.classList.add("show");
  positionTip(evt);
}
function hideTip() {
  tip.classList.remove("show");
}

/* ---------------- Detail ---------------- */
function renderDetail(p) {
  const isSyn = p.type === "synthetic";
  const nn = isSyn ? (nnMap[p.id] || []) : [];

  const nnImgs = nn.slice(0, 8).map((id, idx) => {
    const q = allPoints.find(x => x.id === id);
    if (!q) return "";
    return `
      <div class="thumbItem">
        <img class="thumb" src="${BASE}/${q.thumb}" alt="${q.id}" />
        <div class="cap">#${idx + 1} ${q.id}</div>
      </div>
    `;
  }).join("");

  detailEl.innerHTML = `
    <div class="card">
      <div class="row">
        <span class="pill strong">${p.type}</span>
        <span class="pill">label ${p.label}</span>
        <span class="pill">id: ${p.id}</span>
      </div>

      <div class="row muted">
        <div>Sharpness: <b>${fmt(p.sharpness, 4)}</b></div>
        ${isSyn && p.nn_real_dist != null ? `<div>Nearest real dist: <b>${fmt(p.nn_real_dist, 4)}</b></div>` : ""}
      </div>

      <div class="row" style="align-items:flex-start;">
        <img class="hero" id="heroImg" src="${BASE}/${p.thumb}" alt="${p.id}" />
        <div class="muted" style="max-width: 260px;">
          Kliknite na sliku za uvećanje.
        </div>
      </div>

      ${isSyn ? `
        <div class="hr"></div>
        <div class="muted">Nearest real (top ${Math.min(8, nn.length)})</div>
        <div class="thumbgrid">${nnImgs}</div>
      ` : `
        <div class="hr"></div>
        <div class="muted">Nearest-real prikaz je relevantan samo za synthetic točke.</div>
      `}
    </div>
  `;

  const hero = document.getElementById("heroImg");
  hero.addEventListener("click", () =>
    openModal(`${BASE}/${p.thumb}`, `${p.type} • label ${p.label} • ${p.id}`)
  );
}

/* ---------------- Histogram ---------------- */
function renderNnHistogram(points) {
  const vals = points
    .filter(p => p.type === "synthetic")
    .map(p => p.nn_real_dist)
    .filter(v => v != null && !Number.isNaN(v));

  nnChartEl.innerHTML = "";

  if (!vals.length) {
    nnSummaryEl.textContent = "n=0";
    nnChartEl.innerHTML = `<div class="muted">Nema synthetic točaka ili nn_real_dist nije dostupan.</div>`;
    return;
  }

  const sorted = [...vals].sort((a, b) => a - b);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const median = d3.quantile(sorted, 0.5);
  const min = sorted[0];

  nnSummaryEl.textContent = `n=${vals.length} • mean=${mean.toFixed(2)} • med=${median.toFixed(2)} • min=${min.toFixed(2)}`;

  const width = nnChartEl.clientWidth || 320;
  const height = 160;
  const margin = { top: 12, right: 10, bottom: 26, left: 38 };

  const svg = d3.select(nnChartEl).append("svg")
    .attr("width", width)
    .attr("height", height);

  const x = d3.scaleLinear()
    .domain(d3.extent(vals)).nice()
    .range([margin.left, width - margin.right]);

  const bins = d3.bin()
    .domain(x.domain())
    .thresholds(18)(vals);

  const y = d3.scaleLinear()
    .domain([0, d3.max(bins, d => d.length) || 1]).nice()
    .range([height - margin.bottom, margin.top]);

  svg.append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(5));

  svg.append("g")
    .attr("class", "axis")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(4));

  svg.append("g")
    .selectAll("rect")
    .data(bins)
    .join("rect")
    .attr("class", "bar")
    .attr("x", d => x(d.x0) + 1)
    .attr("y", d => y(d.length))
    .attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 2))
    .attr("height", d => (height - margin.bottom) - y(d.length))
    .append("title")
    .text(d => `${d.x0.toFixed(2)} – ${d.x1.toFixed(2)}: ${d.length}`);

  const y1 = margin.top;
  const y2 = height - margin.bottom;

  const meanX = x(mean);
  const medX = x(median);
  const closePx = Math.abs(meanX - medX) < 18;

  function addMarker(value, cls, label, idx) {
    const px = x(value);

    svg.append("line")
      .attr("x1", px).attr("x2", px)
      .attr("y1", y1).attr("y2", y2)
      .attr("class", cls);

    const dx = closePx ? (idx === 0 ? -18 : 8) : 8;
    const dy = closePx ? (idx === 0 ? 10 : 28) : 10;

    const g = svg.append("g")
      .attr("transform", `translate(${px + dx},${y1 + dy})`)
      .attr("class", "hTag");

    g.append("rect")
      .attr("x", -2).attr("y", -12)
      .attr("rx", 6).attr("ry", 6)
      .attr("width", 58).attr("height", 18)
      .attr("class", "hTagBg");

    g.append("text")
      .attr("class", "hTagText")
      .attr("x", 6).attr("y", 1)
      .text(label);
  }

  addMarker(mean, "hmark mean", "mean", 0);
  addMarker(median, "hmark median", "median", 1);
}

/* ---------------- Scatter ---------------- */
function draw() {
  const container = document.getElementById("plot");
  container.innerHTML = "";

  const width = W();
  const height = H();

  const svg = d3.select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const pts = applyFilters();

  // histogram update
  renderNnHistogram(pts);

  // header stats
  const syn = pts.filter(p => p.type === "synthetic");
  const nnVals = syn.map(p => p.nn_real_dist).filter(v => v != null && !Number.isNaN(v));
  const nnMean = nnVals.length ? (nnVals.reduce((a, b) => a + b, 0) / nnVals.length) : null;
  statsEl.textContent = `points: ${pts.length}/${allPoints.length} • synthetic: ${syn.length} • nn mean: ${nnMean != null ? nnMean.toFixed(3) : "—"}`;

  const M = { left: 56, right: 18, top: 18, bottom: 48 };

  const x = d3.scaleLinear()
    .domain(d3.extent(allPoints, d => d.x)).nice()
    .range([M.left, width - M.right]);

  const y = d3.scaleLinear()
    .domain(d3.extent(allPoints, d => d.y)).nice()
    .range([height - M.bottom, M.top]);

  // grid
  const gridX = d3.axisBottom(x).ticks(7).tickSize(-(height - M.top - M.bottom)).tickFormat("");
  const gridY = d3.axisLeft(y).ticks(7).tickSize(-(width - M.left - M.right)).tickFormat("");

  svg.append("g")
    .attr("class", "grid")
    .attr("transform", `translate(0,${height - M.bottom})`)
    .call(gridX);

  svg.append("g")
    .attr("class", "grid")
    .attr("transform", `translate(${M.left},0)`)
    .call(gridY);

  // axes
  svg.append("g")
    .attr("class", "axis axis-x")
    .attr("transform", `translate(0,${height - M.bottom})`)
    .call(d3.axisBottom(x).ticks(7));

  svg.append("g")
    .attr("class", "axis axis-y")
    .attr("transform", `translate(${M.left},0)`)
    .call(d3.axisLeft(y).ticks(7));

  // zoom group for points
  const gMain = svg.append("g").attr("class", "main");
  const pointsG = gMain.append("g");

  // Real: filled, Synthetic: ring (slightly smaller)
  const rReal = 3.6;
  const rSyn = 3.3;

  const baseOpacity = 0.90;

  const marks = pointsG.selectAll("circle")
    .data(pts, d => d.id)
    .join("circle")
    .attr("cx", d => x(d.x))
    .attr("cy", d => y(d.y))
    .attr("r", d => d.type === "synthetic" ? rSyn : rReal)
    .attr("opacity", baseOpacity)

    // REAL: filled by label
    // SYN: transparent fill, stroke by label
    .attr("fill", d => d.type === "synthetic" ? "transparent" : fillColor(d))
    .attr("stroke", d => d.type === "synthetic" ? fillColor(d) : "rgba(0,0,0,0.35)")
    .attr("stroke-width", d => d.type === "synthetic" ? 1.6 : 0.7)

    .style("cursor", "pointer")
    .on("mouseenter", (evt, d) => showTip(evt, d))
    .on("mousemove", (evt) => { if (tip.classList.contains("show")) positionTip(evt); })
    .on("mouseleave", hideTip)
    .on("click", (evt, d) => {
      if (selectedId === d.id) {
        selectedId = null;
        detailEl.innerHTML = "";
        marks.classed("selected", false).classed("dimmed", false);
        marks.attr("stroke-width", q => q.type === "synthetic" ? 1.6 : 0.7);
        return;
      }

      selectedId = d.id;
      renderDetail(d);

      marks
        .classed("selected", q => q.id === selectedId)
        .classed("dimmed", q => selectedId && q.id !== selectedId);

      marks.attr("stroke-width", q =>
        q.id === selectedId ? 3.0 : (q.type === "synthetic" ? 1.6 : 0.7)
      );
    });

  if (selectedId) {
    marks
      .classed("selected", q => q.id === selectedId)
      .classed("dimmed", q => selectedId && q.id !== selectedId)
      .attr("stroke-width", q =>
        q.id === selectedId ? 3.0 : (q.type === "synthetic" ? 1.6 : 0.7)
      );
  }

  const zoom = d3.zoom()
    .scaleExtent([0.7, 10])
    .on("zoom", (event) => {
      gMain.attr("transform", event.transform);
    });

  svg.call(zoom);
}

/* ---------------- Boot ---------------- */
async function main() {
  allPoints = await d3.json(`${BASE}/points.json`);
  nnMap = await d3.json(`${BASE}/nn_map.json`);

  typeFilter.addEventListener("change", () => { selectedId = null; detailEl.innerHTML = ""; draw(); });
  labelFilter.addEventListener("change", () => { selectedId = null; detailEl.innerHTML = ""; draw(); });
  window.addEventListener("resize", draw);

  draw();
}

main();
