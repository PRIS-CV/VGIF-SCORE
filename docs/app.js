const models = [
  { model: "Kling-V3", group: "commercial", entity: 71.07, attribute: 57.36, location: 75.14, action: 20.73, state: 12.60, causal: 4.21, obj: 42.18, cin: 50.58, pur: 74.35, mot: 49.06, phy: 37.58, sub: 52.89, vgif: 46.30 },
  { model: "Seedance-2.0", group: "commercial", entity: 70.18, attribute: 54.01, location: 75.00, action: 19.34, state: 11.65, causal: 2.98, obj: 40.96, cin: 55.67, pur: 71.52, mot: 55.76, phy: 43.41, sub: 56.59, vgif: 47.59 },
  { model: "Wan-2.7", group: "commercial", entity: 76.46, attribute: 70.91, location: 82.73, action: 22.12, state: 13.21, causal: 3.46, obj: 46.29, cin: 45.88, pur: 64.89, mot: 42.35, phy: 33.21, sub: 46.58, vgif: 46.44 },
  { model: "ViduQ3-Turbo", group: "commercial", entity: 74.35, attribute: 66.07, location: 78.73, action: 35.57, state: 10.95, causal: 3.68, obj: 44.76, cin: 44.93, pur: 67.62, mot: 46.19, phy: 34.35, sub: 48.27, vgif: 45.35 },
  { model: "PixVerse-V6", group: "commercial", entity: 75.41, attribute: 66.97, location: 81.22, action: 25.91, state: 15.50, causal: 4.21, obj: 46.73, cin: 48.79, pur: 66.73, mot: 44.84, phy: 35.34, sub: 48.93, vgif: 47.18 },
  { model: "LTX-2.0", group: "open", entity: 57.38, attribute: 45.65, location: 61.88, action: 8.95, state: 4.34, causal: 0.53, obj: 31.06, cin: 40.00, pur: 57.49, mot: 44.48, phy: 36.59, sub: 44.64, vgif: 36.50 },
  { model: "Wan2.2-A14B", group: "open", entity: 69.33, attribute: 60.06, location: 74.31, action: 14.37, state: 8.88, causal: 1.84, obj: 39.48, cin: 39.10, pur: 56.77, mot: 37.31, phy: 28.61, sub: 40.45, vgif: 39.96 },
  { model: "HyVideo-1.5", group: "open", entity: 59.79, attribute: 50.15, location: 65.75, action: 12.37, state: 6.20, causal: 0.79, obj: 33.76, cin: 44.30, pur: 60.36, mot: 47.53, phy: 37.40, sub: 47.40, vgif: 39.18 },
  { model: "LongCat-Video", group: "open", entity: 64.42, attribute: 55.26, location: 66.02, action: 11.07, state: 5.79, causal: 0.53, obj: 35.27, cin: 39.01, pur: 52.74, mot: 42.06, phy: 32.02, sub: 41.46, vgif: 38.47 },
  { model: "Mochi-1", group: "open", entity: 56.03, attribute: 50.45, location: 65.19, action: 9.66, state: 5.37, causal: 0.26, obj: 31.76, cin: 35.53, pur: 52.65, mot: 33.15, phy: 28.07, sub: 37.35, vgif: 33.28 },
  { model: "CogVideoX-1.5", group: "open", entity: 52.75, attribute: 51.65, location: 63.26, action: 9.54, state: 4.13, causal: 0.00, obj: 30.37, cin: 28.43, pur: 45.74, mot: 31.21, phy: 25.47, sub: 32.71, vgif: 31.54 },
  { model: "MAGI-1", group: "open", entity: 44.74, attribute: 36.34, location: 59.94, action: 3.30, state: 2.27, causal: 0.00, obj: 24.63, cin: 24.04, pur: 41.97, mot: 26.28, phy: 23.14, sub: 28.86, vgif: 26.74 },
  { model: "URSA", group: "open", entity: 51.69, attribute: 49.25, location: 60.22, action: 5.18, state: 2.89, causal: 0.00, obj: 28.33, cin: 29.60, pur: 41.08, mot: 34.80, phy: 27.09, sub: 33.14, vgif: 30.92 },
  { model: "InfinityStar", group: "open", entity: 62.68, attribute: 59.46, location: 72.10, action: 11.07, state: 2.89, causal: 0.00, obj: 35.33, cin: 38.61, pur: 52.24, mot: 44.38, phy: 31.24, sub: 35.52, vgif: 35.43 }
];

const metricLabels = {
  vgif: "VGIF-Score",
  obj: "Objective score",
  sub: "Subjective score",
  causal: "Causal accuracy"
};

const numericKeys = ["entity", "attribute", "location", "action", "state", "causal", "obj", "cin", "pur", "mot", "phy", "sub", "vgif"];
let activeGroup = "all";
let activeMetric = "vgif";
let sortKey = "vgif";
let sortDirection = "desc";

const barChart = document.querySelector("#bar-chart");
const chartTitle = document.querySelector("#chart-title");
const tableBody = document.querySelector("#leaderboard-body");

function filteredModels() {
  return models.filter((item) => activeGroup === "all" || item.group === activeGroup);
}

function sortedModels(key = sortKey, direction = sortDirection) {
  return [...filteredModels()].sort((a, b) => {
    const first = a[key];
    const second = b[key];
    if (typeof first === "string") {
      return direction === "asc" ? first.localeCompare(second) : second.localeCompare(first);
    }
    return direction === "asc" ? first - second : second - first;
  });
}

function renderChart() {
  const chartModels = [...filteredModels()].sort((a, b) => b[activeMetric] - a[activeMetric]);
  chartTitle.textContent = metricLabels[activeMetric];
  barChart.setAttribute("aria-label", `${metricLabels[activeMetric]} ranking for ${activeGroup === "all" ? "all models" : `${activeGroup} models`}`);
  barChart.innerHTML = chartModels.map((item) => `
    <div class="bar-row ${item.group}">
      <span class="bar-label" title="${item.model}">${item.model}</span>
      <span class="bar-track"><span class="bar-fill" style="width: ${Math.max(item[activeMetric], 0.8)}%"></span></span>
      <span class="bar-value">${item[activeMetric].toFixed(2)}</span>
    </div>
  `).join("");
}

function renderTable() {
  const rows = sortedModels();
  tableBody.innerHTML = rows.map((item, index) => {
    const cells = numericKeys.map((key) => `<td class="${key === activeMetric ? "active-metric" : ""}">${item[key].toFixed(2)}</td>`).join("");
    const groupName = item.group === "commercial" ? "Commercial" : "Open-source";
    return `
      <tr>
        <td><span class="rank-number ${index === 0 ? "top-rank" : ""}">${index + 1}</span></td>
        <td class="model-name">${item.model}</td>
        <td><span class="group-label ${item.group}">${groupName}</span></td>
        ${cells}
      </tr>
    `;
  }).join("");
  updateSortIndicators();
}

function updateSortIndicators() {
  document.querySelectorAll("[data-sort]").forEach((button) => {
    const indicator = button.querySelector("span");
    const selected = button.dataset.sort === sortKey;
    indicator.textContent = selected ? (sortDirection === "desc" ? "\u2193" : "\u2191") : "";
    button.setAttribute("aria-sort", selected ? (sortDirection === "desc" ? "descending" : "ascending") : "none");
  });
}

function renderLeaderboard() {
  renderChart();
  renderTable();
}

document.querySelectorAll("[data-group]").forEach((button) => {
  button.addEventListener("click", () => {
    activeGroup = button.dataset.group;
    document.querySelectorAll("[data-group]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    renderLeaderboard();
  });
});

document.querySelectorAll("[data-metric]").forEach((button) => {
  button.addEventListener("click", () => {
    activeMetric = button.dataset.metric;
    sortKey = activeMetric;
    sortDirection = "desc";
    document.querySelectorAll("[data-metric]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    renderLeaderboard();
  });
});

document.querySelectorAll("[data-sort]").forEach((button) => {
  button.addEventListener("click", () => {
    const selectedKey = button.dataset.sort;
    if (sortKey === selectedKey) {
      sortDirection = sortDirection === "desc" ? "asc" : "desc";
    } else {
      sortKey = selectedKey;
      sortDirection = selectedKey === "model" ? "asc" : "desc";
    }
    renderTable();
  });
});

const copyButton = document.querySelector("#copy-citation");
const copyStatus = document.querySelector("#copy-status");

copyButton.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex").innerText;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = "Copied";
    copyStatus.textContent = "BibTeX copied to clipboard.";
  } catch (error) {
    copyStatus.textContent = "Clipboard access is unavailable. Select the citation text manually.";
  }
  window.setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
    copyStatus.textContent = "";
  }, 2200);
});

renderLeaderboard();

const caseElements = {
  sampleId: document.querySelector("#case-sample-id"),
  domain: document.querySelector("#case-domain"),
  prompt: document.querySelector("#case-prompt"),
  video: document.querySelector("#case-video"),
  videoSource: document.querySelector("#case-video-source"),
  qaScore: document.querySelector("#case-qa-score"),
  objective: document.querySelector("#case-objective-score"),
  subjective: document.querySelector("#case-subjective-score"),
  vgif: document.querySelector("#case-vgif-score"),
  dag: document.querySelector("#case-dag"),
  qaId: document.querySelector("#qa-detail-id"),
  qaStatus: document.querySelector("#qa-detail-status"),
  qaLabel: document.querySelector("#qa-detail-label"),
  qaQuestion: document.querySelector("#qa-detail-question"),
  qaDependency: document.querySelector("#qa-detail-dependency"),
  qaReason: document.querySelector("#qa-detail-reason"),
  rubricScore: document.querySelector("#rubric-model-score"),
  rubricDots: document.querySelector("#rubric-score-dots"),
  rubricGoal: document.querySelector("#rubric-goal"),
  rubricCriteria: document.querySelector("#rubric-criteria"),
  rubricReason: document.querySelector("#rubric-reason"),
  rubricAnchors: document.querySelector("#rubric-anchors")
};

const caseState = {
  data: null,
  model: "Kling-V3",
  node: "q1",
  rubric: "cinematography"
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function currentCaseModel() {
  return caseState.data.models.find((item) => item.model === caseState.model);
}

function qaStatusFor(result) {
  if (result.correct) return { key: "pass", label: "Passed" };
  if (result.dependency_passed) return { key: "fail", label: "Missed" };
  return { key: "blocked", label: "Dependency blocked" };
}

function qaDepths(nodes) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const memo = new Map();
  function depth(nodeId, visiting = new Set()) {
    if (memo.has(nodeId)) return memo.get(nodeId);
    if (visiting.has(nodeId)) return 0;
    const node = nodeMap.get(nodeId);
    if (!node || node.parents.length === 0) return 0;
    const nextVisiting = new Set(visiting).add(nodeId);
    const value = Math.max(...node.parents.map((parent) => depth(parent, nextVisiting))) + 1;
    memo.set(nodeId, value);
    return value;
  }
  nodes.forEach((node) => memo.set(node.id, depth(node.id)));
  return memo;
}

function renderCaseDag() {
  const nodes = caseState.data.qa;
  const model = currentCaseModel();
  const depths = qaDepths(nodes);
  const maxDepth = Math.max(...depths.values());
  const width = 920;
  const height = 340;
  const xPadding = 42;
  const yPadding = 38;
  const byDepth = new Map();

  nodes.forEach((node) => {
    const depth = depths.get(node.id);
    if (!byDepth.has(depth)) byDepth.set(depth, []);
    byDepth.get(depth).push(node);
  });

  const positions = new Map();
  byDepth.forEach((levelNodes, depth) => {
    levelNodes.forEach((node, index) => {
      const x = maxDepth === 0 ? width / 2 : xPadding + (depth / maxDepth) * (width - xPadding * 2);
      const usableHeight = height - yPadding * 2;
      const y = levelNodes.length === 1
        ? height / 2
        : yPadding + (index / (levelNodes.length - 1)) * usableHeight;
      positions.set(node.id, { x, y });
    });
  });

  const edges = nodes.flatMap((node) => node.parents.map((parentId) => {
    const start = positions.get(parentId);
    const end = positions.get(node.id);
    const control = Math.max(34, (end.x - start.x) * 0.48);
    const edgeClass = node.type === "causal" ? "dag-edge causal-edge" : "dag-edge";
    return `<path class="${edgeClass}" d="M ${start.x + 23} ${start.y} C ${start.x + control} ${start.y}, ${end.x - control} ${end.y}, ${end.x - 25} ${end.y}" marker-end="url(#dag-arrow)" />`;
  })).join("");

  const nodeMarkup = nodes.map((node) => {
    const position = positions.get(node.id);
    const status = qaStatusFor(model.qa[node.id]);
    const selected = node.id === caseState.node ? " selected" : "";
    return `
      <g class="dag-node ${escapeHtml(node.type)} ${status.key}${selected}" data-node-id="${escapeHtml(node.id)}" tabindex="0" role="button" aria-label="${escapeHtml(`${node.id}: ${node.label}, ${status.label}`)}" transform="translate(${position.x} ${position.y})">
        <circle class="node-core" r="22"></circle>
        <text dy="4">${escapeHtml(node.id.toUpperCase())}</text>
      </g>
    `;
  }).join("");

  caseElements.dag.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" aria-hidden="true" preserveAspectRatio="xMidYMid meet">
      <defs>
        <marker id="dag-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M 0 0 L 8 4 L 0 8 z" fill="#aab5b0"></path>
        </marker>
      </defs>
      ${edges}
      ${nodeMarkup}
    </svg>
  `;

  caseElements.dag.querySelectorAll("[data-node-id]").forEach((node) => {
    const selectNode = () => {
      caseState.node = node.dataset.nodeId;
      renderCaseDag();
      renderQaDetail();
    };
    node.addEventListener("click", selectNode);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectNode();
      }
    });
  });
}

function renderQaDetail() {
  const node = caseState.data.qa.find((item) => item.id === caseState.node);
  const result = currentCaseModel().qa[node.id];
  const status = qaStatusFor(result);
  const failed = result.failed_dependencies.length
    ? ` Failed prerequisite${result.failed_dependencies.length > 1 ? "s" : ""}: ${result.failed_dependencies.join(", ")}.`
    : "";

  caseElements.qaId.textContent = `${node.id.toUpperCase()} · ${node.type}`;
  caseElements.qaStatus.className = `status-label ${status.key}`;
  caseElements.qaStatus.textContent = status.label;
  caseElements.qaLabel.textContent = node.label;
  caseElements.qaQuestion.textContent = node.question;
  caseElements.qaDependency.textContent = node.dependency;
  caseElements.qaReason.textContent = `${result.reason}${failed}`;
}

function renderRubric() {
  const dimension = caseState.data.rubric.find((item) => item.key === caseState.rubric);
  const result = currentCaseModel().rubric[dimension.result_id];
  caseElements.rubricScore.textContent = `${result.score} / 5`;
  caseElements.rubricDots.setAttribute("aria-label", `${dimension.title}: ${result.score} out of 5`);
  caseElements.rubricDots.innerHTML = Array.from({ length: 5 }, (_, index) => `<span class="${index < result.score ? "filled" : ""}"></span>`).join("");
  caseElements.rubricGoal.textContent = dimension.goal;
  caseElements.rubricCriteria.innerHTML = dimension.criteria.slice(0, 5).map((criterion) => `<li>${escapeHtml(criterion)}</li>`).join("");
  caseElements.rubricReason.textContent = result.reason;
  caseElements.rubricAnchors.innerHTML = ["1", "3", "5"].map((score) => `<div><strong>${score}</strong><span>${escapeHtml(dimension.anchors[score])}</span></div>`).join("");
}

function renderCase() {
  const model = currentCaseModel();
  caseElements.sampleId.textContent = caseState.data.sample_id;
  caseElements.domain.textContent = `${caseState.data.macro_domain} · ${caseState.data.micro_domain}`;
  caseElements.prompt.textContent = caseState.data.prompt;
  caseElements.video.pause();
  caseElements.video.poster = model.poster;
  caseElements.videoSource.src = model.video;
  caseElements.video.load();
  caseElements.video.setAttribute("aria-label", `${model.model} generation for the selected prompt`);
  caseElements.qaScore.textContent = `${model.qa_correct} / ${model.qa_total}`;
  caseElements.objective.textContent = model.objective.toFixed(2);
  caseElements.subjective.textContent = model.subjective.toFixed(2);
  caseElements.vgif.textContent = model.vgif.toFixed(2);
  renderCaseDag();
  renderQaDetail();
  renderRubric();
}

document.querySelectorAll("[data-case-model]").forEach((button) => {
  button.addEventListener("click", () => {
    caseState.model = button.dataset.caseModel;
    const model = currentCaseModel();
    caseState.node = caseState.model === "Kling-V3"
      ? "q1"
      : caseState.data.qa.find((node) => !model.qa[node.id].correct).id;
    document.querySelectorAll("[data-case-model]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    renderCase();
  });
});

document.querySelectorAll("[data-rubric-key]").forEach((button) => {
  button.addEventListener("click", () => {
    caseState.rubric = button.dataset.rubricKey;
    document.querySelectorAll("[data-rubric-key]").forEach((item) => item.setAttribute("aria-selected", String(item === button)));
    renderRubric();
  });
});

const categoryElements = {
  select: document.querySelector("#category-select"),
  levelLabel: document.querySelector("#category-level-label"),
  chartTitle: document.querySelector("#category-chart-title"),
  chart: document.querySelector("#category-bar-chart"),
  leaderModel: document.querySelector("#category-leader-model"),
  leaderScore: document.querySelector("#category-leader-score"),
  source: document.querySelector("#category-source"),
  heatmap: document.querySelector("#category-heatmap"),
  heatmapDescription: document.querySelector("#heatmap-description"),
  count: document.querySelector("#category-count")
};

const categoryState = {
  data: null,
  level: "macro",
  group: "all",
  selected: null
};

function currentCategorySet() {
  return categoryState.data[categoryState.level];
}

function visibleCategoryModels() {
  return currentCategorySet().scores.filter((item) => categoryState.group === "all" || item.group === categoryState.group);
}

function populateCategorySelect() {
  const set = currentCategorySet();
  if (!set.categories.some((item) => item.id === categoryState.selected)) {
    categoryState.selected = set.categories[0].id;
  }

  if (categoryState.level === "micro") {
    const macroNames = categoryState.data.macro.categories;
    categoryElements.select.innerHTML = macroNames.map((macro) => {
      const options = set.categories
        .filter((category) => category.macro === macro.id)
        .map((category) => `<option value="${escapeHtml(category.id)}"${category.id === categoryState.selected ? " selected" : ""}>${escapeHtml(category.name)}</option>`)
        .join("");
      return `<optgroup label="${escapeHtml(macro.name)}">${options}</optgroup>`;
    }).join("");
  } else {
    categoryElements.select.innerHTML = set.categories.map((category) => `<option value="${escapeHtml(category.id)}"${category.id === categoryState.selected ? " selected" : ""}>${escapeHtml(category.name)}</option>`).join("");
  }
}

function heatClass(score) {
  if (!Number.isFinite(score)) return "heat-missing";
  if (score < 30) return "heat-1";
  if (score < 40) return "heat-2";
  if (score < 50) return "heat-3";
  if (score < 60) return "heat-4";
  return "heat-5";
}

function renderCategoryChart() {
  const set = currentCategorySet();
  const category = set.categories.find((item) => item.id === categoryState.selected);
  const ranked = visibleCategoryModels()
    .filter((item) => Number.isFinite(item.values[category.id]))
    .sort((a, b) => b.values[category.id] - a.values[category.id]);

  categoryElements.levelLabel.textContent = categoryState.level === "macro" ? "Macro domain ranking" : "Micro domain ranking";
  categoryElements.chartTitle.textContent = category.short || category.name;
  categoryElements.chart.setAttribute("aria-label", `${category.name} VGIF ranking`);
  categoryElements.chart.innerHTML = ranked.map((item) => `
    <div class="bar-row ${item.group}">
      <span class="bar-label" title="${escapeHtml(item.model)}">${escapeHtml(item.model)}</span>
      <span class="bar-track"><span class="bar-fill" style="width: ${Math.max(item.values[category.id], 0.8)}%"></span></span>
      <span class="bar-value">${item.values[category.id].toFixed(2)}</span>
    </div>
  `).join("");

  const leader = ranked[0];
  categoryElements.leaderModel.textContent = leader ? leader.model : "No evaluated model";
  categoryElements.leaderScore.textContent = leader ? `${leader.values[category.id].toFixed(2)} / 100` : "--";
  categoryElements.source.textContent = categoryState.level === "macro"
    ? "Exact aggregate values from camera-ready Table 3."
    : "Per-sample VGIF scores macro-averaged within this micro domain; coverage is shown in heatmap tooltips.";
}

function renderCategoryHeatmap() {
  const set = currentCategorySet();
  const scoreRows = visibleCategoryModels();
  const categoryHeaders = set.categories.map((category) => `<th scope="col" class="${category.id === categoryState.selected ? "selected-domain" : ""}" title="${escapeHtml(category.name)}">${escapeHtml(category.short || category.name)}</th>`).join("");
  const rows = scoreRows.map((item) => {
    const cells = set.categories.map((category) => {
      const value = item.values[category.id];
      const coverage = item.coverage ? item.coverage[category.id] : null;
      const coverageText = Number.isFinite(coverage) ? `, n=${coverage}` : "";
      const title = Number.isFinite(value) ? `${item.model} · ${category.name}: ${value.toFixed(2)}${coverageText}` : `${item.model} · ${category.name}: unavailable`;
      return `<td class="${heatClass(value)} ${category.id === categoryState.selected ? "selected-domain" : ""}" title="${escapeHtml(title)}">${Number.isFinite(value) ? value.toFixed(1) : "—"}</td>`;
    }).join("");
    return `<tr><th scope="row">${escapeHtml(item.model)}</th>${cells}</tr>`;
  }).join("");

  categoryElements.heatmap.innerHTML = `<thead><tr><th scope="col">Model</th>${categoryHeaders}</tr></thead><tbody>${rows}</tbody>`;
  categoryElements.heatmapDescription.textContent = categoryState.level === "macro"
    ? "Eight macro domains from the camera-ready paper."
    : "Thirty-eight micro domains grouped under the eight benchmark domains.";
  categoryElements.count.textContent = `${set.categories.length} domains`;
}

function renderCategories({ repopulate = false } = {}) {
  if (repopulate) populateCategorySelect();
  renderCategoryChart();
  renderCategoryHeatmap();
}

document.querySelectorAll("[data-category-level]").forEach((button) => {
  button.addEventListener("click", () => {
    categoryState.level = button.dataset.categoryLevel;
    categoryState.selected = null;
    document.querySelectorAll("[data-category-level]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    renderCategories({ repopulate: true });
  });
});

document.querySelectorAll("[data-category-group]").forEach((button) => {
  button.addEventListener("click", () => {
    categoryState.group = button.dataset.categoryGroup;
    document.querySelectorAll("[data-category-group]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    renderCategories();
  });
});

categoryElements.select.addEventListener("change", () => {
  categoryState.selected = categoryElements.select.value;
  renderCategories();
});

Promise.all([
  fetch("data/case_study.json", { cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error(`Case data request failed: ${response.status}`);
    return response.json();
  }),
  fetch("data/category_scores.json", { cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error(`Category data request failed: ${response.status}`);
    return response.json();
  })
]).then(([caseData, categoryData]) => {
  caseState.data = caseData;
  categoryState.data = categoryData;
  categoryState.selected = categoryData.macro.categories[0].id;
  renderCase();
  renderCategories({ repopulate: true });
}).catch((error) => {
  caseElements.prompt.textContent = "The interactive data could not be loaded. Serve the project page over HTTP to enable the visualizations.";
  categoryElements.source.textContent = "Domain data unavailable.";
  console.error(error);
});
