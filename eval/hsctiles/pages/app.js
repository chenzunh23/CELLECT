let state = null;
let page = 0;
let busy = false;
let optionsCache = null;
let selectedDataset = 'hsc_raw';
let showDetect = false;
let viewInputScaling = false;
let viewShape = true;
let viewCenter = false;
let viewInvert = false;
let smoothEnabled = false;
let smoothMode = 'gaussian';
let smoothSigma = 1.0;
let smoothRadius = 2;
let pendingSearchPage = null;

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const text = await res.text();
  if (!res.ok) {
    try {
      const payload = JSON.parse(text);
      throw new Error(payload.error || text || res.statusText);
    } catch (_err) {
      throw new Error(text || res.statusText);
    }
  }
  return text ? JSON.parse(text) : {};
}

function setStatus(text, isError=false) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = isError ? 'status error' : 'status';
}

function selectedValues(select) {
  return Array.from(select.selectedOptions).map(opt => opt.value);
}

function fillSelect(select, values, defaults) {
  const defaultSet = new Set(defaults || []);
  select.innerHTML = '';
  for (const value of values || []) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = value;
    opt.selected = defaultSet.has(value);
    select.appendChild(opt);
  }
}

function activeDatasetOptions() {
  return optionsCache && optionsCache.by_dataset ? optionsCache.by_dataset[selectedDataset] : null;
}

function renderDatasetCards() {
  const container = document.getElementById('datasetCards');
  container.innerHTML = '';
  for (const ds of optionsCache.datasets || []) {
    const btn = document.createElement('button');
    btn.className = ds.id === selectedDataset ? 'datasetCard selected' : 'datasetCard';
    btn.disabled = !ds.enabled;
    const status = ds.enabled ? 'available' : (ds.reason || 'placeholder');
    btn.innerHTML = `<strong>${ds.label}</strong><span>${status}</span>`;
    btn.onclick = () => selectDataset(ds.id);
    container.appendChild(btn);
  }
}

function selectDataset(datasetId) {
  selectedDataset = datasetId;
  renderDatasetCards();
  const cfg = activeDatasetOptions();
  if (!cfg) return;
  document.getElementById('tractLabel').textContent = datasetId === 'ztf' ? 'Field' : 'Tract';
  document.getElementById('tractInput').value = cfg.tract || 'default';
  fillSelect(document.getElementById('patchInput'), cfg.patches || [], cfg.default_patches || []);
  fillSelect(document.getElementById('bandInput'), cfg.bands || [], cfg.default_bands || []);
  document.getElementById('nTilesInput').placeholder = String(cfg.default_n_tiles || 4);
  document.getElementById('framesPerTileInput').placeholder = String(cfg.default_frames_per_tile || 1);
  document.getElementById('tilesPerPageInput').placeholder = String(cfg.default_tiles_per_page || 2);
  document.getElementById('startButton').disabled = !cfg.enabled;
  if (!cfg.enabled) setStatus(`${cfg.label} is a placeholder in this browser.`, true);
  else setStatus('');
}

async function loadOptions() {
  optionsCache = await fetchJson('/api/options');
  selectedDataset = optionsCache.dataset || 'hsc_raw';
  renderDatasetCards();
  selectDataset(selectedDataset);
}

function imageUrl(c) {
  const params = [];
  if (viewInputScaling) params.push('input=1');
  if (smoothEnabled) {
    params.push(`smooth_mode=${encodeURIComponent(smoothMode)}`);
    params.push(`smooth_sigma=${encodeURIComponent(String(smoothSigma))}`);
    params.push(`smooth_radius=${encodeURIComponent(String(smoothRadius))}`);
  }
  if (viewInvert) params.push('invert=1');
  if (showDetect) {
    params.push('detect=1');
    params.push(`shape=${viewShape ? 1 : 0}`);
    params.push(`center=${viewCenter ? 1 : 0}`);
    if (viewInputScaling) params.push('input_shape=1');
  }
  return params.length ? `/image/${c.token}.png?${params.join('&')}` : `/image/${c.token}.png`;
}

function updateViewMenu() {
  const states = {inputScaling: viewInputScaling, shape: viewShape, center: viewCenter, invert: viewInvert, smooth: smoothEnabled};
  for (const item of document.querySelectorAll('.viewItem')) {
    const key = item.dataset.view;
    item.querySelector('.viewCheck').textContent = states[key] ? '✓' : '';
  }
}

function resetViewDefaults() {
  viewInputScaling = false;
  viewShape = true;
  viewCenter = false;
  viewInvert = false;
  smoothEnabled = false;
  updateViewMenu();
}

async function toggleViewItem(key) {
  if (!state || !state.started) return;
  if (key === 'inputScaling') viewInputScaling = !viewInputScaling;
  if (key === 'shape') viewShape = !viewShape;
  if (key === 'center') viewCenter = !viewCenter;
  if (key === 'invert') viewInvert = !viewInvert;
  if (key === 'smooth') {
    openSmooth();
    return;
  }
  updateViewMenu();
  setStatus(viewStatusText());
  await loadPage(page, true);
}

function viewStatusText() {
  const smoothText = smoothEnabled
    ? `${smoothMode} ${smoothMode === 'gaussian' ? `sigma=${smoothSigma}, r=${Math.ceil(2 * smoothSigma)}` : `r=${smoothRadius}`}`
    : 'off';
  return `View: input scaling=${viewInputScaling ? 'on' : 'off'}, shape=${viewShape ? 'on' : 'off'}, center=${viewCenter ? 'on' : 'off'}, invert=${viewInvert ? 'on' : 'off'}, smooth=${smoothText}.`;
}

function candidateCell(c) {
  const cell = document.createElement('div');
  cell.className = c.selected ? 'cell selected' : 'cell';
  const img = document.createElement('img');
  img.loading = 'lazy';
  img.src = imageUrl(c);
  img.alt = c.token;
  const label = document.createElement('label');
  label.className = 'label';
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = c.selected;
  box.addEventListener('change', async () => {
    box.disabled = true;
    try {
      await fetchJson('/api/select', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({token: c.token, selected: box.checked})
      });
      c.selected = box.checked;
      cell.className = c.selected ? 'cell selected' : 'cell';
      await refreshState(false);
    } catch (err) {
      box.checked = !box.checked;
      setStatus(String(err), true);
    } finally {
      box.disabled = false;
    }
  });
  const meta = document.createElement('div');
  meta.className = 'meta';
  const visit = c.visit == null ? '' : ` group=${c.visit}`;
  const weight = Number.isFinite(c.weight) ? c.weight.toPrecision(4) : '';
  const scale = Number.isFinite(c.scale) ? c.scale.toPrecision(4) : '';
  const det = c.detected ? ` detections=${c.n_detections}` : '';
  meta.innerHTML = `<strong>${c.token} ${c.band}${visit}${det}</strong><span>${c.dataset || ''} ${c.patch} ${c.tile_id} frame ${c.frame_rank + 1}/${c.tile_length}</span><span>local X=[${c.x0},${c.x1}) Y=[${c.y0},${c.y1})</span><span>weight=${weight} scale=${scale}</span>`;
  label.append(box, meta);
  cell.append(img, label);
  return cell;
}

function showMenu() {
  document.getElementById('menu').classList.remove('hidden');
  document.getElementById('browserHeader').classList.add('hidden');
  document.getElementById('grid').innerHTML = '';
  setStatus('');
  showDetect = false;
  resetViewDefaults();
  loadOptions().catch(err => setStatus(String(err), true));
}

async function refreshState(updateTitle=true) {
  state = await fetchJson('/api/state');
  if (!state.started) {
    showMenu();
    return;
  }
  document.getElementById('menu').classList.add('hidden');
  document.getElementById('browserHeader').classList.remove('hidden');
  document.getElementById('titleDataset').textContent = state.dataset_label || state.dataset || 'Dataset';
  document.getElementById('titleTract').textContent = state.tract;
  const patchSelect = document.getElementById('patchSelect');
  if (updateTitle) fillSelect(patchSelect, state.patches, [state.patch]);
  document.getElementById('detect').textContent = showDetect ? 'Hide Detect' : 'Detect';
  updateViewMenu();
  const warningText = state.warnings && state.warnings.length ? ` Warnings: ${state.warnings.join(' | ')}` : '';
  setStatus(`${state.dataset_label || state.dataset} ${state.patch}: ${state.n_candidates} frames from ${state.n_tiles} spatial tiles; groups/tile=${state.frames_per_tile}; tiles/page=${state.tiles_per_page}; detect batch=${state.detect_batch_size}; selected ${state.n_selected}. Scaling=${state.scaling_mode}; checkpoint=${state.checkpoint_name}.${warningText}`, Boolean(warningText));
}

async function startBrowser() {
  const cfg = activeDatasetOptions();
  if (!cfg || !cfg.enabled) {
    setStatus('Selected dataset is not available yet.', true);
    return;
  }
  const nRaw = document.getElementById('nTilesInput').value.trim();
  const framesRaw = document.getElementById('framesPerTileInput').value.trim();
  const tilesPageRaw = document.getElementById('tilesPerPageInput').value.trim();
  const maxTiles = document.getElementById('maxTilesInput').checked;
  const payload = {
    dataset: selectedDataset,
    tract: document.getElementById('tractInput').value.trim() || cfg.tract || 'default',
    patches: selectedValues(document.getElementById('patchInput')),
    bands: selectedValues(document.getElementById('bandInput')),
    run_name: document.getElementById('runNameInput').value.trim(),
    n_tiles: maxTiles || !nRaw ? null : Number(nRaw),
    frames_per_tile: framesRaw ? Number(framesRaw) : null,
    tiles_per_page: tilesPageRaw ? Number(tilesPageRaw) : null,
    all_tiles: maxTiles
  };
  setStatus('Building browser state...');
  await fetchJson('/api/start', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  page = 0;
  showDetect = false;
  resetViewDefaults();
  await refreshState();
  await loadPage(0);
}

function viewData() {
  const cfg = activeDatasetOptions();
  const label = cfg && cfg.label ? cfg.label : selectedDataset;
  if (selectedDataset !== 'hsc_image') {
    alert(`Data quality preview for ${label}: to be implemented`);
    return;
  }
  window.location.href = '/pages/data_quality.html';
}

async function changePatch() {
  const patch = document.getElementById('patchSelect').value;
  await fetchJson('/api/set_patch', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({patch})});
  page = 0;
  showDetect = false;
  resetViewDefaults();
  await refreshState();
  await loadPage(0);
}

async function loadPage(nextPage, preserveDetect=false) {
  if (busy || !state || !state.started) return;
  busy = true;
  if (!preserveDetect) {
    showDetect = false;
    document.getElementById('detect').textContent = 'Detect';
  }
  updateViewMenu();
  page = Math.max(0, Math.min(nextPage, state.n_pages - 1));
  document.getElementById('pageInput').value = page + 1;
  document.getElementById('pageText').textContent = `/ ${state.n_pages}`;
  document.getElementById('prev').disabled = page <= 0;
  document.getElementById('next').disabled = page >= state.n_pages - 1;
  const grid = document.getElementById('grid');
  grid.innerHTML = '<div class="empty">Loading</div>';
  try {
    const payload = await fetchJson(`/api/page?page=${page}`);
    const nColumns = Math.max(2, payload.bands.length);
    grid.style.gridTemplateColumns = `repeat(${nColumns}, minmax(140px, 1fr))`;
    grid.innerHTML = '';
    if (!payload.candidates.length) {
      grid.innerHTML = '<div class="empty">No frames on this page for the available band intersection.</div>';
    } else {
      for (const c of payload.candidates) grid.appendChild(candidateCell(c));
    }
  } catch (err) {
    grid.innerHTML = `<div class="empty error">${String(err)}</div>`;
  } finally {
    busy = false;
  }
}

function openSearch() {
  if (!state || !state.started) return;
  pendingSearchPage = null;
  document.getElementById('searchStatus').textContent = '';
  document.getElementById('tileMapImage').src = `/tile_map/${encodeURIComponent(state.patch)}.png?patch=${encodeURIComponent(state.patch)}&v=${Date.now()}`;
  document.getElementById('searchOverlay').classList.remove('hidden');
  document.getElementById('tileIdInput').focus();
}

function closeSearch() {
  pendingSearchPage = null;
  for (const id of ['tileIdInput', 'tileYInput', 'pixelXInput', 'pixelYInput']) {
    document.getElementById(id).value = '';
  }
  document.getElementById('searchStatus').textContent = '';
  document.getElementById('searchOverlay').classList.add('hidden');
}

function smoothKernelText() {
  if (smoothMode === 'gaussian') {
    const sigma = Number(smoothSigma);
    const r = Math.max(1, Math.ceil(2 * sigma));
    return `Gaussian: sigma=${sigma.toFixed(2)}, r=ceil(2sigma)=${r}, D=${2 * r + 1}.`;
  }
  const r = Math.max(1, Math.round(Number(smoothRadius)));
  return `${smoothMode === 'boxcar' ? 'Boxcar' : 'Tophat'}: r=${r}, D=${2 * r + 1}.`;
}

function refreshSmoothDialog() {
  document.getElementById('smoothModeSelect').value = smoothMode;
  document.getElementById('smoothSigmaInput').value = String(smoothSigma);
  document.getElementById('smoothRadiusInput').value = String(smoothRadius);
  refreshSmoothDialogLayout();
}

function refreshSmoothDialogLayout() {
  const gaussian = smoothMode === 'gaussian';
  document.getElementById('smoothSigmaLabel').classList.toggle('hidden', !gaussian);
  document.getElementById('smoothSigmaInput').classList.toggle('hidden', !gaussian);
  document.getElementById('smoothRadiusLabel').classList.toggle('hidden', gaussian);
  document.getElementById('smoothRadiusInput').classList.toggle('hidden', gaussian);
  document.getElementById('smoothKernelText').textContent = smoothKernelText();
}

function openSmooth() {
  if (!state || !state.started) return;
  refreshSmoothDialog();
  document.getElementById('smoothOverlay').classList.remove('hidden');
}

function closeSmooth() {
  document.getElementById('smoothOverlay').classList.add('hidden');
}

async function applySmooth() {
  smoothMode = document.getElementById('smoothModeSelect').value;
  smoothSigma = Math.max(0.1, Number(document.getElementById('smoothSigmaInput').value || 1.0));
  smoothRadius = Math.max(1, Math.round(Number(document.getElementById('smoothRadiusInput').value || 1)));
  smoothEnabled = true;
  updateViewMenu();
  closeSmooth();
  setStatus(viewStatusText());
  await loadPage(page, true);
}

async function disableSmooth() {
  smoothEnabled = false;
  updateViewMenu();
  closeSmooth();
  setStatus(viewStatusText());
  await loadPage(page, true);
}

async function runTileSearch(mode, jump=true) {
  if (!state || !state.started) return;
  const isTile = mode === 'tile_xy';
  const xInput = document.getElementById(isTile ? 'tileIdInput' : 'pixelXInput');
  const yInput = document.getElementById(isTile ? 'tileYInput' : 'pixelYInput');
  const body = {mode};
  if (isTile) {
    const raw = xInput.value.trim();
    const yRaw = yInput.value.trim();
    if (!raw) {
      document.getElementById('searchStatus').textContent = 'Enter a tile id such as 0,1, 000,001, or x000_y001.';
      pendingSearchPage = null;
      return;
    }
    body.tile_id = raw;
    if (yRaw) body.y = Number(yRaw);
  } else {
    const x = Number(xInput.value);
    const y = Number(yInput.value);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      document.getElementById('searchStatus').textContent = 'Enter valid x and y values.';
      pendingSearchPage = null;
      return;
    }
    body.x = x;
    body.y = y;
  }
  try {
    const result = await fetchJson('/api/find_tile', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    pendingSearchPage = Number(result.page);
    if (jump) {
      const targetPage = pendingSearchPage;
      closeSearch();
      await loadPage(targetPage);
    } else {
      document.getElementById('searchStatus').textContent =
        `${result.tile_id} local X=[${result.x0},${result.x1}) Y=[${result.y0},${result.y1}) on page ${result.page_display}/${result.n_pages}.`;
    }
  } catch (err) {
    pendingSearchPage = null;
    document.getElementById('searchStatus').textContent = String(err);
  }
}

async function detectPage() {
  if (!state || !state.started) return;
  if (showDetect) {
    showDetect = false;
    document.getElementById('detect').textContent = 'Detect';
    await loadPage(page, true);
    return;
  }
  document.getElementById('detect').disabled = true;
  document.getElementById('viewButton').disabled = true;
  setStatus('Running CELLECT detection on the current page...');
  try {
    const result = await fetchJson('/api/detect_page', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({page})
    });
    showDetect = true;
    setStatus(`Detected current page: ${result.n_images} images, ${result.n_detections} detections.`);
    await refreshState(false);
    await loadPage(page, true);
  } catch (err) {
    setStatus(String(err), true);
  } finally {
    document.getElementById('detect').disabled = false;
    document.getElementById('viewButton').disabled = false;
  }
}

async function saveCsv() {
  const result = await fetchJson('/api/save_selection_csv', {method: 'POST'});
  setStatus(`Saved ${result.n_selected} selected rows to ${result.selection_csv}.`);
}

async function exportSelected() {
  const result = await fetchJson('/api/export', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({write_png: true})
  });
  setStatus(`Exported ${result.n_exported} raw frames and ${result.n_detections} detections to ${result.out_dir}.`);
}

document.getElementById('prev').onclick = () => loadPage(page - 1);
document.getElementById('next').onclick = () => loadPage(page + 1);
document.getElementById('pageInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    loadPage(Number(document.getElementById('pageInput').value || 1) - 1).catch(err => setStatus(String(err), true));
  }
});
document.getElementById('searchButton').onclick = () => openSearch();
document.getElementById('closeSearch').onclick = () => closeSearch();
document.getElementById('tileSearchGo').onclick = () => runTileSearch('tile_xy');
document.getElementById('pixelSearchGo').onclick = () => runTileSearch('pixel_xy');
document.getElementById('searchOverlay').addEventListener('click', (event) => {
  if (event.target.id === 'searchOverlay') closeSearch();
});
document.getElementById('closeSmooth').onclick = () => closeSmooth();
document.getElementById('applySmoothButton').onclick = () => applySmooth().catch(err => setStatus(String(err), true));
document.getElementById('smoothOffButton').onclick = () => disableSmooth().catch(err => setStatus(String(err), true));
document.getElementById('smoothModeSelect').addEventListener('change', () => {
  smoothMode = document.getElementById('smoothModeSelect').value;
  refreshSmoothDialog();
});
document.getElementById('smoothSigmaInput').addEventListener('input', () => {
  smoothSigma = Math.max(0.1, Number(document.getElementById('smoothSigmaInput').value || 1.0));
  refreshSmoothDialogLayout();
});
document.getElementById('smoothRadiusInput').addEventListener('input', () => {
  smoothRadius = Math.max(1, Math.round(Number(document.getElementById('smoothRadiusInput').value || 1)));
  refreshSmoothDialogLayout();
});
document.getElementById('smoothOverlay').addEventListener('click', (event) => {
  if (event.target.id === 'smoothOverlay') closeSmooth();
});
for (const id of ['tileIdInput', 'tileYInput']) {
  document.getElementById(id).addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      runTileSearch('tile_xy').catch(err => {
        document.getElementById('searchStatus').textContent = String(err);
      });
    }
  });
}
for (const id of ['pixelXInput', 'pixelYInput']) {
  document.getElementById(id).addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      runTileSearch('pixel_xy').catch(err => {
        document.getElementById('searchStatus').textContent = String(err);
      });
    }
  });
}
document.getElementById('back').onclick = () => showMenu();
document.getElementById('detect').onclick = () => detectPage();
document.getElementById('viewButton').onclick = (event) => {
  event.stopPropagation();
  document.getElementById('viewDropdown').classList.toggle('hidden');
  updateViewMenu();
};
for (const item of document.querySelectorAll('.viewItem')) {
  item.onclick = (event) => {
    event.stopPropagation();
    toggleViewItem(item.dataset.view).catch(err => setStatus(String(err), true));
  };
}
document.addEventListener('click', (event) => {
  const menu = document.querySelector('.viewMenu');
  if (menu && !menu.contains(event.target)) document.getElementById('viewDropdown').classList.add('hidden');
});
document.getElementById('patchSelect').onchange = () => changePatch().catch(err => setStatus(String(err), true));
document.getElementById('saveCsv').onclick = () => saveCsv().catch(err => setStatus(String(err), true));
document.getElementById('export').onclick = () => exportSelected().catch(err => setStatus(String(err), true));
document.getElementById('startButton').onclick = () => startBrowser().catch(err => setStatus(String(err), true));
document.getElementById('viewDataButton').onclick = () => viewData();

(async () => {
  try {
    await loadOptions();
    await refreshState();
  } catch (err) {
    setStatus(String(err), true);
  }
})();
