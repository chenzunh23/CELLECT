let meta = null;
const state = {source: 'coadd', band: 'HSC-I', group: '0', page: 0, overlay: true};
const $ = id => document.getElementById(id);

async function getJson(url) {
  const res = await fetch(url);
  const text = await res.text();
  if (!res.ok) throw new Error(text || res.statusText);
  return text ? JSON.parse(text) : {};
}

function fillSelect(el, values, selected) {
  el.innerHTML = '';
  for (const value of values || []) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = value;
    opt.selected = value === selected;
    el.appendChild(opt);
  }
}

function sourceConfig() {
  return (meta.sources || []).find(row => row.id === state.source) || {};
}

function bandsForSource() {
  return (meta.source_bands && meta.source_bands[state.source] && meta.source_bands[state.source].length)
    ? meta.source_bands[state.source]
    : meta.bands;
}

function query(extra={}) {
  const params = new URLSearchParams({
    source: state.source,
    band: state.band,
    group: state.group,
    page: String(state.page),
    overlay: state.overlay ? '1' : '0'
  });
  for (const [key, value] of Object.entries(extra)) params.set(key, value);
  return params.toString();
}

function renderPages(pagesWithData=null) {
  const box = $('pageButtons');
  box.innerHTML = '';
  for (let i = 0; i < 9; i++) {
    const btn = document.createElement('button');
    btn.textContent = `page ${i + 1}`;
    btn.className = i === state.page ? 'active' : '';
    if (pagesWithData && pagesWithData.length && !pagesWithData.includes(i)) btn.style.opacity = '0.42';
    btn.onclick = () => {
      state.page = i;
      renderPages(pagesWithData);
      refresh().catch(showError);
    };
    box.appendChild(btn);
  }
}

function showError(err) {
  $('qualityStatus').textContent = String(err);
  $('qualityStatus').className = 'status error';
}

async function refresh() {
  const usesGroup = Boolean(sourceConfig().uses_group);
  $('groupSelect').disabled = !usesGroup;
  $('qualitySubtitle').textContent = `${state.source} ${state.band}${usesGroup ? ` group ${state.group}` : ''}`;
  renderPages();
  $('pageImage').src = `/image/data_quality/page.png?${query({_: Date.now()})}`;
  $('qualityStatus').className = 'status';
  $('qualityStatus').textContent = `EDGE=${meta.edge_weight}, threshold=${meta.threshold}, tile=${meta.tile_size}/${meta.tile_stride}`;
  const drops = await getJson(`/api/data_quality/drops?${query()}`);
  if (drops.pages_with_data && drops.pages_with_data.length && !drops.pages_with_data.includes(state.page)) {
    state.page = drops.pages_with_data[0];
    $('pageImage').src = `/image/data_quality/page.png?${query({_: Date.now()})}`;
  }
  renderPages(drops.pages_with_data || null);
  const pageNote = drops.pages_with_data && drops.pages_with_data.length
    ? ` pages with data: ${drops.pages_with_data.map(p => p + 1).join(', ')}`
    : '';
  $('qualitySummary').textContent = `${drops.source} ${drops.band}${drops.group_note}: ${drops.drop_count}/${drops.ok_count} patches dropped.${pageNote}`;
  $('dropList').textContent = drops.drops.length
    ? drops.drops.map(row => `${row.patch}\t${row.score.toFixed(5)}\t${row.path}`).join('\n')
    : '(none)';
}

async function load() {
  meta = await getJson('/api/data_quality/meta');
  fillSelect($('sourceSelect'), meta.sources.map(row => row.id), state.source);
  const bands = bandsForSource();
  if (!bands.includes(state.band)) state.band = bands[0] || state.band;
  fillSelect($('bandSelect'), bands, state.band);
  fillSelect($('groupSelect'), meta.groups, state.group);
  await refresh();
}

$('sourceSelect').onchange = event => {
  state.source = event.target.value;
  const bands = bandsForSource();
  if (!bands.includes(state.band)) state.band = bands[0] || state.band;
  fillSelect($('bandSelect'), bands, state.band);
  state.page = 0;
  refresh().catch(showError);
};
$('bandSelect').onchange = event => {
  state.band = event.target.value;
  state.page = 0;
  refresh().catch(showError);
};
$('groupSelect').onchange = event => {
  state.group = event.target.value;
  refresh().catch(showError);
};
$('tileOverlay').onchange = event => {
  state.overlay = event.target.checked;
  refresh().catch(showError);
};
$('overviewButton').onclick = () => {
  $('overviewTitle').textContent = `${state.source} ${state.band} score map`;
  $('overviewImage').src = `/image/data_quality/overview.png?${query({_: Date.now()})}`;
  $('overviewOverlay').classList.remove('hidden');
};
$('closeOverview').onclick = () => $('overviewOverlay').classList.add('hidden');
$('helpButton').onclick = () => $('helpOverlay').classList.remove('hidden');
$('closeHelp').onclick = () => $('helpOverlay').classList.add('hidden');
$('backButton').onclick = () => history.back();
for (const id of ['overviewOverlay', 'helpOverlay']) {
  $(id).addEventListener('click', event => {
    if (event.target.id === id) $(id).classList.add('hidden');
  });
}
load().catch(showError);
