const ScaleControls = (() => {
  const modes = ['input', 'zscale', 'zmax', 'asinh', 'log', 'lupton', 'anscombe', 'square'];
  const labels = {
    input: 'Input Scaling',
    zscale: 'ZScale',
    zmax: 'ZMax',
    asinh: 'Asinh',
    log: 'Log',
    lupton: 'Lupton',
    anscombe: 'Anscombe',
    square: 'Square'
  };
  let callbacks = {};
  let mode = 'zscale';
  let custom = false;
  let highPct = 100;
  let stats = null;

  function el(id) {
    return document.getElementById(id);
  }

  function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function format(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '';
    const abs = Math.abs(number);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return number.toExponential(5);
    return number.toPrecision(7);
  }

  function percentileValue(pct) {
    if (!stats || !stats.percentiles) return '';
    let exact = stats.percentiles[String(Number(pct))];
    if (exact == null) exact = stats.percentiles[String(pct)];
    return exact == null ? '' : format(exact);
  }

  function currentParams() {
    if (mode === 'input') return [];
    const params = [`display_scaling=${encodeURIComponent(mode)}`];
    if (custom && mode !== 'zscale') {
      params.push('scale_custom=1');
      params.push('scale_low_pct=0');
      params.push(`scale_high_pct=${encodeURIComponent(String(highPct))}`);
    }
    return params;
  }

  function statusText() {
    let text = labels[mode] || mode;
    if (custom && mode !== 'input' && mode !== 'zscale') {
      text += ` p=[0, ${highPct}]`;
    }
    return text;
  }

  function refreshMenu() {
    for (const item of document.querySelectorAll('.scaleItem')) {
      const active = item.dataset.scale === mode;
      item.classList.toggle('active', active);
    }
    for (const item of document.querySelectorAll('.viewItem')) {
      if (item.dataset.view === 'inputScaling') item.classList.toggle('active', mode === 'input');
    }
    const settings = el('scaleSettingsButton');
    if (settings) settings.disabled = mode === 'input' || mode === 'zscale';
  }

  function setRangesFromStats() {
    el('scalePctMaxRange').value = String(highPct);
    el('scalePctMaxInput').value = String(highPct);
  }

  function refreshPresetButtons() {
    for (const button of document.querySelectorAll('.scalePreset')) {
      const target = Number(button.dataset.highPct);
      button.classList.toggle('active', highPct === target);
    }
  }

  function refreshStatsText() {
    if (!stats) {
      el('scaleStatsText').textContent = 'No image stats loaded.';
      return;
    }
    const finite = Number.isFinite(stats.finite_fraction) ? ` finite=${(100 * stats.finite_fraction).toFixed(2)}%` : '';
    const zscale = `zscale=[${format(stats.zscale_min)}, ${format(stats.zscale_max)}]`;
    const minmax = `min/max=[${format(stats.min)}, ${format(stats.max)}]`;
    const pct = `p=[0, ${highPct}] values=[${percentileValue(0) || 'n/a'}, ${percentileValue(highPct) || 'n/a'}]`;
    el('scaleStatsText').textContent = `${minmax}; ${zscale}; ${pct};${finite}`;
  }

  function refreshDialog() {
    setRangesFromStats();
    refreshPresetButtons();
    refreshStatsText();
  }

  async function openSettings() {
    const token = callbacks.firstToken ? callbacks.firstToken() : '';
    stats = null;
    el('scaleStatsText').textContent = token ? 'Loading image stats...' : 'No image on the current page.';
    el('scaleOverlay').classList.remove('hidden');
    if (token && callbacks.fetchJson) {
      try {
        stats = await callbacks.fetchJson(`/api/scale_stats?token=${encodeURIComponent(token)}`);
      } catch (err) {
        el('scaleStatsText').textContent = String(err);
      }
    }
    refreshDialog();
  }

  function closeSettings() {
    el('scaleOverlay').classList.add('hidden');
  }

  function setPercentileRange(hi) {
    highPct = Math.max(0, Math.min(100, Number(hi)));
    custom = true;
    refreshDialog();
  }

  function readDialogValues() {
    const pHi = finiteNumber(el('scalePctMaxInput').value);
    highPct = Math.max(0, Math.min(100, pHi == null ? 100 : pHi));
    custom = true;
  }

  async function applySettings() {
    if (mode === 'input' || mode === 'zscale') {
      closeSettings();
      return;
    }
    readDialogValues();
    closeSettings();
    refreshMenu();
    if (callbacks.onChange) await callbacks.onChange();
  }

  async function resetSettings() {
    custom = false;
    highPct = 100;
    refreshDialog();
    refreshMenu();
    if (callbacks.onChange) await callbacks.onChange();
  }

  function reset() {
    mode = 'zscale';
    custom = false;
    highPct = 100;
    refreshMenu();
  }

  function init(options) {
    callbacks = options || {};
    for (const item of document.querySelectorAll('.scaleItem')) {
      item.addEventListener('click', async (event) => {
        event.stopPropagation();
        const nextMode = item.dataset.scale;
        if (!modes.includes(nextMode)) return;
        mode = nextMode;
        refreshMenu();
        if (callbacks.onChange) await callbacks.onChange();
      });
    }
    el('scaleSettingsButton').addEventListener('click', (event) => {
      event.stopPropagation();
      openSettings();
    });
    el('closeScale').onclick = () => closeSettings();
    el('scaleOverlay').addEventListener('click', (event) => {
      if (event.target.id === 'scaleOverlay') closeSettings();
    });
    for (const button of document.querySelectorAll('.scalePreset')) {
      button.addEventListener('click', () => setPercentileRange(Number(button.dataset.highPct)));
    }
    for (const id of ['scalePctMaxRange', 'scalePctMaxInput']) {
      el(id).addEventListener('input', () => {
        const value = finiteNumber(el(id).value);
        if (value != null) highPct = Math.max(0, Math.min(100, value));
        el('scalePctMaxRange').value = String(highPct);
        el('scalePctMaxInput').value = String(highPct);
        refreshPresetButtons();
        refreshStatsText();
      });
    }
    el('applyScaleButton').onclick = () => applySettings().catch(err => callbacks.setStatus && callbacks.setStatus(String(err), true));
    el('scaleResetButton').onclick = () => resetSettings().catch(err => callbacks.setStatus && callbacks.setStatus(String(err), true));
    refreshMenu();
  }

  async function setMode(nextMode) {
    if (!modes.includes(nextMode)) return;
    mode = nextMode;
    refreshMenu();
    if (callbacks.onChange) await callbacks.onChange();
  }

  return {
    init,
    params: currentParams,
    isInputMode: () => mode === 'input',
    reset,
    refreshMenu,
    setMode,
    statusText
  };
})();
window.ScaleControls = ScaleControls;
