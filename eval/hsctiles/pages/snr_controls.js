const SnrControls = (() => {
  let callbacks = {};
  let enabled = false;
  let useKron = false;
  let ap2Threshold = 5.0;
  let kronThreshold = 5.0;
  let busy = false;
  let byToken = {};

  function el(id) {
    return document.getElementById(id);
  }

  function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function method() {
    return useKron ? 'kron' : 'ap2';
  }

  function methodLabel() {
    return useKron ? 'Kron' : 'AP2';
  }

  function threshold() {
    return useKron ? kronThreshold : ap2Threshold;
  }

  function params() {
    if (!enabled) return [];
    return [
      'snr_filter=1',
      `snr_method=${encodeURIComponent(method())}`,
      `snr_threshold=${encodeURIComponent(String(threshold()))}`
    ];
  }

  function statusText() {
    if (!enabled) return 'snr=off';
    return `snr=${method()}>=${threshold().toFixed(1)}; bad shown`;
  }

  function labelText(token) {
    if (!enabled) return '';
    const row = byToken[String(token)];
    if (!row) return `${methodLabel()} SNR>${threshold().toFixed(1)}, sources:..., bad:...`;
    return `${methodLabel()} SNR>${Number(row.threshold).toFixed(1)}, sources:${row.sources}, bad:${row.bad}`;
  }

  function refreshUi() {
    for (const item of document.querySelectorAll('.viewItem')) {
      if (item.dataset.view === 'snrFilter') item.classList.toggle('active', enabled);
    }
    const check = el('snrKronToggle');
    if (check) check.checked = useKron;
    const ap2Range = el('ap2SnrRange');
    const ap2Input = el('ap2SnrInput');
    const kronRange = el('kronSnrRange');
    const kronInput = el('kronSnrInput');
    if (ap2Range) ap2Range.value = String(ap2Threshold);
    if (ap2Input) ap2Input.value = String(ap2Threshold);
    if (kronRange) kronRange.value = String(kronThreshold);
    if (kronInput) kronInput.value = String(kronThreshold);
    const fields = document.querySelector('.snrModalFields');
    if (fields) {
      fields.classList.toggle('enabled', enabled);
      fields.classList.toggle('kronActive', useKron);
    }
    const status = el('snrStatusText');
    if (status) status.textContent = statusText();
    for (const id of ['applySnrButton', 'snrOffButton']) {
      const button = el(id);
      if (button) button.disabled = busy;
    }
  }

  function openSettings() {
    refreshUi();
    el('snrOverlay').classList.remove('hidden');
  }

  function closeSettings() {
    el('snrOverlay').classList.add('hidden');
  }

  async function prepareCurrentPage() {
    if (!enabled || !callbacks.fetchJson || !callbacks.currentPage) return null;
    if (callbacks.detectActive && !callbacks.detectActive()) {
      byToken = {};
      if (callbacks.setStatus) callbacks.setStatus('SNR filter is enabled; run Detect before applying it.');
      return null;
    }
    const page = callbacks.currentPage();
    if (callbacks.setStatus) callbacks.setStatus('SNR stage 1/2: collecting current-page detections...');
    await new Promise(resolve => setTimeout(resolve, 0));
    if (callbacks.setStatus) callbacks.setStatus(`SNR stage 2/2: estimating ${method()} background/noise...`);
    const result = await callbacks.fetchJson('/api/prepare_snr', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({page, method: method(), threshold: threshold()})
    });
    byToken = result.by_token || {};
    return result;
  }

  async function applyChange(options={}) {
    if (busy) return;
    busy = true;
    refreshUi();
    try {
      const result = await prepareCurrentPage();
      if (result && callbacks.setStatus) {
        callbacks.setStatus(
          `${methodLabel()} SNR>${Number(result.threshold).toFixed(1)}: sources=${result.n_sources}, bad=${result.n_bad} on ${result.n_images} images.`
        );
      } else if (callbacks.setStatus) {
        callbacks.setStatus(callbacks.statusText ? callbacks.statusText() : statusText());
      }
      if (callbacks.reload) await callbacks.reload();
      if (options.close) closeSettings();
    } finally {
      busy = false;
      refreshUi();
    }
  }

  async function applySettings() {
    enabled = true;
    await applyChange({close: true});
  }

  async function disable() {
    enabled = false;
    byToken = {};
    closeSettings();
    refreshUi();
    if (callbacks.reload) await callbacks.reload();
  }

  function reset() {
    enabled = false;
    useKron = false;
    ap2Threshold = 5.0;
    kronThreshold = 5.0;
    byToken = {};
    refreshUi();
  }

  function bindThreshold(rangeId, inputId, setter) {
    const range = el(rangeId);
    const input = el(inputId);
    if (!range || !input) return;
    const updateLocal = (value) => {
      const next = Math.max(0.0, finiteNumber(value, 5.0));
      setter(next);
      refreshUi();
    };
    range.addEventListener('input', () => updateLocal(range.value));
    input.addEventListener('input', () => updateLocal(input.value));
  }

  function init(options) {
    callbacks = options || {};
    const kron = el('snrKronToggle');
    if (kron) {
      kron.addEventListener('change', () => {
        useKron = Boolean(kron.checked);
        byToken = {};
        refreshUi();
      });
    }
    bindThreshold('ap2SnrRange', 'ap2SnrInput', value => { ap2Threshold = value; });
    bindThreshold('kronSnrRange', 'kronSnrInput', value => { kronThreshold = value; });
    el('closeSnr').onclick = () => closeSettings();
    el('applySnrButton').onclick = () => applySettings().catch(err => callbacks.setStatus && callbacks.setStatus(String(err), true));
    el('snrOffButton').onclick = () => disable().catch(err => callbacks.setStatus && callbacks.setStatus(String(err), true));
    el('snrOverlay').addEventListener('click', (event) => {
      if (event.target.id === 'snrOverlay') closeSettings();
    });
    refreshUi();
  }

  return {
    init,
    params,
    prepare: prepareCurrentPage,
    reset,
    refreshMenu: refreshUi,
    statusText,
    labelText,
    open: openSettings,
    isEnabled: () => enabled,
    method,
    threshold
  };
})();
window.SnrControls = SnrControls;
