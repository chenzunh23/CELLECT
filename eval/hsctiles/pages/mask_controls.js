const MaskControls = (() => {
  let callbacks = {};
  let enabled = false;
  let predIou = 0.8;
  let stability = 0.9;
  let busy = false;
  const el = id => document.getElementById(id);

  function params() {
    return enabled ? ['mask_filter=1', `mask_pred_iou=${predIou}`, `mask_stability=${stability}`] : [];
  }
  function statusText() {
    return enabled ? `mask=pred IoU>=${predIou.toFixed(2)}, stability>=${stability.toFixed(2)}` : 'mask filter=off';
  }
  function refreshUi() {
    for (const item of document.querySelectorAll('.viewItem')) {
      if (item.dataset.view === 'maskFilter') item.classList.toggle('active', enabled);
    }
    for (const id of ['maskPredIouRange', 'maskPredIouInput']) el(id).value = String(predIou);
    for (const id of ['maskStabilityRange', 'maskStabilityInput']) el(id).value = String(stability);
    el('maskFilterStatus').textContent = statusText();
    for (const id of ['applyMaskFilterButton', 'maskFilterOffButton']) el(id).disabled = busy;
  }
  function open() {
    refreshUi();
    el('maskFilterOverlay').classList.remove('hidden');
  }
  function close() { el('maskFilterOverlay').classList.add('hidden'); }
  async function change(next) {
    if (busy) return;
    if (next && callbacks.detectActive && !callbacks.detectActive()) {
      callbacks.setStatus('Run Detect before applying Mask Filter.');
      return;
    }
    const state = callbacks.currentState ? callbacks.currentState() : null;
    if (next && state && state.make_masks === false) {
      callbacks.setStatus('Mask decoding is disabled on this server.', true);
      return;
    }
    busy = true;
    enabled = next;
    refreshUi();
    try {
      if (callbacks.reload) await callbacks.reload();
      close();
    } finally {
      busy = false;
      refreshUi();
    }
  }
  function reset() {
    enabled = false;
    predIou = 0.8;
    stability = 0.9;
    refreshUi();
  }
  function bind(rangeId, inputId, fallback, setter) {
    for (const id of [rangeId, inputId]) {
      el(id).addEventListener('input', () => {
        const value = Number(el(id).value);
        setter(Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : fallback);
        refreshUi();
      });
    }
  }
  function init(options) {
    callbacks = options || {};
    bind('maskPredIouRange', 'maskPredIouInput', 0.8, value => { predIou = value; });
    bind('maskStabilityRange', 'maskStabilityInput', 0.9, value => { stability = value; });
    el('closeMaskFilter').onclick = close;
    const handleError = err => callbacks.setStatus && callbacks.setStatus(String(err), true);
    el('applyMaskFilterButton').onclick = () => change(true).catch(handleError);
    el('maskFilterOffButton').onclick = () => change(false).catch(handleError);
    el('maskFilterOverlay').addEventListener('click', event => {
      if (event.target.id === 'maskFilterOverlay') close();
    });
    refreshUi();
  }
  return {init, params, statusText, reset, open, refreshMenu: refreshUi, isEnabled: () => enabled};
})();
window.MaskControls = MaskControls;
