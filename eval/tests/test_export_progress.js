const assert = require('assert').strict;
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const source = fs.readFileSync(path.join(__dirname, '../hsctiles/pages/app.js'), 'utf8');
const elements = new Map();
const document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, {value: '4', disabled: false, textContent: ''});
  return elements.get(id);
}};
let finish, poll;
let progress = {stage: 'preparing', completed: 0, total: 8, workers: 4, elapsed_seconds: 5};
const context = vm.createContext({
  document, exportBusy: false, state: {n_selected: 8},
  fetchJson(url, options) {
    if (url === '/api/export_status') return Promise.resolve(progress);
    assert.deepEqual(JSON.parse(options.body), {write_png: true, workers: 4});
    return new Promise(resolve => {finish = resolve;});
  },
  setInterval(callback) {poll = callback; return 1;},
  clearInterval() {poll = null;},
});
vm.runInContext(source.match(/function setStatus\([\s\S]*?\n\}/)[0], context);
vm.runInContext(source.slice(source.indexOf('async function exportSelected('),
                            source.indexOf("document.getElementById('prev').onclick")), context);
(async () => {
  const pending = vm.runInContext('exportSelected()', context);
  assert.equal(elements.get('export').textContent, 'Exporting ...');
  assert(elements.get('export').disabled);
  assert(elements.get('status').textContent.includes('loading images'));
  const initial = elements.get('status').textContent;
  vm.runInContext("setStatus('Loading page ...')", context);
  assert.equal(elements.get('status').textContent, initial);
  await poll();
  assert(elements.get('status').textContent.includes('0/8'));
  assert(elements.get('status').textContent.includes('loading images'));
  progress = {...progress, stage: 'exporting', completed: 4};
  await poll();
  assert(elements.get('status').textContent.includes('4/8'));
  finish({n_exported: 8, n_detections: 50, elapsed_seconds: 12.3, workers: 4, out_dir: '/test'});
  await pending;
  assert.equal(elements.get('export').textContent, 'Export Selected');
  assert.equal(elements.get('export').disabled, false);
  assert(elements.get('status').textContent.includes('Exported 8'));
  assert.equal(poll, null);
  console.log('Export loading/progress text and page-loading priority passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
