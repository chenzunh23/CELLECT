const assert = require('assert').strict;
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(require('path').join(__dirname, '../hsctiles/pages/app.js'), 'utf8');
const elements = new Map();
const document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, {disabled:false, textContent:''});
  return elements.get(id);
}};
let finish, poll;
const context = vm.createContext({document, state:{started:true}, detectBusy:false,
  exportBusy:false, showDetect:false, imageVersion:0, page:0, window:{},
  fetchJson(url) {
    if (url === '/api/detect_status') return Promise.resolve({stage:'inference', completed:0, total:10, elapsed_seconds:3});
    return new Promise(resolve => {finish=resolve;});
  },
  setInterval(fn) {poll=fn; return 1;}, clearInterval() {poll=null;},
  refreshState:async()=>{}, loadPage:async()=>{},
});
vm.runInContext(source.match(/function setStatus\([\s\S]*?\n\}/)[0], context);
vm.runInContext(source.slice(source.indexOf('async function detectPage('), source.indexOf('async function saveCsv(')), context);
(async()=>{
  const pending=vm.runInContext('detectPage()',context);
  assert(elements.get('detect').disabled);
  await poll();
  assert(elements.get('status').textContent.includes('model inference'));
  vm.runInContext("setStatus('Page loaded')", context);
  assert(elements.get('status').textContent.includes('model inference'));
  finish({n_images:10, n_detections:1142, n_failed_images:0});
  await pending;
  assert(elements.get('status').textContent.includes('1142 detections'));
  assert.equal(elements.get('detect').disabled,false);
  assert.equal(poll,null);
  const failed=vm.runInContext('showDetect=false; detectPage()',context);
  finish({n_images:0, n_failed_images:10, errors:['bad checkpoint']});
  await failed;
  assert(elements.get('status').textContent.includes('bad checkpoint'));
  assert.equal(context.showDetect,false);
  // A different tab starting a new session must not silently show its empty page.
  context.state.session_name='old';
  context.refreshState=async()=>{context.state.session_name='new';};
  const stale=vm.runInContext('detectPage()',context);
  finish({n_images:10,n_detections:1142,n_failed_images:0,session_name:'old'});
  await stale;
  assert(elements.get('status').textContent.includes('session changed'));
  assert.equal(context.showDetect,false);

  // Two Go clicks while the first request is pending issue only one POST.
  context.startBusy=false;
  context.activeDatasetOptions=()=>({enabled:true});
  context.selectedDataset='jwst_abell';
  context.selectedValues=()=>[];
  context.resetViewDefaults=()=>{};
  context.refreshState=async()=>{};
  let starts=0,finishStart;
  context.fetchJson=()=>{starts++;return new Promise(resolve=>{finishStart=resolve;});};
  for(const id of ['nTilesInput','framesPerTileInput','tilesPerPageInput','nmsRadiusInput','tractInput','runNameInput']) {
    document.getElementById(id).value='';
  }
  vm.runInContext(source.slice(source.indexOf('async function startBrowser('),source.indexOf('function viewData(')),context);
  const starting=vm.runInContext('startBrowser()',context);
  await vm.runInContext('startBrowser()',context);
  assert.equal(starts,1);
  assert(elements.get('startButton').disabled);
  finishStart({});await starting;
  assert.equal(elements.get('startButton').disabled,false);
  assert.equal(context.startBusy,false);
  console.log('Detect progress, completion, and error rendering passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
