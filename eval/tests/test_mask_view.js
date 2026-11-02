const assert = require('assert').strict;
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(require('path').join(__dirname,'../hsctiles/pages/app.js'),'utf8');
const context = vm.createContext({
  state:{started:true,make_masks:true}, showDetect:true, viewMasks:false,
  viewShape:false,viewCenter:false,viewInvert:true,smoothEnabled:false,
  imageVersion:1,page:0, window:{},
  updateViewMenu(){}, viewStatusText(){return '';},setStatus(){},loadPage:async()=>{},
});
vm.runInContext(source.slice(source.indexOf('function imageUrl('),source.indexOf('function updateViewMenu(')),context);
vm.runInContext(source.slice(source.indexOf('async function toggleViewItem('),source.indexOf('function viewStatusText(')),context);
(async()=>{
  await vm.runInContext("toggleViewItem('masks')",context);
  assert.equal(context.viewMasks,true);
  let url=vm.runInContext("imageUrl({token:'x'})",context);
  assert(url.includes('masks=1') && url.includes('invert=1'));
  await vm.runInContext("toggleViewItem('masks')",context);
  assert.equal(context.viewMasks,false);
  context.showDetect=false;
  await vm.runInContext("toggleViewItem('masks')",context);
  assert.equal(context.viewMasks,false);
  // Settings use the SNR-style Apply/Off flow; thresholds are inclusive and
  // only add View query parameters after Apply.
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {value:'',textContent:'',disabled:false,
      classList:{add(){},remove(){},toggle(){}},handlers:{},
      addEventListener(name,handler){this.handlers[name]=handler;}});
    return elements.get(id);
  }
  context.document={getElementById:element,querySelectorAll:()=>[]};
  vm.runInContext(fs.readFileSync(require('path').join(__dirname,'../hsctiles/pages/mask_controls.js'),'utf8'),context);
  context.reloads=0;context.detectReady=true;
  vm.runInContext("MaskControls.init({currentState:()=>state,detectActive:()=>detectReady,setStatus, reload:async()=>{reloads++;}})",context);
  assert.equal(element('maskPredIouInput').value,'0.8');
  assert.equal(element('maskStabilityInput').value,'0.9');
  assert(!vm.runInContext("imageUrl({token:'x'})",context).includes('mask_filter='));
  await element('applyMaskFilterButton').onclick();
  let qualityUrl=vm.runInContext("imageUrl({token:'x'})",context);
  assert(qualityUrl.includes('mask_filter=1') && qualityUrl.includes('mask_pred_iou=0.8') && qualityUrl.includes('mask_stability=0.9'));
  element('maskStabilityRange').value='0.96';element('maskStabilityRange').handlers.input();
  assert.equal(element('maskStabilityInput').value,'0.96');
  await element('applyMaskFilterButton').onclick();
  assert(vm.runInContext("imageUrl({token:'x'})",context).includes('mask_stability=0.96'));
  await element('maskFilterOffButton').onclick();
  assert(!vm.runInContext('MaskControls.isEnabled()',context));
  vm.runInContext('MaskControls.reset();detectReady=false',context);
  await element('applyMaskFilterButton').onclick();
  assert(!vm.runInContext('MaskControls.isEnabled()',context));
  assert.equal(element('maskStabilityInput').value,'0.9');
  console.log('Show Masks toggle / request parameters passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
