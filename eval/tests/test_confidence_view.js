const assert = require('assert').strict;
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(require('path').join(__dirname,'../hsctiles/pages/app.js'),'utf8');
let redraws=0;
const context=vm.createContext({state:{started:true},showDetect:true,viewConfidence:false,
  viewShape:false,viewCenter:false,viewMasks:false,viewInvert:false,smoothEnabled:false,
  imageVersion:1,page:0,window:{},updateViewMenu(){},viewStatusText(){return '';},setStatus(){},
  loadPage:async()=>{redraws++;}});
vm.runInContext(source.slice(source.indexOf('function imageUrl('),source.indexOf('function updateViewMenu(')),context);
vm.runInContext(source.slice(source.indexOf('async function toggleViewItem('),source.indexOf('function viewStatusText(')),context);
(async()=>{
  await vm.runInContext("toggleViewItem('confidence')",context);
  assert(context.viewConfidence);
  assert(vm.runInContext("imageUrl({token:'a'})",context).includes('confidence=1'));
  context.window.ScaleControls=context.ScaleControls={params:()=>['display_scaling=log'],isInputMode:()=>false};
  const changed=vm.runInContext("imageUrl({token:'a'})",context);
  assert(changed.includes('confidence=1') && changed.includes('display_scaling=log'));
  const heatmapLegend=vm.runInContext("confidenceLegendHtml({proposal_mode:'heatmap'})",context);
  assert(heatmapLegend.includes('Heatmap probability') && heatmapLegend.includes('continuous'));
  assert(!heatmapLegend.includes('argmax'));
  const ordinalLegend=vm.runInContext("confidenceLegendHtml({proposal_mode:'ordinal'})",context);
  assert(ordinalLegend.includes('pixel argmax') && !ordinalLegend.includes('Heatmap probability'));
  context.showDetect=false;
  await vm.runInContext("toggleViewItem('confidence')",context);
  assert(context.viewConfidence); // No toggle before detection, and no extra inference.
  assert.equal(redraws,1);
  console.log('Confidence toggle and scaling-preserving overlay requests passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
