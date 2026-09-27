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
  console.log('Show Masks toggle / request parameters passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
