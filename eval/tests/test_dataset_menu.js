// No browser dependencies: exercise dataset grouping and selection against a tiny DOM.
const assert = require('assert').strict;
const fs = require('fs');
const vm = require('vm');
class Element {
  constructor() { this.children = []; this.value = ''; this.open = false; }
  append(...items) { this.children.push(...items); }
  appendChild(item) { this.children.push(item); }
  replaceChildren() { this.children = []; }
  set innerHTML(value) { this.children = []; }
  showModal() { this.open = true; }
  close() { this.open = false; }
}
const elements = new Map();
const document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
  createElement() { return new Element(); },
};
const groups = [
  {label: 'HSC', members: ['hsc_raw', 'hsc_image']},
  {label: 'JWST NIRCam', members: ['jwst', 'jwst_cosmos', 'jwst_abell']},
  {label: 'Sitian', members: ['sitian']},
  {label: 'ZTF', members: ['ztf']},
];
const datasets = Object.fromEntries(groups.flatMap(g => g.members).map(id => [id, {
  id, label: id, enabled: id !== 'ztf', patches: ['p1'], bands: ['F115W', 'F444W'],
  default_bands: ['F444W'], default_patches: ['p1'], default_frames_per_tile: 1,
}]));
const context = vm.createContext({document, selectedDataset: 'hsc_raw',
  optionsCache: {dataset_groups: groups, by_dataset: datasets}, setStatus() {}});
const source = fs.readFileSync(require('path').join(__dirname, '../hsctiles/pages/app.js'), 'utf8');
vm.runInContext(source.slice(source.indexOf('function fillSelect('), source.indexOf('async function loadOptions(')), context);
vm.runInContext('renderDatasetCards()', context);
assert.equal(elements.get('datasetCards').children.length, 4);
assert.equal(elements.get('datasetCards').children[3].disabled, true);
elements.get('datasetCards').children[1].onclick();
assert.equal(elements.get('datasetDialog').open, true);
assert.equal(elements.get('datasetChoices').children.length, 3);
elements.get('datasetChoices').children[2].onclick();
assert.equal(context.selectedDataset, 'jwst_abell');
assert.equal(elements.get('datasetDialog').open, false);
assert.equal(elements.get('framesPerTileInput').disabled, true);
assert.equal(elements.get('bandInput').children.length, 2);
elements.get('datasetCards').children[0].onclick();
assert.equal(elements.get('datasetChoices').children.length, 2);
elements.get('datasetChoices').children[1].onclick();
assert.equal(context.selectedDataset, 'hsc_image');
assert.equal(elements.get('framesPerTileInput').disabled, false);
console.log('Dataset grouping / subtype selection tests passed.');
