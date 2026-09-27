"""Stream independently labelled stamps into their owning parent store.

Temporary part stores live outside the discoverable training tree. Dense planes
are copied by chunks; sparse source and overlapping-mask offsets are rebased.
No scientific labels are stitched, recomputed or changed by this operation.
"""
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np
from direct_zarr_preprocessing.zarr_writer import ZarrGroupWriter, write_json


def _meta(root, name):
    return json.loads((root/name/'.zarray').read_text())


def _read(root, name, start=None, stop=None):
    m=_meta(root,name);shape=m['shape'];chunks=m['chunks'];dtype=np.dtype(m['dtype'])
    start=0 if start is None else start;stop=shape[0] if stop is None else stop
    out=np.empty((stop-start,*shape[1:]),dtype=dtype)
    for c in range(start//chunks[0],(stop+chunks[0]-1)//chunks[0]):
        n=min(chunks[0],shape[0]-c*chunks[0]);key='.'.join(map(str,[c]+[0]*(len(shape)-1)))
        part=np.memmap(root/name/key,mode='r',dtype=dtype,shape=(n,*shape[1:]))
        lo=max(start,c*chunks[0]);hi=min(stop,(c+1)*chunks[0]);out[lo-start:hi-start]=part[lo-c*chunks[0]:hi-c*chunks[0]]
    return out


def merge_parent_stores(parts, output, *, attrs, chunk_tiles=20):
    """parts: [(path, (tile-coordinate dx,dy)), ...], ordinary grid first.

    Stage and publish only after all arrays succeed. The caller removes parts
    only after publication. The old store survives an interrupted merge.
    """
    parts=[(Path(p),tuple(shift)) for p,shift in parts];output=Path(output)
    if not parts:return None
    sizes=[_meta(p,'images')['shape'][0] for p,_ in parts];n=sum(sizes)
    keys=[{p.name for p in root.iterdir() if (p/'.zarray').exists()} for root,_ in parts]
    if any(k!=keys[0] for k in keys):raise ValueError('Parent parts have different array schemas')
    sparse={
        'source_offsets':['source_centers','source_ids'],
        'strict_center_only_offsets':['strict_center_only_centers','strict_center_only_ids'],
        'shape_source_offsets':['shape_source_centers','shape_source_values','shape_source_classes','shape_source_ids'],
    }
    special=set(sparse)|{v for vs in sparse.values() for v in vs}|{f'segmentation_overlap_{v}' for v in ['meta','data','offsets','weights']}
    finalattrs=dict(json.loads((parts[0][0]/'.zattrs').read_text()),**attrs)
    finalattrs.update(num_samples=n,parent_packed_stamps=True,chunk_tail_policy='pool_pad_train',sample_kind='mixed',chunk_tiles=int(chunk_tiles))
    provenance=[]
    for (root,shift),size in zip(parts,sizes):
        a=json.loads((root/'.zattrs').read_text())
        provenance.append(dict(start=sum(q['count'] for q in provenance),count=size,coordinate_shift=list(shift),
            **{k:a[k] for k in ['sample_kind','stamp_bounds','label_window','labels_generated_independently','background_tiles','large_source_policy'] if k in a}))
    finalattrs['sample_blocks']=provenance
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.'+output.name+'.merge-',dir=output.parent));backup=output.with_name('.'+output.name+'.previous')
    try:
        writer=ZarrGroupWriter(temp,overwrite=True,attrs=finalattrs)
        # Dense tensors use chunk20; small per-sample metadata remains a single chunk.
        for name in sorted(keys[0]-special):
            m=_meta(parts[0][0],name);shape=(n,*m['shape'][1:]);dense=name=='images' or name.startswith('band_')
            c=min(int(chunk_tiles),max(1,n)) if dense else max(1,n)
            wa=writer.array(name,shape=shape,chunks=(c,*shape[1:]),dtype=np.dtype(m['dtype']))
            offsets=np.r_[0,np.cumsum(sizes)]
            for start in range(0,n,c):
                stop=min(n,start+c);block=np.empty((stop-start,*shape[1:]),dtype=np.dtype(m['dtype']))
                for j,(root,shift) in enumerate(parts):
                    lo=max(start,int(offsets[j]));hi=min(stop,int(offsets[j+1]))
                    if lo>=hi:continue
                    value=_read(root,name,lo-offsets[j],hi-offsets[j])
                    if name in ('tile_x0','tile_y0'):value=value+shift[0 if name=='tile_x0' else 1]
                    block[lo-start:hi-start]=value
                wa.write_chunk((start//c,*([0]*(len(shape)-1))),block)
        for off,names in sparse.items():
            if off not in keys[0]:continue
            arrays={k:[] for k in [off,*names]};cursor=0
            for root,_ in parts:
                arrays[off].append(_read(root,off)+cursor)
                for name in names:arrays[name].append(_read(root,name))
                cursor+=len(arrays[names[0]][-1])
            for name,values in arrays.items():
                a=np.concatenate(values);writer.array(name,shape=a.shape,chunks=(max(1,len(a)),*a.shape[1:]),dtype=a.dtype).write_full(a)
        if 'segmentation_overlap_meta' in keys[0]:
            arrays={k:[] for k in ['meta','data','offsets','weights']};sample_cursor=byte_cursor=0
            for (root,_),size in zip(parts,sizes):
                a={k:_read(root,'segmentation_overlap_'+k) for k in arrays};a['meta'][:,0]+=sample_cursor
                arrays['meta'].append(a['meta']);arrays['data'].append(a['data']);arrays['weights'].append(a['weights']);arrays['offsets'].append(a['offsets'][:-1]+byte_cursor)
                sample_cursor+=size;byte_cursor+=len(a['data'])
            packed={k:np.concatenate(v) for k,v in arrays.items()};packed['offsets']=np.r_[packed['offsets'],np.int64(byte_cursor)]
            from .segmentation_storage import validate_overlap_masks
            image_shape = _meta(parts[0][0], 'images')['shape']
            validate_overlap_masks(packed, n, image_shape[1], *image_shape[-2:])
            for k,a in packed.items():writer.array('segmentation_overlap_'+k,shape=a.shape,chunks=(max(1,len(a)),*a.shape[1:]),dtype=a.dtype).write_full(a)
        if backup.exists():raise RuntimeError(f'Recover previous interrupted publication first: {backup}')
        if output.exists():output.rename(backup)
        try:
            temp.rename(output)
            write_json(Path(str(output)+'_manifest.json'),dict(output=str(output),num_samples=n,bands=finalattrs['bands'],schema=finalattrs['schema'],parent_packed_stamps=True,chunk_tiles=int(chunk_tiles)))
        except BaseException:
            if output.exists():shutil.rmtree(output)
            if backup.exists():backup.rename(output)
            raise
        if backup.exists():shutil.rmtree(backup)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return str(output)


class ParentStoreAssembly:
    """Own bounded-lived temporary parts and publish one final parent Zarr."""
    def __init__(self, task, scratch):
        from .image_level import _store_output_path
        self.output=_store_output_path(task.output_root,task.patch,task.band,task.dataset_source,task.group)
        if list(self.output.parent.glob(task.patch + '_id*.zarr')):
            raise ValueError('Legacy standalone centered stores exist; use a new output root '
                             f'to avoid duplicate training samples: {self.output.parent}')
        self.attrs=dict(patch=task.patch,tract=str(task.tract),bands=[task.band]);self.chunk_tiles=task.chunk_tiles;self.parts=[]
        Path(scratch).mkdir(parents=True,exist_ok=True)
        self.scratch=Path(tempfile.mkdtemp(prefix='parent-parts-',dir=scratch))

    def task(self, original):
        from dataclasses import replace
        return replace(original,output_root=self.scratch,overwrite=True)

    def add(self,result,shift=(0,0)):
        if result.get('output'):self.parts.append((result['output'],shift))

    def finish(self):
        output=merge_parent_stores(self.parts,self.output,attrs=self.attrs,chunk_tiles=self.chunk_tiles)
        audits = [json.loads(p.read_text()) for p in sorted(self.scratch.rglob('*_tile_filter.json'))]
        if audits:
            write_json(Path(str(self.output) + '_tile_filter.json'), dict(parts=audits))
        shutil.rmtree(self.scratch)
        return output
