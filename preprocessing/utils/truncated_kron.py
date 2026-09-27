"""Reject ordinary JWST tiles containing a truncated large raw-Kron ellipse.

Uses original Kron geometry regardless of source class, never pixel-label colors.
Centered large-source views are intentionally exempt. HSC is not affected.
"""
import numpy as np

POLICY = dict(version='all_classes_raw_kron_partial_grid_v2',major_min=200.,
              fraction_min=.40,fraction_max=.80,ellipse_vertices=2048,
              centered_stamps_exempt=True,source_class_filter=None)


def polygon_area(p):
    if len(p)<3:return 0.
    # Translation reduces cancellation for large absolute celestial-image coordinates.
    q=p-p[0]
    return abs(float(np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))))/2


def clipped_ellipse_fraction(p,x,y,size):
    total=polygon_area(p)
    if total<=0:return 0.
    for axis,bound,sign in [(0,x,1),(0,x+size,-1),(1,y,1),(1,y+size,-1)]:
        if not len(p):return 0.
        previous=np.roll(p,1,axis=0);inside=sign*(p[:,axis]-bound)>=0
        was_inside=np.roll(inside,1);cross=inside!=was_inside
        out=np.empty((len(p)*2,2));chosen=np.zeros(len(p)*2,bool)
        where=np.flatnonzero(cross)
        t=(bound-previous[where,axis])/(p[where,axis]-previous[where,axis])
        out[2*where]=previous[where]+(p[where]-previous[where])*t[:,None];chosen[2*where]=True
        where=np.flatnonzero(inside);out[2*where+1]=p[where];chosen[2*where+1]=True
        p=out[chosen]
    return min(1.,max(0.,polygon_area(p)/total))


def large_kron_geometry(ids,x,y,a,b,theta):
    """Small raw-catalog subset; preserve centers outside the labeling window."""
    vals={k:np.asarray(v) for k,v in zip(('ids','x','y','a','b','theta'),(ids,x,y,a,b,theta))}
    keep=np.isfinite(vals['x']+vals['y']+vals['a']+vals['b']+vals['theta'])
    keep &= (vals['a']>POLICY['major_min']) & (vals['b']>0)
    return {k:v[keep] for k,v in vals.items()}


def filter_truncated_kron_tiles(labels,specs,origin=(0,0)):
    raw=getattr(labels,'truncation_geometry',None)
    if raw is None:
        raw=large_kron_geometry(labels.source_ids,labels.geom_x,labels.geom_y,
                                labels.geom_major,labels.geom_minor,labels.geom_theta)
    a,b=raw['a'],raw['b']
    keep_sources=np.ones(len(a),bool)
    classes={int(s):int(c) for s,c in zip(labels.source_ids,labels.label_classes)}
    phi=np.linspace(0,2*np.pi,POLICY['ellipse_vertices'],endpoint=False);co=np.cos(phi);si=np.sin(phi)
    ellipses=[]
    for i in np.flatnonzero(keep_sources):
        x,y=raw['x'][i]+origin[0],raw['y'][i]+origin[1];t=raw['theta'][i];c,s=np.cos(t),np.sin(t)
        p=np.column_stack([x+a[i]*co*c-b[i]*si*s,y+a[i]*co*s+b[i]*si*c])
        ellipses.append((i,p,p.min(0),p.max(0)))
    kept=[];rejected=[]
    for spec in specs:
        if getattr(spec,'kind','grid')!='grid':kept.append(spec);continue
        matches=[]
        x,y,size=spec.x0,spec.y0,spec.size
        for i,p,lo,hi in ellipses:
            if hi[0]<=x or hi[1]<=y or lo[0]>=x+size or lo[1]>=y+size:continue
            if lo[0]>=x and lo[1]>=y and hi[0]<=x+size and hi[1]<=y+size:continue
            fraction=clipped_ellipse_fraction(p,x,y,size)
            if POLICY['fraction_min']<=fraction<=POLICY['fraction_max']:
                matches.append(dict(source_id=int(raw['ids'][i]),source_class=classes.get(int(raw['ids'][i])),
                    raw_a=float(a[i]),raw_b=float(b[i]),kron_fraction=fraction))
        if matches:rejected.append(dict(tile=spec.name,x0=x,y0=y,size=size,sources=matches))
        else:kept.append(spec)
    return kept,dict(policy=POLICY,input_tiles=len(specs),kept_tiles=len(kept),rejected_tiles=rejected)
