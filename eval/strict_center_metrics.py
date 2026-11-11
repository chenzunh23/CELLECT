"""One-to-one point metrics including sparse strict centers (no mask filtering)."""
import numpy as np
from eval.matching import greedy_match


def point_metrics(pred, base, strict, strict_ids, radius=3.):
    pred=np.asarray(pred,float).reshape(-1,2)
    base=np.asarray(base,float).reshape(-1,2)
    strict=np.asarray(strict,float).reshape(-1,2)
    strict_ids=np.asarray(strict_ids,np.int64)
    if len(strict)!=len(strict_ids):raise ValueError('strict IDs/coordinates length mismatch')
    # Remove exact coordinate duplicates, prioritizing original shape-supervised GT.
    seen=set(map(tuple,base));unique=[]
    for i,q in enumerate(strict):
        if tuple(q) not in seen:unique.append(i);seen.add(tuple(q))
    duplicates=len(strict)-len(unique)
    strict=strict[unique];strict_ids=strict_ids[unique]
    combined=np.concatenate([base,strict]);rows=lambda a:[dict(x=float(x),y=float(y)) for x,y in a]
    _,_,base_matches=greedy_match(rows(pred),rows(base),radius=radius)
    _,_,matches=greedy_match(rows(pred),rows(combined),radius=radius)
    hit={g for _,g,_ in matches};n=len(base)
    # Same namespace as uncover_labels compact-center insertion: uint32 scope + 6-digit debug ID.
    inserted=(strict_ids<=-8_000_000_000_000_000)&(strict_ids>=-(8_000_000_000_000_000+(2**32)*1_000_000-1))
    counts=dict(pred=len(pred),base_gt=n,base_tp=len(base_matches),strict_gt=len(strict),
        strict_tp=sum(n+i in hit for i in range(len(strict))),inserted_gt=int(inserted.sum()),
        inserted_tp=sum(n+i in hit for i in np.flatnonzero(inserted)),combined_gt=len(combined),combined_tp=len(matches),
        combined_base_tp=sum(g<n for g in hit),strict_exact_duplicates_removed=duplicates)
    return counts,matches,combined


def finalize(counts):
    out=dict(counts)
    for prefix in ('base','strict','inserted','combined'):
        gt=counts.get(prefix+'_gt',0);tp=counts.get(prefix+'_tp',0)
        out[prefix+'_completeness']=tp/gt if gt else None
    for prefix in ('base','combined'):
        tp=counts.get(prefix+'_tp',0);n=counts.get('pred',0)
        p=tp/n if n else 0.;r=out[prefix+'_completeness'] or 0.
        out[prefix+'_purity']=p;out[prefix+'_f1']=2*p*r/(p+r) if p+r else 0.
    return out
