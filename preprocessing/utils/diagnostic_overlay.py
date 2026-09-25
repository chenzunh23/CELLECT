"""Diagnostic colors matching the 2026-09-08 Abell dense/confidence overlays."""
import numpy as np

DENSE_RGBA = {
    1: (0.00, 0.82, 0.32, 0.52),  # clean
    2: (0.00, 0.85, 1.00, 0.48),  # weak shape
    3: (1.00, 0.00, 0.00, 0.10),  # ordinary ignore
    4: (0.12, 0.36, 1.00, 0.16),  # background: highly transparent blue
    5: (1.00, 0.66, 0.00, 0.38),  # center-only
    6: (1.00, 0.66, 0.00, 0.38),  # bright region
    7: (1.00, 0.00, 0.00, 0.10),  # strict ignore
}
CONFIDENCE_RGBA = {
    1: (0.12, 0.40, 1.00, 0.44), 2: (0.00, 0.85, 0.25, 0.52),
    3: (1.00, 0.82, 0.00, 0.62), 4: (1.00, 0.00, 0.55, 0.92),
}
SOURCE_COLORS = {1:'#00d152',2:'#00d9ff',3:'#ff0000',4:'#ffa800',5:'#ffa800',6:'#ff0000'}


def rgba(array, colors):
    result=np.zeros((*array.shape,4),np.float32)
    for key,color in colors.items():result[array==key]=color
    return result


def zscale_display(image):
    """Use the reference plot's ZScale, rather than bright-source percentiles."""
    from astropy.visualization import ZScaleInterval
    finite=image[np.isfinite(image)]
    if not finite.size:
        return np.zeros(image.shape,np.float32)
    sample=finite[::max(1,finite.size//500_000)]
    try:
        lo,hi=ZScaleInterval(contrast=.25,krej=2.5,max_iterations=5).get_limits(sample)
    except (ValueError,IndexError):
        lo,hi=np.percentile(sample,[.5,99.7])
    if not np.isfinite(lo+hi) or hi<=lo:
        lo,hi=np.percentile(sample,[.5,99.7])
    if hi<=lo:
        return np.full(image.shape,.5,np.float32)
    safe=np.nan_to_num(image,nan=lo,posinf=hi,neginf=lo)
    return np.clip((safe-lo)/(hi-lo),0,1).astype(np.float32)


def plot_diagnostic(path, name, stamp_name, stamp, dense, confidence, geometry, bounds):
    """geometry columns: local x,y, original a,b,theta(rad), SourceClass."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse,Patch
    display=zscale_display(stamp)
    fig,axes=plt.subplots(1,2,figsize=(14,7.8))
    fig.subplots_adjust(left=.055,right=.99,bottom=.15,top=.85,wspace=.12)
    x0,y0,x1,y1=bounds
    for ax in axes:
        ax.imshow(display,origin='lower',cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        for x in range((x0//4096+1)*4096,x1,4096):ax.axvline(x-x0,color='orange',ls='--',lw=1.5)
        for y in range((y0//4096+1)*4096,y1,4096):ax.axhline(y-y0,color='orange',ls='--',lw=1.5)
        ax.set(xlim=(-.5,511.5),ylim=(-.5,511.5),xlabel='stamp x [pixel]',ylabel='stamp y [pixel]')
    axes[0].imshow(rgba(dense,DENSE_RGBA),origin='lower',interpolation='nearest')
    axes[0].imshow(rgba(confidence,CONFIDENCE_RGBA),origin='lower',interpolation='nearest')
    axes[0].set_title('Dense targets + PSF-EE confidence core')
    for x,y,a,b,theta,cls in geometry:
        color=SOURCE_COLORS[int(cls)]
        axes[1].add_patch(Ellipse((x,y),2*a,2*b,angle=np.rad2deg(theta),fill=False,ec=color,lw=.8,alpha=.85))
        axes[1].plot(x,y,'+',color=color,ms=3)
    axes[1].set_title('Original source Kron + center')
    handles=[Patch(color=DENSE_RGBA[k][:3],label=label) for k,label in
             [(1,'clean'),(2,'weak shape'),(6,'bright / center-only'),(3,'ignore'),(4,'background')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.052),ncol=5,frameon=False,fontsize=10)
    fig.text(.5,.025,'09-08 ZScale (contrast=.25); alpha: clean .52 | weak .48 | bright .38 | ignore .10 | background .16',ha='center',fontsize=9)
    fig.suptitle(f'{name} | {stamp_name}\nOrange dashed: 4096 parent boundary; labels generated in an independent window',y=.975)
    fig.savefig(path,dpi=130);plt.close(fig)
