"""Bounded-memory window reads from C-order NPY entries in compressed NPZ files."""
import zipfile
import numpy as np


def mask_window(path, key, shape, x, y, width, height):
    # ZipExtFile seek decompresses skipped bytes, but never retains the full field.
    # Read complete rows in small batches to avoid thousands of tiny inflate calls.
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):dims,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):dims,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError(f'Unsupported mask NPY version: {version}')
        if tuple(dims)!=tuple(shape) or fortran or dtype.hasobject:
            raise ValueError('Quality mask must be a C-order numeric array with science-image shape')
        if not (0<=x<x+width<=shape[1] and 0<=y<y+height<=shape[0]):
            raise ValueError('Invalid mask window')
        row_bytes=shape[1]*dtype.itemsize
        f.seek(y*row_bytes,1)
        out=np.empty((height,width),bool)
        for offset in range(0,height,64):
            rows=min(64,height-offset)
            buf=f.read(rows*row_bytes)
            if len(buf)!=rows*row_bytes:raise ValueError('Truncated NPZ mask')
            a=np.frombuffer(buf,dtype=dtype).reshape(rows,shape[1])
            out[offset:offset+rows]=a[:,x:x+width]
        return out
