"""Map local output channels to physical filters for independent image stores."""
import numpy as np


def sample_band_name(batch, item_idx, band_idx, fallback):
    names = batch.get("band_names")
    names = names[item_idx] if names is not None else None
    if names:
        if not 0 <= band_idx < len(names):
            raise ValueError("Output channel does not match sample band metadata")
        return str(names[band_idx])
    return str(fallback[band_idx]) if band_idx < len(fallback) else str(band_idx)


def valid_predictions(predictions, batch, item_idx, band_idx):
    """Exclude centers on NO DATA, without excluding ordinary ignore sources."""
    xy = np.asarray(predictions, dtype=np.float32).reshape(-1, 2)
    masks = batch.get("band_valid_mask")
    if masks is None or not len(xy):
        return xy
    mask = masks[item_idx][band_idx]
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    finite = np.isfinite(xy).all(axis=1)
    ij = np.rint(np.where(np.isfinite(xy), xy, -1)).astype(np.int64)
    good = finite & (ij[:, 0] >= 0) & (ij[:, 0] < mask.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < mask.shape[0])
    indices = np.flatnonzero(good)
    good[indices] &= np.asarray(mask, dtype=bool)[ij[indices, 1], ij[indices, 0]]
    return xy[good]
