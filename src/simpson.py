"""Simpson's method of disks on a predicted volume tracing, and the polygon-to-mask rendering used for Dice.

A tracing has 42 points in decoder coordinates (x = column + 1, y = 112 - row): the two ends of the long axis, A1 (index 0) and
A2 (index 41), and the left and right end of each of the 20 chords, ordered [A1, L1..L20, R20..R1, A2]. Chord i has width d_i.

  single plane (EchoNet-Dynamic convention): V = pi/4 * sum_i d_i^2 * L / 20, with each phase's own long-axis length L
  biplane (CAMUS convention):                V = pi/4 * sum_i a_i b_i * h, one disk height h shared by ED and ES, so
                                             EF = 1 - sum_i a_i^ES b_i^ES / sum_i a_i^ED b_i^ED  (a: 4-chamber, b: 2-chamber widths)
"""
import numpy as np
import torch


def chord_widths(P):
    """(..., 42, 2) points -> chord widths (..., 20) and axis length (...). Works on torch tensors and numpy arrays."""
    if torch.is_tensor(P):
        d = (P[..., 1:21, :] - P[..., 21:41, :].flip(-2)).norm(dim=-1)
        return d, (P[..., 0, :] - P[..., 41, :]).norm(dim=-1)
    d = np.linalg.norm(P[..., 1:21, :] - P[..., 21:41, :][..., ::-1, :], axis=-1)
    return d, np.linalg.norm(P[..., 0, :] - P[..., 41, :], axis=-1)


def volume(P):
    """Single-plane method of disks straight from the points, in pixel^3 (or the cube of whatever unit P is in)."""
    d, axis = chord_widths(P)
    return (np.pi / 4.0) * (d ** 2).sum(-1) * axis / 20.0


def ef_single_plane(P_ed, P_es, common_height=False):
    """EF in points from the ED and ES tracings of one view, each phase with its own disk height (EchoNet-Dynamic labels), or
    with common_height=True one height for both phases (CAMUS convention, single view)."""
    if common_height:
        d_ed, _ = chord_widths(P_ed); d_es, _ = chord_widths(P_es)
        return float(100.0 * (1.0 - (d_es ** 2).sum() / (d_ed ** 2).sum()))
    return float(100.0 * (1.0 - volume(P_es) / volume(P_ed)))


def ef_biplane(P4_ed, P4_es, P2_ed, P2_es):
    """EF in points from the 4- and 2-chamber ED and ES tracings with one disk height for both phases (CAMUS labels).
    The common height cancels, so pixel spacings only need to be the same within each view."""
    a_ed, _ = chord_widths(P4_ed); a_es, _ = chord_widths(P4_es)
    b_ed, _ = chord_widths(P2_ed); b_es, _ = chord_widths(P2_es)
    return float(100.0 * (1.0 - (a_es * b_es).sum() / (a_ed * b_ed).sum()))


def outline(P):
    """(42, 2) tracing -> closed outline (40, 2): the chord ends without the two axis ends, which lie on the long axis."""
    return P[1:41]


def contour_to_mask(poly, size=112):
    """Hard mask of a tracing with the convention of EchoNet-Dynamic's released masks: cv2.fillPoly of all 42 points with vertex
    coordinates truncated to integers (this shifts every boundary by about half a pixel; the same truncation is applied here)."""
    import cv2
    corner = np.stack([poly[:, 0] - 0.5, size - poly[:, 1] + 0.5], 1)
    out = np.zeros((size, size), np.uint8)
    cv2.fillPoly(out, [np.trunc(corner).astype(np.int32).reshape(-1, 1, 2)], 1)
    return out.astype(bool)


def to_pixels(P, size=112):
    """Decoder coordinates -> (column, row) image coordinates, for drawing."""
    return np.stack([P[..., 0] - 1.0, size - P[..., 1]], -1)


def contour_to_mask_repo(poly, size=112):
    """Hard mask with the convention of the EchoNet-Dynamic code's own masks: skimage.draw.polygon of the 40 chord ends (the two
    axis ends excluded) with rounded coordinates."""
    import skimage.draw
    q = np.stack([poly[:, 0] - 0.5, size - poly[:, 1] + 0.5], 1)[1:41]
    r, c = skimage.draw.polygon(np.rint(q[:, 1]).astype(int), np.rint(q[:, 0]).astype(int), (size, size))
    out = np.zeros((size, size), bool); out[r, c] = True
    return out
