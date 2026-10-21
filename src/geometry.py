"""Geometry of the LV tracing polygon: chord widths and volume, and the polygon-to-mask rendering used for Dice.

The polygon has 42 points in decoder coordinates (x = column + 1, y = 112 - row): the two ends of the long axis, A1 (index 0)
and A2 (index 41), and the left and right end of each of the 20 chords, in the order [A1, L1..L20, R20..R1, A2].
"""
import numpy as np
import torch


def pixel_grid(size, device):
    """(size*size, 2) pixel coordinates in the 1-based, y-up convention of the polygons."""
    xs, ys = np.arange(1, size + 1), np.arange(size, 0, -1)
    XX, YY = np.meshgrid(xs, ys)
    return torch.tensor(np.stack([XX.ravel(), YY.ravel()], 1), device=device, dtype=torch.float32)


def chord_widths(P):
    """(..., 42, 2) points -> chord widths (..., 20) and axis length (...). Works on torch tensors and numpy arrays."""
    if torch.is_tensor(P):
        d = (P[..., 1:21, :] - P[..., 21:41, :].flip(-2)).norm(dim=-1)
        return d, (P[..., 0, :] - P[..., 41, :]).norm(dim=-1)
    d = np.linalg.norm(P[..., 1:21, :] - P[..., 21:41, :][..., ::-1, :], axis=-1)
    return d, np.linalg.norm(P[..., 0, :] - P[..., 41, :], axis=-1)


def chord_volume(P):
    """Method of disks straight from the points: V = pi/4 * sum_i d_i^2 * (axis length / 20), in pixel^3."""
    d, axis = chord_widths(P)
    return (np.pi / 4.0) * (d ** 2).sum(-1) * axis / 20.0


def soft_membership(poly, grid, tau=0.7):
    """Soft inside-mask of closed polygons. poly (F,P,2), grid (N,2) -> (F,N) in (0,1).

    Signed distance (unsigned distance to the boundary times a winding-number sign) through a sigmoid. A 42-point tracing
    polygon is drawn without its two axis ends (points 0 and 41 lie on the long axis, the outline excludes them).
    """
    if poly.shape[1] == 42:
        poly = poly[:, 1:41]
    a = poly
    b = torch.roll(poly, -1, dims=1)
    p = grid[None, :, None, :]
    a_ = a[:, None, :, :]
    b_ = b[:, None, :, :]

    ab = b_ - a_
    ap = p - a_
    denom = (ab * ab).sum(-1).clamp(min=1e-9)
    t = ((ap * ab).sum(-1) / denom).clamp(0, 1)
    closest = a_ + t[..., None] * ab
    d2 = ((p - closest) ** 2).sum(-1).clamp(min=1e-6)
    d = d2.sqrt().min(dim=-1).values

    v1 = a_ - p
    v2 = b_ - p
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(-1)
    ang = torch.atan2(cross, dot + 1e-8).sum(-1)
    wind = (ang.abs() / (2 * np.pi)).clamp(0, 1)

    return torch.sigmoid(d * (2 * wind - 1) / tau)


def contour_to_mask(poly, size=112):
    """Hard mask of a polygon with the dataset's convention.

    poly (P, 2) in decoder coordinates. The dataset's masks are cv2.fillPoly of the tracing polygon with vertex coordinates
    truncated to integers, which shifts every boundary by about half a pixel; the same truncation is applied here.
    """
    import cv2
    corner = np.stack([poly[:, 0] - 0.5, size - poly[:, 1] + 0.5], 1)
    out = np.zeros((size, size), np.uint8)
    cv2.fillPoly(out, [np.trunc(corner).astype(np.int32).reshape(-1, 1, 2)], 1)
    return out.astype(bool)
