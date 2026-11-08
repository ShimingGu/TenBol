"""Device discovery for the torch backend.

MIG slices on this machine are shared with other users' processes, so
``torch.cuda.device_count()`` counting 7 devices does not mean 7 are usable.
Measured on one occasion: slice 0 had 7.79 of 10.8 GiB already taken and ran a
2048x2048 matmul benchmark 2.4x slower than the free slices.  Any code that
fans work out across devices must therefore probe first.

Two things are checked per slice:

* free memory via ``torch.cuda.mem_get_info`` -- cheap and permission-free
  (``nvidia-smi --query-compute-apps`` needs privileges this account lacks);
* a short timing probe, because a slice can be memory-free yet compute-busy.

Nothing here assumes a fixed device count.
"""

from __future__ import annotations

import time

import torch

__all__ = ["probe_devices", "usable_devices", "DeviceInfo"]


class DeviceInfo(dict):
    """Plain dict of per-device facts; kept as a class for nicer repr."""

    @property
    def index(self):
        return self["index"]

    @property
    def free_gib(self):
        return self["free_gib"]

    @property
    def ms(self):
        return self["ms"]


def _timing_ms(device: int, size: int = 2048, iters: int = 20) -> float:
    """Short matmul benchmark; returns ms per iteration."""
    x = torch.randn(size, size, device=f"cuda:{device}", dtype=torch.float32)
    for _ in range(2):                     # warm up
        x = (x @ x) * 1.0e-3
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        x = (x @ x) * 1.0e-3
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters * 1e3


def probe_devices(min_free_gib: float = 6.0, do_timing: bool = True) -> list:
    """Describe every visible CUDA device.

    Returns a list of dicts with ``index``, ``name``, ``free_gib``, ``total_gib``,
    ``ms`` (matmul time, ``None`` if not timed) and ``ok`` (enough free memory and
    -- when timed -- within 2x of the fastest slice).

    A probe is a *snapshot*: the neighbours on these slices are other users'
    processes, and a slice observed free can become busy a second later.  The
    default thresholds are deliberately loose (6 of 10.8 GiB must be free) so
    that a slice carrying someone else's context is excluded rather than
    half-used; observed on one occasion a slice holding 7.79 GiB still passed a
    2 GiB test and ran 2.4x slow.
    """
    infos = []
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "free_gib": free / 2 ** 30,
                "total_gib": total / 2 ** 30,
                "ms": _timing_ms(i) if do_timing else None,
            }
            info["ok"] = info["free_gib"] >= min_free_gib
            infos.append(info)
        except Exception as exc:                       # noqa: BLE001 - report, don't crash
            infos.append({"index": i, "name": "?", "free_gib": 0.0, "total_gib": 0.0,
                          "ms": None, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    timed = [d["ms"] for d in infos if d["ms"]]
    if timed:
        fastest = min(timed)
        for d in infos:
            if d["ms"] and d["ms"] > 2.0 * fastest:
                d["ok"] = False                        # compute-busy slice
                d["busy"] = True
    return infos


def usable_devices(min_free_gib: float = 6.0, do_timing: bool = True) -> list:
    """Indices of devices that pass :func:`probe_devices`."""
    return [d["index"] for d in probe_devices(min_free_gib, do_timing) if d["ok"]]
