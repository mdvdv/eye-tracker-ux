from __future__ import annotations

from math import atan2, degrees
from scipy.signal import butter, lfilter


def pix_from_deg(h: int, d: int, res: int, size_in_deg: int, verbose: bool = False) -> float:
    """Obtain pixels from visual degrees.

    Args:
        h (int): Monitor height in cm.
        d (int): Distance between monitor and participant in cm.
        res (int): Vertical resolution of the monitor.
        size_in_deg (int): The stimulus size in degrees.
        verbose (bool): If true, print information about the results.

    Returns:
        size_in_px: Size of the stimuli in pixels.
    """
    # Calculate the number of degrees that correspond to a single pixel. This will
    # generally be a very small value, something like 0.03.
    deg_per_px = degrees(atan2(.5*h, d)) / (.5*res)
    if verbose: print("%s degrees correspond to a single pixel" % deg_per_px)
    # Calculate the size of the stimulus in degrees
    size_in_px = size_in_deg / deg_per_px
    if verbose: print("The size of the stimulus is %s pixels and %s visual degrees" \
        % (size_in_px, size_in_deg))

    return size_in_px


def deg_from_pix(h: int, d: int, res: int, size_in_deg: int, verbose: bool = False) -> float:
    """Obtain visual degrees from pixels.

    Args:
        h (int): Monitor height in cm.
        d (int): Distance between monitor and participant in cm.
        res (int): Vertical resolution of the monitor.
        size_in_deg (int): The stimulus size in degrees.
        verbose (bool): If true, print information about the results.

    Returns:
        size_in_px: Size of the stimuli in degrees.
    """
    # Calculate the number of degrees that correspond to a single pixel. This will
    # generally be a very small value, something like 0.03.
    deg_per_px = degrees(atan2(.5*h, d)) / (.5*res)
    if verbose: print("%s degrees correspond to a single pixel" % deg_per_px)
    # Calculate the size of the stimulus in degrees
    size_in_deg  = size_in_deg * deg_per_px
    if verbose: print("The size of the stimulus is %s pixels and %s visual degrees" \
        % (size_in_px, size_in_deg))

    return size_in_deg


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)

    return y