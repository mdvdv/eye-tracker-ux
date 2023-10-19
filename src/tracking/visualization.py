from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .utils import butter_lowpass, butter_lowpass_filter


def plot_gaze(gaze_data):
    """Plot the gaze data.

    Args:
        gaze_data: gaze data in (x_vals, y_vals) format.
    """
    x_data = gaze_data[0]
    y_data = gaze_data[1]
    plt.title("Gaze data", fontsize=20)
    plt.xlabel("x position", fontsize=14)
    plt.ylabel("y position", fontsize=14)
    plt.scatter(x_data, y_data, s=5, color='navy')
    plt.show()


def plot_vels(vels, sacc_thrs=None, fix_thrs=None):
    """Plot the velocity data. Optionally plot the saccade and fixation thresholds.

    Args:
        vels: a list with the velocity data
        sacc_thrs: Velocity threshold in pixels for a point to be considered a saccade.
        fix_thrs: Velocity threshold in pixels for a point to be considered a fixation.
    """
    plt.plot(vels, label='Velocities')
    if sacc_thrs:
        plt.hlines(sacc_thrs, 
                   0, 
                   len(vels), 
                   colors="k", 
                   linestyles="dashed",
                   label="Saccaade threshold")
    if fix_thrs:
        plt.hlines(fix_thrs, 
                   0, 
                   len(vels), 
                   colors="k", 
                   linestyles="dashdot",
                   label="Fixation threshold")
    plt.xlim(0, len(vels))
    plt.title("Velocity between points", fontsize=20)
    plt.xlabel("Point number (n)", fontsize=14)
    plt.ylabel("Point to point velocity (u/sec)", fontsize=14)
    plt.legend()
    plt.show()


def plot_simple(saccades, fixations, centroids):
    flat_fix = [point for fix_group in fixations for point in fix_group]
    flat_saccade = [point for saccade_group in saccades for point in saccade_group]
    fixX, fixY = np.array(flat_fix).T
    saccX, saccY = np.array(flat_saccade).T
    centX, centY = np.array(centroids).T
    plt.title("Saccades, fixations and centroids", fontsize=20)
    plt.xlabel("x position", fontsize=14)
    plt.ylabel("y position", fontsize=14)
    plt.scatter(saccX, saccY, s=5, color="b", label="Saccades")
    plt.scatter(fixX, fixY, s=5, color="g", label="Fixations")
    plt.scatter(centX, centY, s=300, color="r", alpha=0.7, label="Centroids")
    plt.legend(borderpad=1)
    plt.axis("equal")
    plt.show()


def plot_centroids(centroids, centroids_count):
    fig, ax = plt.subplots() 
    x, y = np.array(centroids).T
    # Draw fixation circles
    size = max(centroids_count)/50
    for i in range(len(centroids)):
        ax.add_artist(plt.Circle((x[i], y[i]), centroids_count[i]/size, color="r"))
    # Draw saccade lines
    for i in range(1, len(centroids)):
        plt.plot(x[i-1:i+1], y[i-1:i+1], "b-")
    plt.title("Centroids", fontsize=20)
    plt.xlabel("x position", fontsize=14)
    plt.ylabel("y position", fontsize=14)
    plt.axis("equal")
    plt.show()


def plot_velocity(velocity):
    vel = butter_lowpass_filter(velocity, cutoff=1.2, fs=30, order=5)
    plt.plot(vel, "r", label="Velocity")
    plt.xlim(0, 2000)
    plt.hlines(ET.angle2pix(sac_min_thres), 0, 2000, colors="k", linestyles="dotted", label="Saccade min")
    plt.hlines(ET.angle2pix(fix_max_thres), 0, 2000, colors="k", linestyles="dashed", label="Fixation max")
    plt.legend(fontsize="xx-small", loc="upper right")
    plt.ylabel("Velocity (°/sec)")
    plt.show()


def plot_gaze_data(x, y, subject_id, fixations=None, saccades=None, centroid_x=None, centroid_y=None):
    plt.rcParams["figure.dpi"] = 200
    plt.scatter(x, y, s=0.5, color="blue")
    if fixations is not None:
        fix_x = x[fixations]
        fix_y = y[fixations]
        plt.scatter(fix_x, fix_y, s=0.5, color="red", label="Fixation")
        sac_x = x[saccades]
        sac_y = y[saccades]
        plt.scatter(sac_x, sac_y, s=0.5, color="blue", label="Saccades")
        plt.scatter(centroid_x, centroid_y, s=50, alpha=0.6, color="orange", label="Centroids")
        plt.legend(borderpad=1)
        plt.title("Gaze Data with Fixations")
    else: 
        plt.title("Gaze Data: " + subject_id)

    plt.xticks(range(-700, 800, 100))
    plt.yticks(range(-700, 800, 100))
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()


def plot_relative_position(x, y):
    plt.plot(x[0:5000], "r", label="x-position")
    plt.plot(y[0:5000], "b", label="y-position")
    plt.ylabel("Relative Position")
    plt.legend(fontsize="xx-small", loc="upper right")
    plt.show()


def plot_saccades_prob(x, saccades):
    is_saccades = [ True if (i in saccades) else False for i in range(len(x))]
    plt.ylim(0, 1.5)
    plt.step(is_saccades[0:5000], "r", linewidth = 0.5, where="post", label="Saccade Probability")
    plt.legend(fontsize="xx-small", loc="upper right")
    plt.show()


def plot_saccade_amplitude(x, saccades, saccades_amplitude):
    amplitude = [saccades_amplitude[saccades.index(i)] if (i in saccades) else 0 for i in range(len(x))]
    plt.plot(amplitude, "r", linewidth = 0.5, label="Saccades Amplitude")
    plt.legend(fontsize="xx-small", loc="upper right")
    plt.xlabel("Data Points")
    plt.ylabel("Amplitude (°)")
    plt.show()


def plot_centroid_duration(centroid_x, centroid_y, duration, mean_duration):
    plt.plot(centroid_x, centroid_y, linewidth = 0.5, label="Saccades")
    plt.scatter(centroid_x, centroid_y, s=duration, alpha=0.6, color="orange", label="Centroids (size is duration in ms)")

    plt.legend(borderpad=1, fontsize="xx-small", markerscale=0.5)
    plt.xticks(range(-700, 800, 100))
    plt.yticks(range(-700, 800, 100))
    plt.title("Gaze Data with Fixations Mean Duration: " + str("{:.2f}".format(mean_duration)) + "ms")
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()


def plot_MFD_Subject(sid, MFD, MFD_SD, text):
    plt.errorbar(sid, MFD, MFD_SD, linestyle="None", fmt="-o", label="mean & std", capsize=2, ecolor="red")
    plt.legend(fontsize="xx-small", loc=4)
    plt.xlabel("Subjects")
    plt.ylabel(text)
    plt.show()


def plot_MSA_MFD(sid, MFD, MSA, text):
    plt.scatter(MFD, MSA, s=2, color="r", label="MFD-MSA (" + text + ")")
    for i, txt in enumerate(sid):
        plt.annotate(txt, (MFD[i]+0.3, MSA[i]), size=5)

    plt.legend(fontsize="xx-small", loc=4)
    plt.title("MSA "+ text + " x MFD " + text)
    plt.tick_params(axis="both", which="major", labelsize=5)
    plt.xlabel("MFD " + text + " (ms)")
    plt.ylabel("MSA " + text + " (°)")
    plt.show()
