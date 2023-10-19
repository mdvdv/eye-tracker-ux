from __future__ import annotations

import numpy as np
from math import atan2, degrees, sqrt
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff=1.2, fs=30, order=2):
    """Returns the result of the filtering of the input data.

    Args:
        data: The data to be filtered.
        cutoff: The cutoff frequency.
        fs: The sample rate.
        order: The order of the filter.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)

    return y


def get_vels(points, sample_rate):
    """Function for geting the velocities between the points in a list of gaze data.

    Args:
        points: Eye tracking data points in the form of (x_values, y_values).
        sample_rate: Sample rate in which the gaze data was recorded.
    """
    velocities = [] # List to store velocities
    x_data = points[0]
    y_data = points[1]
    # Calculate velocities between all the points
    for i in range(len(x_data) - 1):
        # Velocity (sqrt((x2-x1)^2+(y2-y1)^2)/time), with 1/time = sample rate
        velocities.append(
            sqrt(((x_data[i + 1] - x_data[i]) ** 2 + (y_data[i + 1] - y_data[i]) ** 2))
            * sample_rate
        )

    return velocities


def I_VT_alg(points, sample_rate, up_sacc_thrs, sacc_thrs, fix_thrs, filt=True):
    """Implementation of the Velocity-Threshold Identification (I-VT) algorithm.

    Args:
        points: Eye tracking data points in the form of (x_values, y_values).
        sample_rate: Sample rate for calculating the velocities.
        sacc_thrs: Velocity threshold in pixels for a point to be considered a saccade.
        fix_thrs: Velocity threshold in pixels for a point to be considered a fixation.
        filt: When true, the velocity data is passed through a low pass filter
    
    Returns:
        saccades: A list with the points corresponding to saccades.
        fixations: A list with the groups of points corresponding to fixation groups.
        centroids: A list with the centroids of each fixation group.
        centroids_count: A list with the number of points correspondent to each centroid.
    """
    saccades = [] # List to store saccades
    fixations = [] # List to store fixations
    fix_group = [] # Buffer list to store fixation groups
    saccade_group = [] # Buffer list to store saccade groups
    x_data = points[0]
    y_data = points[1]
    vels = get_vels(points, sample_rate)
    if filt:
        vels = butter_lowpass_filter(vels)
    # Saccades and fixations calculation
    for i in range(len(vels)):
        # If velocity over saccade threshold, add point to saccades
        if (vels[i] >= sacc_thrs and vels[i] <= up_sacc_thrs):
            # If saccade detected and fix_group buffer contains points, add the
            # group in the buffer to fixations and clear buffer
            if fix_group:
                fixations.append(fix_group.copy())
                fix_group.clear()
            saccade_group.append([x_data[i], y_data[i]])
        # If velocity below fixation threshold, add point to fixations
        elif vels[i] <= fix_thrs:
            # If fixation detected and saccade_group buffer contains points, add the
            # group in the buffer to saccades and clear buffer
            if saccade_group:
                saccades.append(saccade_group.copy())
                saccade_group.clear()
            fix_group.append([x_data[i], y_data[i]])
    # If fix_group or saccade_group buffers are not empty after
    # transversing all the points,add group in buffer to fixations
    if fix_group:
        fixations.append(fix_group)
    if saccade_group:
        saccades.append(saccade_group)
    # Centroids calculation
    centroids = []
    centroids_count = []
    # For all fixation groups, calculate the middle point 
    for group in fixations:
        group = np.array(group)
        group = group.T
        centroids.append([group[0].mean(), group[1].mean()])
        centroids_count.append(len(group[0]))

    return (saccades, fixations, centroids, centroids_count)


def get_dists(points):
    """Function for geting the distances between the points in a list of gaze data.

    Args:
        points: Eye tracking data points in the form of (x_values, y_values).
        sample_rate: Sample rate in which the gaze data was recorded.
    """
    distances = []
    x_data = points[0]
    y_data = points[1]
    for i in range(len(x_data) - 1):
        distances.append(
            sqrt(((x_data[i + 1] - x_data[i]) ** 2 + (y_data[i + 1] - y_data[i]) ** 2))
        )

    return distances


def I_DT_alg(points, disp_thrs, min_count):
    saccades = [[]]
    fixations = []
    centroids = []
    centroids_count = []
    x_data, y_data, = points
    dists = get_dists(points)
    count = 0
    i = 0
    while i < len(dists):
        current = dists[i:i+min_count]
        if len(current) < min_count or sum(current) > disp_thrs:
            saccades[-1].append([x_data[i], y_data[i]])
            i += 1
        else:
            missing = dists[i+min_count:]
            for j in range(len(missing)):
                if (sum(current) + sum(missing[:j+1])) > disp_thrs:
                    break
            x, y = x_data[i:i+min_count+j], y_data[i:i+min_count+j]
            fixations.append(np.array([x,y]).T.tolist())
            centroids.append([np.mean(x), np.mean(y)])
            centroids_count.append(len(x))
            if len(saccades[-1]):
                saccades.append([])
            i += min_count + j
    fixations = np.array(fixations).T

    return (saccades, fixations, centroids, centroids_count)


class EyeTracker(object):
    def __init__(self, sampling_rate, distance, height, res_height, sac_min_thres, sac_max_thres, fix_max_thres, Vel_filter=None):
        self.sampling_rate = sampling_rate   # sampling rate = 1/T in Hz

        # Setup
        self.distance   = distance             # mm
        self.height     = height               # mm
        self.res_height = res_height           # mm

        # Calculating fixation and saccades threshold in pixel
        self.sac_min_thres = self.angle2pix(sac_min_thres)
        self.sac_max_thres = self.angle2pix(sac_max_thres)
        self.fix_max_thres = self.angle2pix(fix_max_thres)

        # Velocity FIR filter (two-tap filter as default)
        self.Vel_filter = Vel_filter
        if self.Vel_filter is None: self.Vel_filter = [1, -1]
        self.Vel_filter = [i*self.sampling_rate for i in self.Vel_filter]

        print("sac min: ", self.sac_min_thres, "\nSac max: ", self.sac_max_thres, "\nFix max: ", self.fix_max_thres)

    # Method for converting visual angle to pixel
    def angle2pix(self, value):
        return value / (degrees(atan2(.5*self.height, self.distance)) / (.5*self.res_height))

    def pix2angle(self, value):
        return value * (degrees(atan2(.5*self.height, self.distance)) / (.5*self.res_height))

    # Fixation detection Method
    def detect_fixation(self, x, y, filter_data=False):
        self.x = x
        self.y = y

        N = len(x)                # Size of data
        K = len(self.Vel_filter)  # Size of velocity FIR filter

        # Calculating velocity x and y
        velocity_x = [0 for i in range (N-K-1)]
        velocity_y = [0 for i in range (N-K-1)]

        for i in range (N-K-1):
            for j in range (K):
                velocity_x[i] = velocity_x[i] + (x[i+j]*self.Vel_filter[j])
                velocity_y[i] = velocity_y[i] + (y[i+j]*self.Vel_filter[j])

        velocity = [((velocity_x[i] ** 2 + velocity_y[i] ** 2)**0.5) for i in range (N-K-1)]

        # Passing data through a low-pass filter to eliminate noise
        if filter_data:
            velocity = butter_lowpass_filter(velocity, cutoff=1.2, fs=30, order=5)

        # Detection of fixation and saccades
        self.fixations_index = []
        self.saccades_index = []
        fixation_group = []
        saccades_group = []

        for i in range (N-K-1):
            # Saccade detected
            if(velocity[i] >= self.sac_min_thres and velocity[i] <= self.sac_max_thres):
                if(fixation_group):
                    self.fixations_index.append(fixation_group)
                    fixation_group = []
                saccades_group.append(i)

            # Fixation detected
            elif(velocity[i] <= self.fix_max_thres):
                if(saccades_group):
                    self.saccades_index.append(saccades_group)
                    saccades_group = []
                fixation_group.append(i)

        if(fixation_group):
            self.fixations_index.append(fixation_group)
        if(saccades_group):
            self.saccades_index.append(saccades_group)

        return (self.fixations_index, self.saccades_index, velocity)

    # Get centroids x and y of each fixation groups
    def get_centroids(self):
        centroid_x = []
        centroid_y = []

        for group in self.fixations_index:
            centroid_x.append(sum([self.x[i] for i in group])/len(group))
            centroid_y.append(sum([self.y[i] for i in group])/len(group))
        return (centroid_x, centroid_y)

    # Get average duration and durations of each fixation groups in ms
    def get_durations(self):
        duration = []
        for fixation in self.fixations_index:
            duration.append((len(fixation)/self.sampling_rate) * 1000) # Duration in ms
        return (numpy.mean(duration), duration)

    def get_saccades_amplitude(self):
        saccades   = [index for sublist in self.saccades_index for index in sublist]
        amplitudes = []
        for i in saccades:
            amplitudes.append(self.pix2angle((self.x[i] ** 2 + self.y[i] ** 2)**0.5))
        return amplitudes
