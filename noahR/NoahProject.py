

import pandas as pd
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import modal
import modal.onsetdetection as od
import modal.ui.plot as trplot

from sklearn.cluster import KMeans

file_name = 'OpenCloseDoor.wav'

sampling_rate, temp1 = wf.read(file_name)
temp2 = pd.DataFrame(temp1)

# choose start and stop times to analyze
startTime, stopTime = 40, 65
audioClip = temp2.iloc[sampling_rate*startTime:sampling_rate*stopTime, :]

# ================     ONSETS     ================

# collect the audio channels to be analyzed

audio1 = audioClip[0] # Channel 1
audio1 = np.asarray(audio1, dtype=np.double)
audio1 /= np.max(audio1)

audio2 = audioClip[1] # Channel 2
audio2 = np.asarray(audio2, dtype=np.double)
audio2 /= np.max(audio2)

audio3 = (audio1 + audio2)/2 # Channel 3 is an element-wise mean of channels 1&2

audio = [audio1, audio2, audio3]

# ================     detect onsets     ================
frame_size = 2048
hop_size = 512

# Channel 1 onsets:
odf = modal.ComplexODF()
odf.set_hop_size(hop_size)
odf.set_frame_size(frame_size)
odf.set_sampling_rate(sampling_rate)
odf_values = np.zeros(len(audio1) / hop_size, dtype=np.double)
odf.process(audio1, odf_values)

onset_det = od.OnsetDetection()
onset_det.peak_size = 3
onsets1 = onset_det.find_onsets(odf_values) * odf.get_hop_size()

# Channel 2 onsets:
odf = modal.ComplexODF()
odf.set_hop_size(hop_size)
odf.set_frame_size(frame_size)
odf.set_sampling_rate(sampling_rate)
odf_values = np.zeros(len(audio2) / hop_size, dtype=np.double)
odf.process(audio2, odf_values)

onset_det = od.OnsetDetection()
onset_det.peak_size = 3
onsets2 = onset_det.find_onsets(odf_values) * odf.get_hop_size()

# Channel 3 onsets:
odf = modal.ComplexODF()
odf.set_hop_size(hop_size)
odf.set_frame_size(frame_size)
odf.set_sampling_rate(sampling_rate)
odf_values = np.zeros(len(audio3) / hop_size, dtype=np.double)
odf.process(audio3, odf_values)

onset_det = od.OnsetDetection()
onset_det.peak_size = 3
onsets3 = onset_det.find_onsets(odf_values) * odf.get_hop_size()

# array of onsets, by time in secconds
onsets = [onsets1/np.float(sampling_rate), onsets2/np.float(sampling_rate), onsets3/np.float(sampling_rate)]

onsetStart = np.concatenate((onsets1, np.delete(onsets2, 5), onsets3))
onsetStart = np.sort(onsetStart)
# ================     plot onset detection results     ================

f = 1 # index of figure number

# Channel 1
fig = plt.figure(f, figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.title('Channel 1 Onset detection with ' + odf.__class__.__name__)
plt.plot(audio1, '0.4')
plt.subplot(3, 1, 2)
trplot.plot_detection_function(onset_det.odf, hop_size)
trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
plt.subplot(3, 1, 3)
trplot.plot_onsets(onsets1, 1.0)
plt.plot(audio1, '0.4')
plt.show()

f += 1

# Channel 2
fig = plt.figure(f, figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.title('Channel 2 Onset detection with ' + odf.__class__.__name__)
plt.plot(audio2, '0.4')
plt.subplot(3, 1, 2)
trplot.plot_detection_function(onset_det.odf, hop_size)
trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
plt.subplot(3, 1, 3)
trplot.plot_onsets(onsets2, 1.0)
plt.plot(audio2, '0.4')
plt.show()

f += 1

# Channel 3
fig = plt.figure(f, figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.title('Channel 1&2 Mean Onset detection with ' + odf.__class__.__name__)
plt.plot(audio3, '0.4')
plt.subplot(3, 1, 2)
trplot.plot_detection_function(onset_det.odf, hop_size)
trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
plt.subplot(3, 1, 3)
trplot.plot_onsets(onsets3, 1.0)
plt.plot(audio3, '0.4')
plt.show()

f += 1

# Visually Compare onsets of the three channels
fig = plt.figure(f, figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.title('Channels 1, 2, and 1&2Mean Onset detection with ' + odf.__class__.__name__)
trplot.plot_onsets(onsets1, 1.0)
plt.plot(audio1, '0.4')
plt.subplot(3, 1, 2)
trplot.plot_onsets(onsets2, 1.0)
plt.plot(audio2, '0.4')
plt.subplot(3, 1, 3)
trplot.plot_onsets(onsets3, 1.0)
plt.plot(audio3, '0.4')
plt.show()

f += 1

# ================     REVERSE ONSETS     ================

# collect and reverse the audio channels to be analyzed
audio1R = audioClip[0][::-1] # Channel 1R
audio1R = np.asarray(audio1R, dtype=np.double)
audio1R /= np.max(audio1R)

audio2R = audioClip[1][::-1] # Channel 2R
audio2R = np.asarray(audio2R, dtype=np.double)
audio2R /= np.max(audio2R)

audio3R = (audio1R + audio2R)/2 # Channel 3R is an element-wise mean of channels 1R&2R

audioR = [audio1R, audio2R, audio3R] 

frame_size = 2048*10
hop_size = 512

# Channel 1 Reverse onsets:
odf = modal.LPEnergyODF()
odf.set_hop_size(hop_size)
odf.set_frame_size(frame_size)
odf.set_sampling_rate(sampling_rate)
odf_values = np.zeros(len(audio1R) / hop_size, dtype=np.double)
odf.process(audio1R, odf_values)

onset_det = od.OnsetDetection()
onset_det.peak_size = 3
onsets1R = onset_det.find_onsets(odf_values) * odf.get_hop_size()

# Channel 2 Reverse onsets:
odf = modal.LPEnergyODF()
odf.set_hop_size(hop_size)
odf.set_frame_size(frame_size)
odf.set_sampling_rate(sampling_rate)
odf_values = np.zeros(len(audio2R) / hop_size, dtype=np.double)
odf.process(audio2R, odf_values)

onset_det = od.OnsetDetection()
onset_det.peak_size = 3
onsets2R = onset_det.find_onsets(odf_values) * odf.get_hop_size()

# Channel 3 Reverse onsets:
odf = modal.LPEnergyODF()
odf.set_hop_size(hop_size)
odf.set_frame_size(frame_size)
odf.set_sampling_rate(sampling_rate)
odf_values = np.zeros(len(audio3R) / hop_size, dtype=np.double)
odf.process(audio3R, odf_values)

onset_det = od.OnsetDetection()
onset_det.peak_size = 3
onsets3R = onset_det.find_onsets(odf_values) * odf.get_hop_size()

# array of Reverse onsets, by time in secconds
reverseOnsets = [onsets1R/np.float(sampling_rate), onsets2R/np.float(sampling_rate), onsets3R/np.float(sampling_rate)]

onsetStop = np.concatenate((onsets1R, onsets2R, onsets3R))
onsetStop = np.sort(onsetStop)
# ================     plot onset detection results     ================

# Channel 1
fig = plt.figure(f, figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.title('Channel 1 Reverse-Onset detection with ' + odf.__class__.__name__)
plt.plot(audio1R, '0.4')
plt.subplot(3, 1, 2)
trplot.plot_detection_function(onset_det.odf, hop_size)
trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
plt.subplot(3, 1, 3)
trplot.plot_onsets(onsets1R, 1.0)
plt.plot(audio1R, '0.4')
plt.show()

f += 1

# Channel 2
fig = plt.figure(f, figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.title('Channel 2 Reverse-Onset detection with ' + odf.__class__.__name__)
plt.plot(audio2R, '0.4')
plt.subplot(3, 1, 2)
trplot.plot_detection_function(onset_det.odf, hop_size)
trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
plt.subplot(3, 1, 3)
trplot.plot_onsets(onsets2R, 1.0)
plt.plot(audio2R, '0.4')
plt.show()

f += 1

# Channel 3
fig = plt.figure(f, figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.title('Channel 1&2Mean Reverse-Onset detection with ' + odf.__class__.__name__)
plt.plot(audio3R, '0.4')
plt.subplot(3, 1, 2)
trplot.plot_detection_function(onset_det.odf, hop_size)
trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
plt.subplot(3, 1, 3)
trplot.plot_onsets(onsets3R, 1.0)
plt.plot(audio3R, '0.4')
plt.show()

f += 1

# Visually Compare onsets of the three channels
fig = plt.figure(f, figsize=(10, 12))
plt.subplot(3, 1, 1)
plt.title('Channels 1, 2, and 1&2Mean Reverse-Onset detection with ' + odf.__class__.__name__)
trplot.plot_onsets(onsets1R, 1.0)
plt.plot(audio1R, '0.4')
plt.subplot(3, 1, 2)
trplot.plot_onsets(onsets2R, 1.0)
plt.plot(audio2R, '0.4')
plt.subplot(3, 1, 3)
trplot.plot_onsets(onsets3R, 1.0)
plt.plot(audio3R, '0.4')
plt.show()

f += 1

# =============     PREDICT START AND END OF ONSETS by k-n classifer     =============

# set domain and range for clustering algorithm
x, y = np.arange(len(onsetStart)), onsetStart # forward onsets

xR, yR = np.arange(len(onsetStop)), onsetStop # reverse onsets

# elementwise consolidation of domain and range for KMeans
X = np.column_stack((x, y)) # forward onsets
XR = np.column_stack((xR, yR)) # reverse onsets

# run KMeans (clusters chosen by inspection of the audio data)

# forward onsets
est = KMeans(n_clusters=7, init='random')
est.fit(X)
centers = est.cluster_centers_

# reverse onsets
est = KMeans(n_clusters=4, init='random')
est.fit(XR)
centersR = est.cluster_centers_

# split up the centroid data for easy plotting & convert to sampling_rate space

# forward onsets
a, b = np.hsplit(centers, 2)
a, b = a.flatten(), b.flatten()
startSpots = (b).astype(int)
startSpots = np.sort(startSpots)

# reverse onsets
aR, bR = np.hsplit(centersR, 2)
aR, bR = aR.flatten(), bR.flatten()
endSpots = (bR).astype(int)
endSpots = np.sort(endSpots)

# =============     PLOT AUDIO WITH PREDICTED START AND STOP LOCATIONS     =============

# scatter-plot the onset times                           
fig = plt.figure(f, figsize=(10, 5))
plt.title('Predicted onset start and stops')
plt.scatter(a, b, c='#FF0054', marker='+', s=500)
plt.scatter(aR, bR)

f += 1

# plot onset times onto audio data
fig = plt.figure(f, figsize=(10, 5))
plt.title('audio sample with onset start and stops')
plt.plot(audio1, '0.4')
plt.vlines(startSpots, -1, 1, linestyles=u'dashed')
plt.vlines(endSpots, -1, 1, linestyles=u'solid')

f += 1

