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

sr = sampling_rate

# choose start and stop times to analyze
startTime, stopTime = 40, 65
netTime = stopTime - startTime
audioClip = temp2.iloc[sr*startTime:sr*stopTime, :]

# ================     ONSETS     ================

# collect and normalize audio channels 1 & 2
audio1, audio2 = audioClip[0], audioClip[1]
audio1, audio2 = np.asarray(audio1, dtype=np.double), np.asarray(audio2, dtype=np.double)
audio1 /= np.max(audio1)
audio2 /= np.max(audio2)

audio3 = (audio1 + audio2)/2 # Channel 3 is an element-wise mean of channels 1&2

audio = [audio1, audio2, audio3] # an array of the three channels being analyzed

# ================     REVERSE ONSETS     ================

# collect, reverse, and normalize reverse audio channels 1 & 2
audio1R, audio2R = audioClip[0][::-1], audioClip[1][::-1]
audio1R, audio2R = np.asarray(audio1R, dtype=np.double), np.asarray(audio2R, dtype=np.double)
audio1R /= np.max(audio1R)
audio2R /= np.max(audio2R)

audio3R = (audio1R + audio2R)/2 # Channel 3R is an element-wise mean of channels 1R&2R

audioR = [audio1R, audio2R, audio3R] # an array of the three channels being analyzed

# ================     detect onsets     ================

# define a function to calculate onset of an array using modal.ComplexODF()

def modalOnsets(frameSize, hopSize, inputArray, modalFunction, peakSize):
	frame_size = frameSize
	hop_size = hopSize
	lenArray = len(inputArray)

	odf = modalFunction()
	odf.set_hop_size(hop_size)
	odf.set_frame_size(frame_size)
	odf.set_sampling_rate(sampling_rate)
	odf_values = np.zeros(lenArray / hop_size, dtype=np.double)
	odf.process(inputArray, odf_values)

	onset_det = od.OnsetDetection()
	onset_det.peak_size = peakSize
	onsets = onset_det.find_onsets(odf_values) * odf.get_hop_size()

	return onsets, odf, onset_det, hop_size

# set the frame, hop, and peak sizes for the onsets
frameSize     = 2048
hopSize       = 512
onsetFunction = modal.ComplexODF
peakSize      = 3

frameSizeR     = 2048*10
hopSizeR       = 512
onsetFunctionR = modal.LPEnergyODF
peakSizeR      = 3

# detect onsets and reverseOnsets for audio Channels 1, 2, and 3:
onsets, onsetsR = {}, {}   # initiate dictionary to store onset and reverse onset arrays
i = 0                      # initiate i counter


# store onset detections in the dictionary onsets

while i < 3:
    x  = modalOnsets(frameSize, hopSize, audio[i], onsetFunction, peakSize)
    xR = modalOnsets(frameSizeR, hopSizeR, audioR[i], onsetFunctionR, peakSizeR)
    onsets[i]  = x[0]
    onsetsR[i] = xR[0]
    i += 1

# remove outlier at onsets[1][5]
onsets[1] = np.delete(onsets[1], 5)

# ================     plot onset detection results     ================

f , i = 0, 0
odf, onset_det, hop_size = x[1], x[2], x[3]

while f < 3:
	fig = plt.figure((f), figsize=(10, 12))
	ax1 = fig.add_subplot(3, 1, 1)
	plt.title('Channel %d Onset detection with ' % (i + 1) + odf.__class__.__name__)
	plt.plot(audio[i], '0.4')
	plt.setp(ax1.get_xticklabels(), visible=False) # remmove numbers on the axis

	ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
	trplot.plot_detection_function(onset_det.odf, hop_size)
	trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
	plt.setp(ax1.get_xticklabels(), visible=False) # remmove numbers on the axis

	ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
	trplot.plot_onsets(onsets[i], 1.0)
	plt.plot(audio[i], '0.4')
	plt.show()

	i += 1
	f += 1

# Visually Compare onsets of the three channels
fig = plt.figure(f, figsize=(10, 12))
ax1 = fig.add_subplot(4, 1, 1)
fig.suptitle('Channels 1, 2, and 1&2Mean Onset detection with ' + odf.__class__.__name__)
trplot.plot_onsets(onsets[0], 1.0) # channel 1
ax1.set_title('Channel 1 Onsets') # set subplot axis title
plt.plot(audio1, '0.4')
plt.setp(ax1.get_xticklabels(), visible=False) # remmove numbers on the x axis that overlap with the title of the following subplot

ax2 = fig.add_subplot(4, 1, 2, sharex=ax1) # set axis as shared with initial subplot
trplot.plot_onsets(onsets[1], 1.0) # channel 2
ax2.set_title('Channel 2 Onsets')
plt.plot(audio2, '0.4')
plt.setp(ax2.get_xticklabels(), visible=False) # remmove numbers on the x axis that overlap with the title of the following subplot

ax3 = fig.add_subplot(4, 1, 3, sharex=ax1) # set axis as shared with initial subplot
trplot.plot_onsets(onsets[2], 1.0) # channel 3
ax3.set_title('Channel 3 Onsets')
plt.plot(audio3, '0.4')
plt.setp(ax3.get_xticklabels(), visible=False) # remmove numbers on the x axis that overlap with the title of the following subplot

ax4 = fig.add_subplot(4, 1, 4, sharex=ax1) # set axis as shared with initial subplot
trplot.plot_onsets((onsets[0], onsets[1], onsets[2]), 1.0) # channels 1, 2, and 3
ax4.set_title('Combined Onsets')
plt.plot(audio3, '0.4')
plt.show()

f += 1

# ================     plot reverse onset detection results     ================
i = 0
odf, onset_det, hop_size = xR[1], xR[2], xR[3]

while f < 7:
	fig = plt.figure((f), figsize=(10, 12))
	ax1 = fig.add_subplot(3, 1, 1)
	plt.title('Channel %d Reverse Onset detection with ' % (i+1) + odf.__class__.__name__)
	plt.plot(audioR[i], '0.4')
	plt.setp(ax1.get_xticklabels(), visible=False) # remmove numbers on the axis

	ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
	trplot.plot_detection_function(onset_det.odf, hop_size)
	trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
	plt.setp(ax1.get_xticklabels(), visible=False) # remmove numbers on the axis

	ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
	trplot.plot_onsets(onsetsR[i], 1.0)
	plt.plot(audioR[i], '0.4')
	plt.show()
	i += 1
	f += 1

# Visually Compare Reverse onsets of the three channels
fig = plt.figure(f, figsize=(10, 12))
ax1 = fig.add_subplot(4, 1, 1)
fig.suptitle('Channels 1, 2, and 1&2Mean Reverse Onset detection with ' + odf.__class__.__name__)
trplot.plot_onsets(onsetsR[0], 1.0) # channel 1
plt.plot(audioR[0], '0.4')
plt.setp(ax1.get_xticklabels(), visible=False) # remmove numbers on the x axis that overlap with the title of the following subplot

ax2 = fig.add_subplot(4, 1, 2, sharex=ax1) # set axis as shared with initial subplot
trplot.plot_onsets(onsetsR[1], 1.0) # channel 2
plt.plot(audioR[1], '0.4')
plt.setp(ax2.get_xticklabels(), visible=False) # remmove numbers on the x axis that overlap with the title of the following subplot

ax3 = fig.add_subplot(4, 1, 3, sharex=ax1) # set axis as shared with initial subplot
trplot.plot_onsets(onsetsR[2], 1.0) # channel 3
plt.plot(audioR[2], '0.4')
plt.setp(ax3.get_xticklabels(), visible=False) # remmove numbers on the x axis that overlap with the title of the following subplot

ax4 = fig.add_subplot(4, 1, 4, sharex=ax1) # set axis as shared with initial subplot
trplot.plot_onsets((onsetsR[0], onsetsR[1], onsetsR[2]), 1.0) # channels 1, 2, and 3
ax4.set_title('Combined Onsets')
plt.plot(audioR[2], '0.4')
plt.show()

f += 1

# =============     PREDICT START AND END OF ONSETS by KMeans classifer     =============

# initiate arrays for collecting and organizing all onsets and all reverse onsets
allOnsets, allOnsetsR = [], []
n = 0

while n < len(onsets):
    allOnsets.append(list(onsets[n]))
    allOnsetsR.append(list(onsetsR[n]))
    n += 1

# collect all onets and reverse onsets into one array (each,) and sort it
allOnsets  = np.concatenate(allOnsets)
allOnsetsR = np.concatenate(allOnsetsR) # re-reverse the reverse onsets to get the stop points

# allOnsetsR = ((sr*netTime) - np.concatenate(allOnsetsR)) # re-reverse the reverse onsets to get the stop points
onesOnsets, onesOnsetsR = np.ones(len(allOnsets)), np.ones(len(allOnsetsR))
allOnsets.sort(), allOnsetsR.sort()

# Format the onset and reverse onset arrays for use in KMeans
onsets_KM  = np.column_stack((onesOnsets, allOnsets))
onsetsR_KM = np.column_stack((onesOnsetsR, allOnsetsR))

# KMeans forward onsets
est = KMeans(n_clusters=3, init='random')
cluster = est.fit_predict(onsets_KM) # get the associated cluster number
centers = est.cluster_centers_

# combine the list of reverse onsets with their associated cluster number
onsetClusters = np.column_stack((allOnsets, cluster))
onsetClusters = pd.DataFrame(onsetClusters)

# return the first number in each cluser as the onset of that event
groupedOnsets = onsetClusters.groupby(onsetClusters[1])
onsets1 = np.array(groupedOnsets.first())

# KMeans reverse onsets
est = KMeans(n_clusters=3, init='random')
clusterR = est.fit_predict(onsetsR_KM) # get the associated cluster number
centersR = est.cluster_centers_

# combine the list of reverse onsets with their associated cluster number
onsetRClusters = np.column_stack((allOnsetsR, clusterR))
onsetRClusters = pd.DataFrame(onsetRClusters)

# return the first number in each cluser as the reverse onset of that event
groupedOnsetsR = onsetRClusters.groupby(onsetRClusters[1])
onsetsR1 = np.array(groupedOnsetsR.first())

onsets1  = onsets1.flatten()
onsetsR1 = onsetsR1.flatten()

# since the reverse onsets are reversed, I need to subtract their timestamp from the total timestamp to get their correct timestamp
onsetsR1 = (sr*netTime) - onsetsR1 

# =============     PLOT AUDIO WITH PREDICTED START AND STOP LOCATIONS     =============

# scatter-plot the onset times                           
fig = plt.figure(f, figsize=(10, 5))
plt.title('Predicted onset start and stops')
plt.scatter(onsets1, np.ones(len(onsets1)), c='#FF0054', marker='+', s=500)
plt.scatter(onsetsR1, np.ones(len(onsetsR1)))

f += 1

# plot onset times onto audio data
fig = plt.figure(f, figsize=(10, 5))
plt.title('audio sample with onset start and stops')
plt.plot(audio1, '0.4')
plt.vlines(onsets1, -1, 1, linestyles=u'dashed', color='#FF0054')
plt.vlines(onsetsR1, -1, 1, linestyles=u'solid', color='#FF0054')

f += 1
