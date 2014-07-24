import numpy as np
import matplotlib.pyplot as plt
import math 
import random
import sys
import scipy 
from sklearn.neighbors.nearest_centroid import NearestCentroid

np.set_printoptions(threshold='nan')

def trainSet(arr):
	centr,lb = sklearn.cluster.vq.kmeans2(arr, 3, iter=10, thresh=1e-05, minit='random', missing='warn')
	return centr,lb


def mean(signal):
	s = 0.0
	for i in signal:
		s = s + i
	return s/10.0


def MAF(arr):
	#for j in range(50000)
	for i in range(10,len(arr)):
		tmp = []
		for j in range(10):
			tmp.append(i-j)
		arr[i] = mean(tmp)
	return arr

def rms(signal):
	sum = 0.0 
	for i in signal:
		sum = sum + i*i
	return np.sqrt(sum/len(signal))
def rmsFilter(arr):
	for i in range(15,len(arr)):
		tmp =[]
		for j in range(15):
			tmp.append(arr[i-j])
		arr[i] = rms(tmp)
	return arr
	#print rms(tmp)

def butterworth(arr):
	pass
def analyze_data(arr):
	#plt.plot(arr)
	peak = np.amax(arr)
	area = np.trapz(arr)
	mean = np.mean(arr)
	mean_peak = mean/peak
	"""
	print "-----------------------"
	print "Mean = " 
	print np.mean(arr)
	print "Peak is "
	print peak
	print "Area under the emg = "
	print area
	#print "area to peak ratio is ="
	#print area/peak
	print "mean to peak ratio"
	print mean_peak
	print "------------------------"""
	return [peak,mean, area,mean_peak]

def analyze_x(arr):
	data_x_1 = []
	for i in range(3500,6500):
		data_x_1.append(arr[i])
	data_x_2 = []
	for i in range(7500,10500):
		data_x_2.append(arr[i])
	data_x_3 = []
	for i in range(11500,14500):
		data_x_3.append(arr[i])
	data_x_4 = []
	for i in range(15500,18500):
		data_x_4.append(arr[i])
	data_x_5 = []
	for i in range(19500,22500):
		data_x_5.append(arr[i])
	data_x_6 = []
	for i in range(27000,30000):
		data_x_6.append(arr[i])
	data_x_7 = []
	for i in range(31000,34000):
		data_x_7.append(arr[i])
	data_x_8 = []
	for i in range(35000,38000):
		data_x_8.append(arr[i])
	data_x_9 = []
	for i in range(39000,42000):
		data_x_9.append(arr[i])
	data_x_10 = []
	for i in range(43000,46000):
		data_x_10.append(arr[i])
	data_x_11 = []
	for i in range(50000,53000):
		data_x_11.append(arr[i])
	data_x_12 = []
	for i in range(55500,58500):
		data_x_12.append(arr[i])
	data_x_13 = []
	for i in range(59500,62500):
		data_x_13.append(arr[i])
	data_x_14 = []
	for i in range(63000,66000):
		data_x_14.append(arr[i])
	data_x_15 = []
	for i in range(67000,70000):
		data_x_15.append(arr[i])
	
	
	filtered_data_x_1 = rmsFilter(data_x_1)
	filtered_data_x_2 = rmsFilter(data_x_2)
	filtered_data_x_3 = rmsFilter(data_x_3)
	filtered_data_x_4 = rmsFilter(data_x_4)
	filtered_data_x_5 = rmsFilter(data_x_5)
	filtered_data_x_6 = rmsFilter(data_x_6)
	filtered_data_x_7 = rmsFilter(data_x_7)
	filtered_data_x_8 = rmsFilter(data_x_8)
	filtered_data_x_9 = rmsFilter(data_x_9)
	filtered_data_x_10 = rmsFilter(data_x_10)
	filtered_data_x_11 = rmsFilter(data_x_11)
	filtered_data_x_12 = rmsFilter(data_x_12)
	filtered_data_x_13 = rmsFilter(data_x_13)
	filtered_data_x_14 = rmsFilter(data_x_14)
	filtered_data_x_15 = rmsFilter(data_x_15)
	
	#print filtered_x
	"""plt.plot(filtered_data_x_1)
	print np.mean(filtered_data_x_1)
	print np.amax(filtered_data_x_1)
	print np.trapz(filtered_data_x_1)
	plt.plot(filtered_data_x_2)
	print np.mean(filtered_data_x_2)
	print np.amax(filtered_data_x_2)
	print np.trapz(filtered_data_x_2)
	plt.plot(filtered_data_x_2)
	print np.mean(filtered_data_x_2)
	print np.amax(filtered_data_x_2)
	print np.trapz(filtered_data_x_2)
	"""
	res_x_1 =analyze_data(filtered_data_x_1)
	res_x_2 =analyze_data(filtered_data_x_2)
	res_x_3 =analyze_data(filtered_data_x_3)
	res_x_4 =analyze_data(filtered_data_x_4)
	res_x_5 =analyze_data(filtered_data_x_5)
	res_x_6 =analyze_data(filtered_data_x_6)
	res_x_7 =analyze_data(filtered_data_x_7)
	res_x_8 =analyze_data(filtered_data_x_8)
	res_x_9 =analyze_data(filtered_data_x_9)
	res_x_10 =analyze_data(filtered_data_x_10)
	res_x_11 =analyze_data(filtered_data_x_11)
	res_x_12 =analyze_data(filtered_data_x_12)
	res_x_13 =analyze_data(filtered_data_x_13)
	res_x_14 =analyze_data(filtered_data_x_14)
	res_x_15 =analyze_data(filtered_data_x_15)
	return [res_x_1,res_x_2,res_x_3,res_x_4,res_x_5,res_x_6,res_x_7,res_x_8,res_x_9,res_x_10,res_x_11,res_x_12,res_x_13,res_x_14,res_x_15]
#This function is made to analyze Y 
def analyze_y(signal):
	data_y_1 = []
	for i in range(3500,6000):
		data_y_1.append(signal[i])
	data_y_2 = []
	for i in range(7500,10000):
		data_y_2.append(signal[i])
	data_y_3 = []
	for i in range(11500,14000):
		data_y_3.append(signal[i])
	data_y_4 = []
	for i in range(15500,18000):
		data_y_4.append(signal[i])
	data_y_5 = []
	for i in range(19500,22000):
		data_y_5.append(signal[i])
	data_y_6 = []
	for i in range(27500,30000):
		data_y_6.append(signal[i])
	data_y_7 = []
	for i in range(31500,34000):
		data_y_7.append(signal[i])
	data_y_8 = []
	for i in range(35500,38000):
		data_y_8.append(signal[i])
	data_y_9 = []
	for i in range(39500,42000):
		data_y_9.append(signal[i])
	data_y_10 = []
	for i in range(43500,46000):
		data_y_10.append(signal[i])
	data_y_11 = []
	for i in range(51500,54000):
		data_y_11.append(signal[i])
	data_y_12 = []
	for i in range(55500,58000):
		data_y_12.append(signal[i])
	data_y_13 = []
	for i in range(59500,62000):
		data_y_13.append(signal[i])
	data_y_14 = []
	for i in range(63500,66000):
		data_y_14.append(signal[i])
	data_y_15 = []
	for i in range(67500,70000):
		data_y_15.append(signal[i])
	
	filtered_data_y_1 = rmsFilter(data_y_1)
	filtered_data_y_2 = rmsFilter(data_y_2)
	filtered_data_y_3 = rmsFilter(data_y_3)
	filtered_data_y_4 = rmsFilter(data_y_4)
	filtered_data_y_5 = rmsFilter(data_y_5)
	filtered_data_y_6 = rmsFilter(data_y_6)
	filtered_data_y_7 = rmsFilter(data_y_7)
	filtered_data_y_8 = rmsFilter(data_y_8)
	filtered_data_y_9 = rmsFilter(data_y_9)
	filtered_data_y_10 = rmsFilter(data_y_10)
	filtered_data_y_11 = rmsFilter(data_y_11)
	filtered_data_y_12 = rmsFilter(data_y_12)
	filtered_data_y_13 = rmsFilter(data_y_13)
	filtered_data_y_14 = rmsFilter(data_y_14)
	filtered_data_y_15 = rmsFilter(data_y_15)

	res_y_1=analyze_data(filtered_data_y_1)
	res_y_2=analyze_data(filtered_data_y_2)
	res_y_3=analyze_data(filtered_data_y_3)
	res_y_4=analyze_data(filtered_data_y_4)
	res_y_5=analyze_data(filtered_data_y_5)
	res_y_6=analyze_data(filtered_data_y_6)
	res_y_7=analyze_data(filtered_data_y_7)
	res_y_8=analyze_data(filtered_data_y_8)
	res_y_9=analyze_data(filtered_data_y_9)
	res_y_10=analyze_data(filtered_data_y_10)
	res_y_11=analyze_data(filtered_data_y_11)
	res_y_12=analyze_data(filtered_data_y_12)
	res_y_13=analyze_data(filtered_data_y_13)
	res_y_14=analyze_data(filtered_data_y_14)
	res_y_15=analyze_data(filtered_data_y_15)
	
	return [res_y_1,res_y_2,res_y_3,res_y_4,res_y_5,res_y_6,res_y_7,res_y_8,res_y_9,res_y_10,res_y_11,res_y_12,res_y_13,res_y_14,res_y_15]
def analyze_z(signal):
	data_z_1 = []
	for i in range(3500,6500):
		data_z_1.append(signal[i])
	data_z_2 = []
	for i in range(7500,10500):
		data_z_2.append(signal[i])
	data_z_3 = []
	for i in range(11500,14500):
		data_z_3.append(signal[i])
	data_z_4 = []
	for i in range(15500,18500):
		data_z_4.append(signal[i])
	data_z_5 = []
	for i in range(19500,22500):
		data_z_5.append(signal[i])
	data_z_6 = []
	for i in range(27500,30500):
		data_z_6.append(signal[i])
	data_z_7 = []
	for i in range(31500,34500):
		data_z_7.append(signal[i])
	data_z_8 = []
	for i in range(35500,38500):
		data_z_8.append(signal[i])
	data_z_9 = []
	for i in range(39500,42500):
		data_z_9.append(signal[i])
	data_z_10 = []
	for i in range(43500,46500):
		data_z_10.append(signal[i])
	data_z_11 = []
	for i in range(51000,54000):
		data_z_11.append(signal[i])
	data_z_12 = []
	for i in range(55500,58500):
		data_z_12.append(signal[i])
	data_z_13 = []
	for i in range(59500,62500):
		data_z_13.append(signal[i])
	data_z_14 = []
	for i in range(63500,66500):
		data_z_14.append(signal[i])
	data_z_15 = []
	for i in range(67500,70500):
		data_z_15.append(signal[i])
	
	filtered_data_z_1 = rmsFilter(data_z_1)
	filtered_data_z_2 = rmsFilter(data_z_2)
	filtered_data_z_3 = rmsFilter(data_z_3)
	filtered_data_z_4 = rmsFilter(data_z_4)
	filtered_data_z_5 = rmsFilter(data_z_5)
	filtered_data_z_6 = rmsFilter(data_z_6)
	filtered_data_z_7 = rmsFilter(data_z_7)
	filtered_data_z_8 = rmsFilter(data_z_8)
	filtered_data_z_9 = rmsFilter(data_z_9)
	filtered_data_z_10 = rmsFilter(data_z_10)
	filtered_data_z_11 = rmsFilter(data_z_11)
	filtered_data_z_12 = rmsFilter(data_z_12)
	filtered_data_z_13 = rmsFilter(data_z_13)
	filtered_data_z_14 = rmsFilter(data_z_14)
	filtered_data_z_15 = rmsFilter(data_z_15)
	
	res_z_1 = analyze_data(filtered_data_z_1)
	res_z_2 = analyze_data(filtered_data_z_2)
	res_z_3 = analyze_data(filtered_data_z_3)
	res_z_4 = analyze_data(filtered_data_z_4)
	res_z_5 = analyze_data(filtered_data_z_5)
	res_z_6 = analyze_data(filtered_data_z_6)
	res_z_7 = analyze_data(filtered_data_z_7)
	res_z_8 = analyze_data(filtered_data_z_8)
	res_z_9 = analyze_data(filtered_data_z_9)
	res_z_10 = analyze_data(filtered_data_z_10)
	res_z_11 = analyze_data(filtered_data_z_11)
	res_z_12 = analyze_data(filtered_data_z_12)
	res_z_13 = analyze_data(filtered_data_z_13)
	res_z_14 = analyze_data(filtered_data_z_14)
	res_z_15 = analyze_data(filtered_data_z_15)
	
	return [res_z_1,res_z_2,res_z_3,res_z_4,res_z_5,res_z_6,res_z_7,res_z_8,res_z_9,res_z_10,res_z_11,res_z_12,res_z_13,res_z_14,res_z_15]

def analyze_k(signal):
	data_k_1 = []
	for i in range(3500,6500):
		data_k_1.append(signal[i])
	data_k_2 = []
	for i in range(7500,10500):
		data_k_2.append(signal[i])
	data_k_3 = []
	for i in range(11500,14500):
		data_k_3.append(signal[i])
	data_k_4 = []
	for i in range(15500,18500):
		data_k_4.append(signal[i])
	data_k_5 = []
	for i in range(19500,22500):
		data_k_5.append(signal[i])
	data_k_6 = []
	for i in range(27500,30500):
		data_k_6.append(signal[i])
	data_k_7 = []
	for i in range(31500,34500):
		data_k_7.append(signal[i])
	data_k_8 = []
	for i in range(35000,38000):
		data_k_8.append(signal[i])
	data_k_9 = []
	for i in range(39000,42000):
		data_k_9.append(signal[i])
	data_k_10 = []
	for i in range(43000,46000):
		data_k_10.append(signal[i])
	data_k_11 = []
	for i in range(51000,54000):
		data_k_11.append(signal[i])
	data_k_12 = []
	for i in range(55500,58500):
		data_k_12.append(signal[i])
	data_k_13 = []
	for i in range(59500,62500):
		data_k_13.append(signal[i])
	data_k_14 = []
	for i in range(63500,66500):
		data_k_14.append(signal[i])
	data_k_15 = []
	for i in range(67500,70500):
		data_k_15.append(signal[i])
	filtered_data_k_1 = rmsFilter(data_k_1)
	filtered_data_k_2 = rmsFilter(data_k_2)
	filtered_data_k_3 = rmsFilter(data_k_3)
	filtered_data_k_4 = rmsFilter(data_k_4)
	filtered_data_k_5 = rmsFilter(data_k_5)
	filtered_data_k_6 = rmsFilter(data_k_6)
	filtered_data_k_7 = rmsFilter(data_k_7)
	filtered_data_k_8 = rmsFilter(data_k_8)
	filtered_data_k_9 = rmsFilter(data_k_9)
	filtered_data_k_10 = rmsFilter(data_k_10)
	filtered_data_k_11 = rmsFilter(data_k_11)
	filtered_data_k_12 = rmsFilter(data_k_12)
	filtered_data_k_13 = rmsFilter(data_k_13)
	filtered_data_k_14 = rmsFilter(data_k_14)
	filtered_data_k_15 = rmsFilter(data_k_15)
	
	res_k_1 = analyze_data(filtered_data_k_1)
	res_k_2 = analyze_data(filtered_data_k_2)
	res_k_3 = analyze_data(filtered_data_k_3)
	res_k_4 = analyze_data(filtered_data_k_4)
	res_k_5 = analyze_data(filtered_data_k_5)
	res_k_6 = analyze_data(filtered_data_k_6)
	res_k_7 = analyze_data(filtered_data_k_7)
	res_k_8 = analyze_data(filtered_data_k_8)
	res_k_9 = analyze_data(filtered_data_k_9)
	res_k_10 = analyze_data(filtered_data_k_10)
	res_k_11 = analyze_data(filtered_data_k_11)
	res_k_12 = analyze_data(filtered_data_k_12)
	res_k_13 = analyze_data(filtered_data_k_13)
	res_k_14 = analyze_data(filtered_data_k_14)
	res_k_15 = analyze_data(filtered_data_k_15)
	
	return [res_k_1,res_k_2,res_k_3,res_k_4,res_k_5,res_k_6,res_k_7,res_k_8,res_k_9,res_k_10,res_k_11,res_k_12,res_k_13,res_k_14,res_k_15]
	
if __name__=='__main__':
	with open(sys.argv[1],'r') as data_file:
		data=[]
		x=[]
		y=[]
		z=[]
		k=[]
		for line in data_file:
			tmp = line.strip().split() 
			data.append(tmp)
			x.append(abs(float(tmp[1])))
			y.append(abs(float(tmp[2])))
			z.append(abs(float(tmp[3])))
			k.append(abs(float(tmp[4])))

	#print len(data),len(x),len(y),len(z),len(k)
	#filtered_x = rmsFilter(x)		#MAF(x)
	#print "Analyzing x"
	res_x = analyze_x(x)
	#plt.plot(z)
	#print "Analyzing y"
	res_y =analyze_y(y)
	#print "Analyzing z"
	res_z = analyze_z(z)
	#print res_z
	#print "Analyzing k"
	res_k = analyze_k(k)
	#print res_k
	means_data_1 =[]
	means_data_2 = []
	means_data_3=[]
	means_data_4=[]
	ratio_data_1 = []
	ratio_data_2=[]
	ratio_data_3=[]
	ratio_data_4=[]
	for i in res_x:
		means_data_1.append(i[1])
		ratio_data_1.append(i[3])
	
	for i in res_y:
		means_data_2.append(i[1])
		ratio_data_2.append(i[3])
	for i in res_z:
		means_data_3.append(i[1])
		ratio_data_3.append(i[3])
	for i in res_k:
		means_data_4.append(i[1])
		ratio_data_4.append(i[3]) 
	
	# averaging out the means for all channels
	mean_avg = []
	for i in range(0,15):
		mean_avg.append((means_data_1[i] + means_data_2[i] + means_data_3[i] + means_data_4[i])/4)	
	#print len(mean_avg)	
	
	ratio_avg = []
	for i in range(0,15):
		ratio_avg.append((ratio_data_1[i]+ratio_data_2[i]+ratio_data_3[i]+ratio_data_4[i])/4)
	#print (ratio_avg)

	#mean_center,mean_lab = trainSet(mean_avg)
	#ratio_center, ratio_lab = trainSet(ratio_avg)

	clf = NearestCentroid()
	X = []
	Y = []
	for i in range(0,4):
		X.append([int(mean_avg[i]),ratio_avg[i]] )
		Y.append(0)
	for i in range(5,9):
		X.append([int(mean_avg[i]),ratio_avg[i]])
		Y.append(1)
	for i in range(10,14):
		X.append([int(mean_avg[i]),ratio_avg[i]])
		Y.append(2)
	#print X
	#print Y
	clf.fit(X,Y)
	res = clf.predict([[mean_avg[14],ratio_avg[14]]])
	if res == 0:
		print "rock"
	if res == 1:
		print "scissor"
	if res == 2:
		print "paper"



	"""
	# Plot individual channels data
	plt.figure(1)
	plt.subplot(431)
	plt.plot(x)
	plt.ylabel('x')
	#plt.figure(2)
	plt.subplot(412)
	plt.plot(y)
	plt.ylabel('y')
	#plt.figure(3)
	plt.subplot(421)
	plt.plot(z)
	plt.ylabel('z')
	#plt.figure(4)
	plt.subplot(422)
	plt.plot(k)
	plt.ylabel('k')
	
	
	# Plot the individual means of the data
	plt.figure(5)
	plt.plot(means_data_1,'.')
	plt.plot(means_data_2,'.')
	plt.plot(means_data_3,'.')
	plt.plot(means_data_4,'.')
	plt.ylabel('mean points')
	
	
	#plot the ratios of mean and peak for different channels
	plt.figure(6)
	plt.plot(ratio_data_1,'.')
	plt.plot(ratio_data_2,'.')
	plt.plot(ratio_data_3,'.')
	plt.plot(ratio_data_4,'.')
	plt.ylabel('ratio points')
	"""
	#plt.plot(mean_avg,'.')
	#plt.plot(mean_lab,'.')
	#plt.plot(ratio_avg,'.')
	#plt.plot(X)
	#plt.plot(Y)
	plt.show()