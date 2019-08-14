import os
import wave
import timeit; program_start_time = timeit.default_timer()
import random; random.seed(int(timeit.default_timer()))
from six.moves import cPickle 

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from general_tools import *
import python_speech_features as features
	# https://github.com/jameslyons/python_speech_features

##### 音频和音素文件用对齐程序输出的复制进来即可，不要改变文件名 #####
##### 可以处理多个文件，最后将所有结果输出到一个output.pkl文件 #####

##### THRESHOLD PARAMETER FOR VALID PHONEME JUDGEMENT #####
##### 为了数据更好，抛弃了太短的音素，下面的参数可以设置音素长度的下限值，单位为帧 #####
frame_threshold = 5
##### SCRIPT META VARIABLES #####

data_type = 'float32'

source_path	= 'test'

phonemes = ['sil', 'dh', 'ae', 'r', 'sp', 'w', 'ah', 'z', 't', 'ay', 'm', 'hh', 'eh', 'n', 'l', 'ow', 'uw', 'g', 'd', 'p', 'ey', 's', 'k', 'ao', 'iy', 'f', 'ih', 'v', 'uh', 'er', 'aa', 'y', 'b', 'oy', 'zh', 'ng', 'aw', 'th']

	# With the use of CMU Dictionary there are 38 different phonemes


def find_phoneme (phoneme_idx):
	for i in range(len(phonemes)):
		if phoneme_idx == phonemes[i]:
			return i
	print("PHONEME NOT FOUND, NaN CREATED!")
	print("\t" + phoneme_idx + " wasn't found!")
	return -1

def create_mfcc(method, filename):
	(rate,sample) = wav.read(filename)
	mfcc = features.mfcc(sample, rate, winlen=0.023, winstep=0.01, numcep = 13, nfilt=26,	preemph=0.97, appendEnergy=True)
	derivative = np.zeros(mfcc.shape)
	for i in range(1, mfcc.shape[0]-1):
		derivative[i, :] = mfcc[i+1, :] - mfcc[i-1, :]
	out = np.concatenate((mfcc, derivative), axis=1)
	return out, out.shape[0]

def calc_norm_param(X):
	total_len = 0
	mean_val = np.zeros(X.shape[1])
	std_val = np.zeros(X.shape[1])

	X_len = X.shape[0]
	mean_val += np.mean(X,axis=0)*X_len
	std_val += np.std(X, axis=0)*X_len
	total_len += X_len
	
	mean_val /= total_len
	std_val /= total_len

	return mean_val, std_val, total_len

def normalize(X, mean_val, std_val):
	for i in range(len(X)):
		X[i] = (X[i] - mean_val)/std_val
	return X

def set_type(X, type):
	for i in range(len(X)):
		X[i] = X[i].astype(type)
	return X


def preprocess_dataset(phn_fname, wav_fname):
	i = 0
	X = []
	Y = []
	fig = []
	num_plot = 4
	fr = open(phn_fname)
	X_val, total_frames = create_mfcc('DUMMY', wav_fname)
	total_frames = int(total_frames)
	for line in fr:
		line_split = line.rstrip('\n').split()
		if len(line_split) == 3:
			[start_time, end_time, phoneme] = line.rstrip('\n').split()
			start_frame = int(float(start_time)*100) + 1
			end_frame = int(float(end_time)*100) - 3
			if (end_frame - start_frame) >= frame_threshold and (end_frame - start_frame) <= 200:
				phoneme_num = find_phoneme(phoneme)
				if phoneme_num != -1:			
					for i in range(start_frame,end_frame):
						X.append(X_val[i])
						Y.append(phoneme_num)
	fr.close()
	X = np.array(X)
	Y = np.array(Y)
	return X, Y, fig

##### PREPROCESSING #####
X_output = np.empty(shape = [0,26])
y_output = np.empty(shape = [0])
for dirName, subdirList, fileList in os.walk(source_path):
	i = 0
	for fname in fileList:
		if not fname.endswith('.phn') or (fname.startswith("SA")): 
			continue
		phn_fname = dirName + '/' + fname
		wav_fname = dirName + '/' + fname[0:-11] + '.wav'
		i += 1
		print('Preprocessing file' + str(i))
		X, y, _ =preprocess_dataset(phn_fname,wav_fname)									
		print('File ' + str(i) + ' preprocessing complete')
		print(X.shape)
		mean_val, std_val, _ = calc_norm_param(X)
		X = normalize(X, mean_val, std_val)
		X = set_type(X, data_type)
		X_output = np.append(X_output,X,axis=0)
		y_output = np.append(y_output,y,axis=0)

print(X_output.shape)
print(y_output.shape)

with open('test.pkl', 'wb') as cPickle_file:
	cPickle.dump(
	[X_output, y_output], 
	cPickle_file, 
	protocol=cPickle.HIGHEST_PROTOCOL)

print('Preprocessing complete!')
print()


print('Total time: {:.3f}'.format(timeit.default_timer() - program_start_time))



