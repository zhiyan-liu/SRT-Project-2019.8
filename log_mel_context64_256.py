import os
import wave
import timeit; program_start_time = timeit.default_timer()
import random; random.seed(int(timeit.default_timer()))
from six.moves import cPickle 

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa

##### 音频和音素文件用对齐程序输出的复制进来即可，不要改变文件名 #####
##### 可以处理多个文件，最后将所有结果输出到一个output.pkl文件 #####

##### THRESHOLD PARAMETER FOR VALID PHONEME JUDGEMENT #####
##### 为了数据更好，抛弃了太短的音素，下面的参数可以设置音素长度的下限值，单位为帧 #####
frame_threshold = 0

data_type = 'float32'

source_path_train = '/media/lzy/AA7EAC2F7EABF26D/database_aligned/train'
source_path_val = '/media/lzy/AA7EAC2F7EABF26D/database_aligned/val'
source_path_test = '/media/lzy/AA7EAC2F7EABF26D/database_aligned/test'

phonemes = ['sil', 'dh', 'ae', 'r', 'sp', 'w', 'ah', 'z', 't', 'ay', 'm', 'hh', 'eh', 'n', 'l', 'ow', 'uw', 'g', 'd', 'p', 'ey', 's', 'k', 'ao', 'iy', 'f', 'ih', 'v', 'uh', 'er', 'aa', 'y', 'b', 'oy', 'zh', 'ng', 'aw', 'th','sh','ch','jh']

	# With the use of CMU Dictionary there are 41 different phonemes


def find_phoneme (phoneme_idx):
	for i in range(len(phonemes)):
		if phoneme_idx == phonemes[i]:
			return i
	print("PHONEME NOT FOUND, NaN CREATED!")
	print("\t" + phoneme_idx + " wasn't found!")
	return -1


def create_logmel(filename):
	y, sr = librosa.load(filename, sr=None)
	# extract mel spectrogram feature
	melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=256, n_mels=64)
	# convert to log scale
	logmelspec = librosa.power_to_db(melspec)
	out=np.transpose(logmelspec)
	return out, out.shape[0]

def calc_norm_param(X):
	mean_val = np.mean(X,axis=0)
	std_val = np.std(X,axis=0)
	print(mean_val)
	print(std_val)
	return mean_val, std_val

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
	fr = open(phn_fname)
	X_logmel, total_frames = create_logmel(wav_fname)
	total_frames = int(total_frames)
	for line in fr:
		line_split = line.rstrip('\n').split()
		if len(line_split) == 3:
			[start_time, end_time, phoneme] = line.rstrip('\n').split()
			start_frame = int(float(start_time)*1000/11.61) + 1
			end_frame = int(float(end_time)*1000/11.61) - 4
			if (end_frame - start_frame) >= frame_threshold and (end_frame - start_frame) <= 90:
				phoneme_num = find_phoneme(phoneme)
				if phoneme_num != -1:			
					for i in range(start_frame,end_frame):
						X_frame = X_logmel[i]
						X_frame = np.append(X_frame,X_logmel[i-1])
						X_frame = np.append(X_frame,X_logmel[i-2])
						X_frame = np.append(X_frame,X_logmel[i-3])
						X_frame = np.append(X_frame,X_logmel[i-4])
						X_frame = np.append(X_frame,X_logmel[i+1])
						X_frame = np.append(X_frame,X_logmel[i+2])
						X_frame = np.append(X_frame,X_logmel[i+3])
						X_frame = np.append(X_frame,X_logmel[i+4])
						X.append(X_frame)
						Y.append(phoneme_num)
	fr.close()
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

##### PREPROCESSING #####
##### 训练集 #####
X_train = np.empty(shape = [0,576])
y_train = np.empty(shape = [0])
norm_param = []
total_frames_train = 0
for dirName, subdirList, fileList in os.walk(source_path_train):
	i = 0
	for fname in fileList:
		if not fname.endswith('.phn'): 
			continue
		phn_fname = dirName + '/' + fname
		wav_fname = dirName + '/' + fname[0:-11] + '.wav'
		i += 1
		print('Preprocessing file' + str(i))
		X, y =preprocess_dataset(phn_fname,wav_fname)									
		print('File ' + str(i) + ' preprocessing complete')
		print(X.shape)
		X = set_type(X, data_type)
		total_frames_train += X.shape[0]
		X_train = np.append(X_train,X,axis=0)
		y_train = np.append(y_train,y,axis=0)
		if X_train.shape[0] >= 500000:
			mean_val,std_val = calc_norm_param(X_train)
			norm_param.append([X_train.shape[0],mean_val,std_val])
			with open('temp', 'ab') as cPickle_file:
				cPickle.dump(
				[X_train, y_train], 
				cPickle_file, 
				protocol=cPickle.HIGHEST_PROTOCOL)
			del X_train
			del y_train
			X_train = np.empty(shape = [0,576])
			y_train = np.empty(shape = [0])		

with open('temp', 'ab') as cPickle_file:
	cPickle.dump(
	[X_train, y_train], 
	cPickle_file, 
	protocol=cPickle.HIGHEST_PROTOCOL)
mean_val,std_val = calc_norm_param(X_train)
norm_param.append([X_train.shape[0],mean_val,std_val])

print(total_frames_train)
mean_total = np.zeros(576)
std_total = np.zeros(576)

for weight,mean,std in norm_param:
	print(weight)
	print(mean)
	mean_total += mean * weight
	std_total += std * weight
mean_val = mean_total / total_frames_train
std_val = std_total / total_frames_train

with open('temp','rb') as cPickle_file:
	while True:
		try:
			[X_train, y_train] = cPickle.load(cPickle_file)
			X_train = normalize(X_train,mean_val,std_val)
			print(np.mean(X_train,axis = 0))
			print(np.std(X_train,axis = 0))
			with open('train256.pkl','ab') as target_file:
				cPickle.dump(
				[X_train, y_train],
				target_file, 
				protocol=cPickle.HIGHEST_PROTOCOL)
		except EOFError:
			break


os.system('rm temp')

##### 验证集 #####
X_val = np.empty(shape = [0,576])
y_val = np.empty(shape = [0])
total_frames_val = 0

for dirName, subdirList, fileList in os.walk(source_path_val):
	i = 0
	for fname in fileList:
		if not fname.endswith('.phn'): 
			continue
		phn_fname = dirName + '/' + fname
		wav_fname = dirName + '/' + fname[0:-11] + '.wav'
		i += 1
		print('Preprocessing file' + str(i))
		X, y =preprocess_dataset(phn_fname,wav_fname)									
		print('File ' + str(i) + ' preprocessing complete')
		print(X.shape)
		X = set_type(X, data_type)
		total_frames_val += X.shape[0]
		X_val = np.append(X_val,X,axis=0)
		y_val = np.append(y_val,y,axis=0)
		if X_val.shape[0] >= 500000:
			with open('temp', 'ab') as cPickle_file:
				cPickle.dump(
				[X_val, y_val], 
				cPickle_file, 
				protocol=cPickle.HIGHEST_PROTOCOL)
			del X_val
			del y_val
			X_val = np.empty(shape = [0,576])
			y_val = np.empty(shape = [0])		

with open('temp', 'ab') as cPickle_file:
	cPickle.dump(
	[X_val, y_val], 
	cPickle_file, 
	protocol=cPickle.HIGHEST_PROTOCOL)
del X_val
del y_val

with open('temp','rb') as cPickle_file:
	while True:
		try:
			[X_val, y_val] = cPickle.load(cPickle_file)
			X_val = normalize(X_val,mean_val,std_val)
			print(np.mean(X_val,axis = 0))
			print(np.std(X_val,axis = 0))
			with open('val256.pkl','ab') as target_file:
				cPickle.dump(
				[X_val, y_val],
				target_file, 
				protocol=cPickle.HIGHEST_PROTOCOL)
			del X_val
			del y_val
		except EOFError:
			break

os.system('rm temp')

print(total_frames_val)

##### 测试集 #####
X_test = np.empty(shape = [0,576])
y_test = np.empty(shape = [0])
total_frames_test = 0
for dirName, subdirList, fileList in os.walk(source_path_test):
	i = 0
	for fname in fileList:
		if not fname.endswith('.phn'): 
			continue
		phn_fname = dirName + '/' + fname
		wav_fname = dirName + '/' + fname[0:-11] + '.wav'
		i += 1
		print('Preprocessing file' + str(i))
		X, y =preprocess_dataset(phn_fname,wav_fname)									
		print('File ' + str(i) + ' preprocessing complete')
		print(X.shape)
		X = set_type(X, data_type)
		X_test = np.append(X_test,X,axis=0)
		y_test = np.append(y_test,y,axis=0)
		total_frames_test += X.shape[0]
		if X_test.shape[0] >= 500000:
			with open('temp', 'ab') as cPickle_file:
				cPickle.dump(
				[X_test, y_test], 
				cPickle_file, 
				protocol=cPickle.HIGHEST_PROTOCOL)
			del X_test
			del y_test
			X_test = np.empty(shape = [0,576])
			y_test = np.empty(shape = [0])	

with open('temp', 'ab') as cPickle_file:
	cPickle.dump(
	[X_test, y_test], 
	cPickle_file, 
	protocol=cPickle.HIGHEST_PROTOCOL)
del X_test
del y_test

with open('temp','rb') as cPickle_file:
	while True:
		try:
			[X_test, y_test] = cPickle.load(cPickle_file)
			X_test = normalize(X_test,mean_val,std_val)
			print(np.mean(X_test,axis = 0))
			print(np.std(X_test,axis = 0))
			with open('test256.pkl','ab') as target_file:
				cPickle.dump(
				[X_test, y_test],
				target_file, 
				protocol=cPickle.HIGHEST_PROTOCOL)
			del X_test
			del y_test
		except EOFError:
			break

os.system('rm temp')

print(total_frames_test)

print('Preprocessing complete!')
print()


print('Total time: {:.3f}'.format(timeit.default_timer() - program_start_time))

