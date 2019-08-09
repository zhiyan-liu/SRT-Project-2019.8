import os

##### DEFINING PATH #####
##### 先定义路径，目标文件夹需要是已存在的，工作文件夹不需要 #####
##### 由于一些不可理解的问题，配置文件config_fname必须使用绝对路径！！ #####
source_path = ''
lyric_fname = ''
target_path = ''
work_path = ''
config_fname = ''

##### STARTING ALIGNMENT #####
for dirName, subdirList, fileList in os.walk(source_path):
	i = 0
	for fname in fileList:
		i += 1
		print('Starting processing for file ' + i + '...')
		wav_fname = dirName + '/' + fname
		os.system('rm -r ' + work_path)
		os.system('mkdir ' + work_path)
		os.system('sail_align -i ' + wav_fname + ' -t ' + lyric_fname + ' -w ' + work_path + ' -c ' + config_fname)
		os.system('cp ' + work_path + '/' + wav_fname[0:-4] + '.forced.phn ' + target_path)
		os.system('cp ' + wav_fname + ' ' + target_path)
		print('Finished processing for file ' + i)
