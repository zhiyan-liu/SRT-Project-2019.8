import os

audio_list = list(open('index.txt'))

os.system('mkdir selected_audio')

for index in audio_list:
	file_name = index + '.m4a'
	os.system('cp audio/' + file_name + ' selected_audio')
	
