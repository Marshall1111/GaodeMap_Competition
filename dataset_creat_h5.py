import cv2
import h5py
import time
import json



numbers = 1500

max_length = 5

image_path = './data/amap_traffic_train'

file_0 = h5py.File("./data/images.h5", "w")
dset = file_0.create_dataset("img", (numbers, max_length, 1280,720,3), dtype='uint8')


file_1 = h5py.File("./data/labels.h5", "w")
dset_1 = file_1.create_dataset("label", (numbers, 1), dtype='float32')



user_dic = './data/amap_traffic_annotations_train.json'

f = open(user_dic, 'r', encoding='utf-8')
ret_dic = json.load(f)
label_data = ret_dic['annotations']



for i in range(numbers):
	n = str(i+1)
	s = n.zfill(6)
	frame_path = image_path + '\\' + s + '\\'
	print(frame_path)

	step_size = len(label_data[i]['frames'])

	for j in range(step_size):
		frame_name = label_data[i]['frames'][j]['frame_name']
		print(frame_name)
		data = cv2.imread(frame_path + frame_name)
		data = cv2.resize(data,dsize=(720, 1280),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
		dset[i,j,:,:,:] = data

	dset_1[i,:] = label_data[i]['status']
	print(i)

file_0.close()
file_1.close()
f.close()
