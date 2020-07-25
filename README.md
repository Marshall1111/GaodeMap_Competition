# GaodeMap_Competition

# Put data files ‘amap_traffic_train', 'amap_traffic_annotations_train.json' in the folder "data"!


# convert data to h5py format
python dataset_creat_h5.py

# run the main model for training in cuda mdoe
python mian.py --cuda 1  

# run the main model for training in cpu mdoe
python mian.py --cuda 0
