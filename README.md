# BachProp
Contains the code to generate symbolic music with BachProp

Requirements:

python 3
tensorflow
h5py
keras
tqdm
midi


To create a folder in the save directory with generated MIDI files from a trained model on Bach's Chorales:

python BachProp.py ChoralesMusic21 load

To train and save a model on DATASET (provided the midi data are in a midi folder inside a directory named DATASET in the data directory):

python BachProp.py DATASET train


