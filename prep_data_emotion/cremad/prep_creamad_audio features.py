from aud_feat_functions import get_fbanks, get_freqspectrum, get_mfcc, delta, raw_frames
from scipy.io.wavfile import read
import numpy
import tables
import os
import pickle
from pathlib import Path

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)


def audio_features (params, audio_path, append_name, node_list):
    
    output_file = params[5]
    # create pytable atom for the features   
    f_atom= tables.Float32Atom() 
    count = 1
    # keep track of the nodes for which no features could be made, basically empty audio files
    invalid = []
    for node in node_list:
        print('processing file:' + str(count))
        count+=1
        # create a group for the desired feature type (e.g. a group called 'fbanks')
        audio_node = output_file.create_group(node, params[4])
        # get the base name of the node this feature will be appended to
        base_name = node._v_name.split(append_name)[1]
        # get the caption file names corresponding to the image of this node
        audio_file = base_name + ".wav"
        
       
        input_data = read(os.path.join(audio_path, audio_file))
        
        if len(input_data[1]) == 0:
            print("$")
 
            
        # sampling frequency
        fs = input_data[0]
        # get window and frameshift size in samples
        window_size = int(fs*params[2])
        frame_shift = int(fs*params[3])
        
        # create features (implemented are raw audio, the frequency spectrum, fbanks and
        # mfcc's)
        if params[4] == 'raw':
            [features, energy] = raw_frames(input_data, frame_shift, window_size)
        
        elif params[4] == 'freq_spectrum':
            [frames, energy] = raw_frames(input_data, frame_shift, window_size)
            features = get_freqspectrum(frames, params[0], fs, window_size)
        
        elif params[4] == 'fbanks':
            [frames, energy] = raw_frames(input_data, frame_shift, window_size)
            freq_spectrum = get_freqspectrum(frames, params[0], fs, window_size)
            features = get_fbanks(freq_spectrum, params[1], fs) 
            
        elif params[4] == 'mfcc':
            [frames, energy] = raw_frames(input_data, frame_shift, window_size)
            freq_spectrum = get_freqspectrum(frames, params[0], fs, window_size)
            fbanks = get_fbanks(freq_spectrum, params[1], fs)
            features = get_mfcc(fbanks)
            
        # optionally add the frame energy
        if params[7]:
            features = numpy.concatenate([energy[:,None], features],1)
        # optionally add the deltas and double deltas
        if params[6]:
            single_delta= delta (features,2)
            double_delta= delta(single_delta,2)
            features= numpy.concatenate([features,single_delta,double_delta],1)
           
        # create new leaf node in the feature node for the current audio file
        feature_shape= numpy.shape(features)[1] #39
        f_table = output_file.create_earray(audio_node, append_name + base_name, f_atom, (0,feature_shape),expectedrows=5000)
        
        # append new data to the tables
        f_table.append(features)
        
        if audio_node._f_list_nodes() == []:
            # keep track of all the invalid nodes for which no features could be made
            invalid.append(node._v_name)
            # remove the top node including all other features if no captions features could be created
            output_file.remove_node(node, recursive = True)
    
    print(invalid)
    return 



data_loc = os.path.join(os.getcwd(),'prep_data/cremad_features.h5')
audio_path = os.path.join(os.getcwd(),'AudioWAV')

#Create node names for all the audio files for this dataset

node_names = []
dic = load_obj('cremad_info')

for obs in dic:
    feature_dic = dic[obs]
    lis = [str(feature_dic[x]) for x in feature_dic]
    
    name =  "_".join(lis)
    node_names.append(name)

append_name = 'cremad_'

if not Path(data_loc).is_file():
    # Create an output h5 file for the output
    output_file = tables.open_file(data_loc, mode='a')    
    # create the h5 file to hold all image and audio features. This will fail if they already excist such
    # as when you run this file to append new features to an excisting feature file
    
    for x in node_names:
        output_file.create_group("/", append_name + x)    

# else load an existing file to append new features to      
else:
    output_file = tables.open_file(data_loc, mode='a')


node_list = output_file.root._f_list_nodes()

#speech = ['raw','freq_spectrum','fbanks','mfcc']
speech = ['mfcc']
# Options avaliable are 'raw', 'freq_spectrum','fbanks','mfcc'


feat = ''
params = []
# set alpha for the preemphasis
alpha = 0.97
# set the number of desired filterbanks
nfilters = 40
# windowsize and shift in seconds
t_window = .025
t_shift = .010
# option to include delta and double delta features
use_deltas = True
# option to include frame energy
use_energy = True
# put paramaters in a list
params.append(alpha)
params.append(nfilters) 
params.append(t_window)
params.append(t_shift)
params.append(feat)
params.append(output_file)
params.append(use_deltas)
params.append(use_energy)


for ftype in speech:
    params[4] = ftype
    audio_features(params, audio_path, append_name, node_list)

output_file.close()

'''
#Understanding the obtained h5 structure

import h5py
f = h5py.File('prep_data/cremad_features.h5', 'r')

len(list(f.keys())) #7442

#name of all nodes
for name in f:
    print(name)
    
#Picking one node
for elem in f['cremad_1091_WSI_SAD_XX']:
    print(elem)
    
# Each node has four sub groups - fbanks, freq_spectrum, mfcc, raw
    
for x in f['cremad_1091_WSI_SAD_XX/mfcc']:
    print(x)       

# Now mfcc contains one group - which is basically the corresponding example
    
for y in f['cremad_1091_WSI_SAD_XX/mfcc/cremad_1091_WSI_SAD_XX']:
    print(y.shape) #(39,)   
    print(type(y)) #numpy.ndarray
    
len([y for y in f['cremad_1091_WSI_SAD_XX/mfcc/cremad_1091_WSI_SAD_XX']]) #250   
  
#Trying some other node

for elem in f['cremad_1007_IEO_NEU_XX']:
    print(elem)
    

for x in f['cremad_1007_IEO_NEU_XX/mfcc']:
    print(x)       


for y in f['cremad_1007_IEO_NEU_XX/mfcc/cremad_1007_IEO_NEU_XX']:
    print(y.shape) #(39,)   
    print(type(y)) #numpy.ndarray
    
len([y for y in f['cremad_1007_IEO_NEU_XX/mfcc/cremad_1007_IEO_NEU_XX']]) #206   

f.close()

'''    