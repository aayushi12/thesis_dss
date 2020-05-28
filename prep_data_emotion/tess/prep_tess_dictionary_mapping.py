# Dataset info: https://tspace.library.utoronto.ca/handle/1807/24487
# There should be 2800 stimuli in total
import os
from scipy.io.wavfile import read
import librosa
from scipy.io import wavfile
import numpy as np


participants = ['OAF','YAF']

emotions = ['angry','disgust','fear','happy','neutral','ps','sad']

#Set of 200 words
words = []

data_loc = os.path.join(os.getcwd(),'toronto-emotional-speech-set')

folders = ['OAF_' + emotion for emotion in emotions] + ['YAF_' + emotion for emotion in emotions]

#getting all the words, so that they can be iterated over later

file_names = os.listdir(os.path.join(data_loc,folders[0]))
words = [file_name.split("_")[1] for file_name in file_names]

dic = {}
obs = 0
files_skipped = []

for participant in participants:
    for emotion in emotions:
        current_loc = os.path.join(data_loc,"_".join([participant,emotion]))
        
        for word in words:
            file_name = "_".join([participant,word,emotion]) + ".wav"
            file_loc = os.path.join(current_loc,file_name)
            
            try:
                input_data = read(file_loc)
            except:
                files_skipped.append(file_name)
                continue
            
            obs += 1
            feature_dic = {'participant': participant,'emotion': emotion, 'word':word}
            
            dic[obs] = feature_dic
            
#3 files are skipped
#['OA_bite_neutral.wav', 'YAF_germ_angry.wav', 'YAF_neat_fear.wav']
# Name corrected for first (OA -> OAF). 2 remaining files seem to be corrupted. Will try fix_wav on them
            
def fix_wav(path_to_file):
    y, sr = librosa.load(path_to_file, sr = None, mono=True)
    y = y * 32767/max(0.01, np.max(np.abs(y)))
    wavfile.write(path_to_file, sr, y.astype(np.int16))

dic = {}
obs = 0
files_skipped = []

for participant in participants:
    for emotion in emotions:
        current_loc = os.path.join(data_loc,"_".join([participant,emotion]))
        
        for word in words:
            file_name = "_".join([participant,word,emotion]) + ".wav"
            file_loc = os.path.join(current_loc,file_name)
            
            try:
                input_data = read(file_loc)
            except:
                try:
                    fix_wav(file_loc)
                    input_data = read(file_loc)
                except:    
                    files_skipped.append(file_name)
                    continue
            
            obs += 1
            feature_dic = {'participant': participant,'emotion': emotion, 'word':word}
            
            dic[obs] = feature_dic

#All 2800 files read

import pickle
f = open("tess_info.pkl","wb")
pickle.dump(dic,f)
f.close()
