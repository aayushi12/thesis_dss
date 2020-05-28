#Dictionary mapping for Ravdess dataset

#File naming convention at: https://zenodo.org/record/1188976#.XqRHt2gzY2w

'''
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
'''


#All filenames start with 03-01

import os
from scipy.io.wavfile import read
import librosa
from scipy.io import wavfile
import numpy as np



def fix_wav(path_to_file):
    y, sr = librosa.load(path_to_file, sr = None, mono=True)
    y = y * 32767/max(0.01, np.max(np.abs(y)))
    wavfile.write(path_to_file, sr, y.astype(np.int16))

data_loc = os.path.join(os.getcwd(),'ravdess-emotional-speech-audio')

actor_list = ["Actor_0" + str(i) if i <= 9 else "Actor_" + str(i)  for i in range(1,25) ]

dic = {}

emotions = ["0" + str(i) for i in range(1,9)]
intensities = ["0" + str(i) for i in range(1,3)]
statements = ["0" + str(i) for i in range(1,3)]
repititions = ["0" + str(i) for i in range(1,3)]
#files_skipped = []

obs = 0
for actor_path in actor_list:
    data_loc_current = os.path.join(data_loc,actor_path)
    
    actor = actor_path.split("_")[1]
    
    
    for emotion in emotions:
        for intensity in intensities:
            for statement in statements:
                for repitition in repititions:
                    file_name = "03-01-" + "-".join([emotion,intensity,statement,repitition,actor]) + ".wav"
                    path = os.path.join(data_loc_current,file_name)
                    
                    try:
                        fix_wav(path) #had to apply this fix to all the audio files, so that they can be accepted by the read function
                        input_data = read(path)
                    except:
                        #files_skipped.append(file_name)
                        continue
                    
                    obs += 1
                    feature_dic = {}
                    feature_dic['emotion'] = emotion
                    feature_dic['intensity'] = intensity
                    feature_dic['statement'] = statement
                    feature_dic['repitition'] = repitition
                    feature_dic['actor'] = actor
                    
                    dic[obs] = feature_dic
    
#1440 files correctly read   
                    
import pickle
f = open("ravdess_info.pkl","wb")
pickle.dump(dic,f)
f.close()
