#Filename labelling convention
#https://github.com/CheyneyComputerScience/CREMA-D

import numpy as np
from scipy.io.wavfile import read
import os
import wave

input_data = read('AudioWAV/1001_DFA_ANG_XX.wav')
print(input_data[0])
print(input_data[1])

actor_id = np.arange(1001,1092) #1001 to 1091, total 91 participants
sentences = ['IEO','TIE','IOM','IWW','TAI','MTI','IWL','ITH','DFA','ITS','TSI','WSI']
emotions = ['ANG','DIS','FEA','HAP','NEU','SAD']
intensities = ['XX','HI','MD','LO']

#Create a dictionary containing information of this dataset

#info = {'observation_number': , 'actor_id':, sentence:, emotion:, intensity:}

dic = {}

obs = 0
exceptions = 0
for actor in actor_id:
    for sentence in sentences:
        for emotion in emotions:
            for intensity in intensities:
                try:
                    file_path = 'AudioWAV/' + str(actor) + "_" + sentence + "_" + emotion + "_" + intensity + ".wav"
                    input_data = read(file_path)
                    
                    if len(input_data[1]) == 0:
                        print("$")
                    
                except:
                    exceptions += 1
                    continue
                
                obs += 1
                info_dic = {'actor_id': actor, 'sentence': sentence, 'emotion': emotion, "intensity": intensity}
                dic[obs] = info_dic
                
expected_obs = len(actor_id)*len(sentences)*len(emotions)*len(intensities) - exceptions
         
# I am getting 7441 observations, no audio file is empty, should be getting 7442

audio_path = os.path.join(os.getcwd(),'AudioWAV')
audio = os.listdir(audio_path)

audio_read = []

for obs in dic:
    feature_dic = dic[obs]
    actor = feature_dic['actor_id']
    sentence = feature_dic['sentence']
    emotion = feature_dic['emotion']
    intensity = feature_dic['intensity']
    file_name = str(actor) + "_" + sentence + "_" + emotion + "_" + intensity + ".wav"
    audio_read.append(file_name)
    

audio = set(audio)
audio_read = set(audio_read)


audio - audio_read #'1040_ITH_SAD_X.wav' This file has follows a wrong naming convention

#File name changed manually

dic = {}

obs = 0
exceptions = 0
for actor in actor_id:
    for sentence in sentences:
        for emotion in emotions:
            for intensity in intensities:
                try:
                    file_path = 'AudioWAV/' + str(actor) + "_" + sentence + "_" + emotion + "_" + intensity + ".wav"
                    input_data = read(file_path)
                    
                    if len(input_data[1]) == 0:
                        print("$")
                    
                except:
                    exceptions += 1
                    continue
                
                obs += 1
                info_dic = {'actor_id': actor, 'sentence': sentence, 'emotion': emotion, "intensity": intensity}
                dic[obs] = info_dic

#Saving the dictionary
import pickle
f = open("cremad_info.pkl","wb")
pickle.dump(dic,f)
f.close()
