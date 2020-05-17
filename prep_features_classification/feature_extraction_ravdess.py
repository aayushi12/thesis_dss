#Note: The labels need to be shuffled too

#%%
import sys
sys.path.append('..visually_grounded_model/functions')
import torch
import torch.nn as nn
from encoders import audio_rnn_encoder
import h5py
import numpy as np

#%%
class audio_rnn_sublayers(nn.Module):
    def __init__(self, config):
        super(audio_rnn_sublayers, self).__init__()
        conv = config['conv']
        rnn= config['rnn']
        self.Conv = nn.Conv1d(in_channels = conv['in_channels'], 
                                  out_channels = conv['out_channels'], kernel_size = conv['kernel_size'],
                                  stride = conv['stride'], padding = conv['padding'])
        self.RNN = nn.GRU(input_size = rnn['input_size'], hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], dropout = rnn['dropout'])
        self.pool = nn.functional.avg_pool1d
        
    def forward(self, input, l):
        x = self.Conv(input)
        # update the lengths to compensate for the convolution
        l = [int((y-(self.Conv.kernel_size[0]-self.Conv.stride[0]))/self.Conv.stride[0]) for y in l]
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True)
        x, hx = self.RNN(x)      
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)

        x = nn.functional.normalize(self.pool(x.permute(0,2,1), x.size(1)).squeeze(), p=2, dim=1)  
        return x


#%%
audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 'kernel_size': 6, 'stride': 2,
               'padding': 0, 'bias': False}, 'rnn':{'input_size': 64, 'hidden_size': 1024, 
               'num_layers': 3, 'batch_first': True, 'bidirectional': True, 'dropout': 0}, 
               'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}  

trained_loc = "..visually_grounded_model/results/caption_model.30"    
#%%

full_net = audio_rnn_encoder(audio_config)
        
cap_state = torch.load(trained_loc, map_location=torch.device('cpu'))

for layer in cap_state:
    print(layer)

full_net.load_state_dict(cap_state)

#%%
#Attention part removed

three_layer_net = audio_rnn_sublayers(audio_config)

cap_state = torch.load(trained_loc, map_location=torch.device('cpu'))

n = 26
        
while len(cap_state) > n:
    cap_state.popitem(-1)
        
three_layer_net.load_state_dict(cap_state)

#%%
#Last RNN layer removed

audio_config['rnn']['num_layers'] = 2

two_layer_net = audio_rnn_sublayers(audio_config)

cap_state = torch.load(trained_loc, map_location=torch.device('cpu'))

n = 18
        
while len(cap_state) > n:
    cap_state.popitem(-1)
        
two_layer_net.load_state_dict(cap_state)

#%%
#Two RNN layers removed

audio_config['rnn']['num_layers'] = 1

single_layer_net = audio_rnn_sublayers(audio_config)

cap_state = torch.load(trained_loc, map_location=torch.device('cpu'))

n = 10
        
while len(cap_state) > n:
    cap_state.popitem(-1)
        
single_layer_net.load_state_dict(cap_state)


#%%
#only convolutions

class conv_layer(nn.Module):
    def __init__(self, config):
        super(conv_layer, self).__init__()
        conv = config['conv']
        
        self.Conv = nn.Conv1d(in_channels = conv['in_channels'], 
                                  out_channels = conv['out_channels'], kernel_size = conv['kernel_size'],
                                  stride = conv['stride'], padding = conv['padding'])
                
    def forward(self, input, l):
        x = self.Conv(input)
        return x

untrained_net = conv_layer(audio_config)

cap_state = torch.load(trained_loc, map_location=torch.device('cpu'))

n = 2
        
while len(cap_state) > n:
    cap_state.popitem(-1)
        
untrained_net.load_state_dict(cap_state)

#%%

def batcher(lis, batch_size):
    obs = len(lis)
    
    if obs%batch_size == 0:
        num_batches = int(obs/batch_size)
    else:
        num_batches = int(obs/batch_size) + 1
    
    new_lis = [[] for i in range(num_batches)]
   
    batch_number = 0
    
    
    for file in lis:
        if len(new_lis[batch_number]) < batch_size:
            new_lis[batch_number].append(file)
        else:
            batch_number += 1
            new_lis[batch_number].append(file)
    
    return new_lis
#%%

mapping = {'01': "neutral", '02':"calm", '03': 'happy', '04': 'sad', '05': 'angry', '06' : 'fearful', '07':'disgust', '08': 'surprised'}


#%%
data_loc = "../prep_data_emotion/ravdess/prep_data/ravdess_features.h5"

f = h5py.File(data_loc, 'r')

lis = list(f.keys())

batch_size = 100

batches = batcher(lis,batch_size)

feature_matrix = np.zeros((len(list(f.keys())),2048))

targets = []
intensities = []
actor_ids = []

for batch in batches:
    lengths = []
    num_examples = len(batch)
    labels = []
    intensity = []
    actor = []
    
    for example in batch:
        s = "/".join([example,"mfcc",example])
        length = len([y for y in f[s]])
        lengths.append(length)
        labels.append( mapping[example.split("_")[3]])
        intensity.append(example.split("_")[4])
        actor.append(example.split("_")[-1])
                    
    max_length = max(lengths)
    
    labels = list(np.array(labels)[np.argsort(- np.array(lengths))])
    intensity = list(np.array(intensity)[np.argsort(- np.array(lengths))])
    actor = list(np.array(actor)[np.argsort(- np.array(lengths))])

    targets += labels
    intensities += intensity
    actor_ids += actor

print(len(targets))
print(len(intensities))
print(len(actor_ids))

print(np.unique(targets))
print(np.unique(intensities))
print(np.unique(actor_ids))

np.save('ravdess/' + 'correct_labels.npy',targets)
np.save('ravdess/' + 'intensity_labels.npy',intensities)
np.save('ravdess/' + 'actor_labels.npy', actor_ids)

#%%
    
#Full net
    
data_loc = "prep_data/ravdess_features.h5"
full_net.eval()

#Loading features extracted from sound clips

f = h5py.File(data_loc, 'r')

len(list(f.keys())) #1440

lis = list(f.keys())

batch_size = 100

batches = batcher(lis,batch_size)

#for name in f:
    #print(name)

feature_matrix = np.zeros((len(list(f.keys())),2048))

targets = []

start = 0

for batch in batches:
    lengths = []
    num_examples = len(batch)
    labels = []
    
    for example in batch:
        s = "/".join([example,"mfcc",example])
        length = len([y for y in f[s]])
        lengths.append(length)
        labels.append( mapping[example.split("_")[3]])
    
    max_length = max(lengths)
    
    speech_arr = np.zeros((num_examples,max_length,39))   
    
    j = 0
    for example in batch:
        s = "/".join([example,"mfcc",example])
        
        i = 0
    
        for feature_frame in f[s]:
            speech_arr[j,i,:] = feature_frame
            i += 1
        
        j += 1

    speech_arr = np.swapaxes(speech_arr,2,1) 
    # We need to sort according to length first
    speech_arr = speech_arr[np.argsort(- np.array(lengths))]
    labels = list(np.array(labels)[np.argsort(- np.array(lengths))])
    lengths = list(np.array(lengths)[np.argsort(- np.array(lengths))]) 
    
    
    speech_arr = torch.FloatTensor(speech_arr)
    
    activation = full_net(speech_arr,lengths)
    
    activation = activation.detach().numpy()
    
    feature_matrix[start:(start+num_examples),:] = activation
    
    targets += labels
    
    start = start + num_examples

    print(start)

print(len(targets))
np.save('ravdess/' + 'correct_labels.npy',targets)

print(feature_matrix[-1,:])
np.save('ravdess/' + 'activations_attention_ravdess.npy',feature_matrix)
f.close()


#%%

#Attention layer removed

data_loc = "prep_data/ravdess_features.h5"
three_layer_net.eval()

#Loading features extracted from sound clips

f = h5py.File(data_loc, 'r')

len(list(f.keys())) #1440

lis = list(f.keys())

batch_size = 100

batches = batcher(lis,batch_size)

#for name in f:
    #print(name)

feature_matrix = np.zeros((len(list(f.keys())),2048))

targets = []

start = 0

for batch in batches:
    lengths = []
    num_examples = len(batch)
    labels = []
    
    for example in batch:
        s = "/".join([example,"mfcc",example])
        length = len([y for y in f[s]])
        lengths.append(length)
        labels.append(mapping[example.split("_")[3]])
    
    max_length = max(lengths)
    
    speech_arr = np.zeros((num_examples,max_length,39))   
    
    j = 0
    for example in batch:
        s = "/".join([example,"mfcc",example])
        
        i = 0
    
        for feature_frame in f[s]:
            speech_arr[j,i,:] = feature_frame
            i += 1
        
        j += 1

    speech_arr = np.swapaxes(speech_arr,2,1) 
    # We need to sort according to length first
    speech_arr = speech_arr[np.argsort(- np.array(lengths))]
    labels = list(np.array(labels)[np.argsort(- np.array(lengths))])
    lengths = list(np.array(lengths)[np.argsort(- np.array(lengths))]) 
    
    
    speech_arr = torch.FloatTensor(speech_arr)
    
    activation = three_layer_net(speech_arr,lengths)
    
    activation = activation.detach().numpy()
    
    feature_matrix[start:(start+num_examples),:] = activation
    
    targets += labels
    
    start = start + num_examples

    print(start)


print(feature_matrix[-1,:])
np.save('ravdess/' + 'activations_rnn_3_ravdess.npy',feature_matrix)
f.close()


#%%

#Two layer RNN

data_loc = "prep_data/ravdess_features.h5"
two_layer_net.eval()

#Loading features extracted from sound clips

f = h5py.File(data_loc, 'r')

len(list(f.keys())) #1440

lis = list(f.keys())

batch_size = 100

batches = batcher(lis,batch_size)

#for name in f:
    #print(name)

feature_matrix = np.zeros((len(list(f.keys())),2048))

targets = []

start = 0

for batch in batches:
    lengths = []
    num_examples = len(batch)
    labels = []
    
    for example in batch:
        s = "/".join([example,"mfcc",example])
        length = len([y for y in f[s]])
        lengths.append(length)
        labels.append(mapping[example.split("_")[3]])
    
    max_length = max(lengths)
    
    speech_arr = np.zeros((num_examples,max_length,39))   
    
    j = 0
    for example in batch:
        s = "/".join([example,"mfcc",example])
        
        i = 0
    
        for feature_frame in f[s]:
            speech_arr[j,i,:] = feature_frame
            i += 1
        
        j += 1

    speech_arr = np.swapaxes(speech_arr,2,1) 
    # We need to sort according to length first
    speech_arr = speech_arr[np.argsort(- np.array(lengths))]
    labels = list(np.array(labels)[np.argsort(- np.array(lengths))])
    lengths = list(np.array(lengths)[np.argsort(- np.array(lengths))]) 
    
    
    speech_arr = torch.FloatTensor(speech_arr)
    
    activation = two_layer_net(speech_arr,lengths)
    
    activation = activation.detach().numpy()
    
    feature_matrix[start:(start+num_examples),:] = activation
    
    targets += labels
    
    start = start + num_examples

    print(start)

print(len(targets))

print(feature_matrix[-1,:])
np.save('ravdess/' + 'activations_rnn_2_ravdess.npy',feature_matrix)
f.close()
    
#%%

#Single layer RNN

data_loc = "prep_data/ravdess_features.h5"
single_layer_net.eval()

#Loading features extracted from sound clips

f = h5py.File(data_loc, 'r')

len(list(f.keys())) #1440

lis = list(f.keys())

batch_size = 100

batches = batcher(lis,batch_size)

#for name in f:
    #print(name)

feature_matrix = np.zeros((len(list(f.keys())),2048))

targets = []

start = 0

for batch in batches:
    lengths = []
    num_examples = len(batch)
    labels = []
    
    for example in batch:
        s = "/".join([example,"mfcc",example])
        length = len([y for y in f[s]])
        lengths.append(length)
        labels.append(mapping[example.split("_")[3]])
    
    max_length = max(lengths)
    
    speech_arr = np.zeros((num_examples,max_length,39))   
    
    j = 0
    for example in batch:
        s = "/".join([example,"mfcc",example])
        
        i = 0
    
        for feature_frame in f[s]:
            speech_arr[j,i,:] = feature_frame
            i += 1
        
        j += 1

    speech_arr = np.swapaxes(speech_arr,2,1) 
    # We need to sort according to length first
    speech_arr = speech_arr[np.argsort(- np.array(lengths))]
    labels = list(np.array(labels)[np.argsort(- np.array(lengths))])
    lengths = list(np.array(lengths)[np.argsort(- np.array(lengths))]) 
    
    
    speech_arr = torch.FloatTensor(speech_arr)
    
    activation = single_layer_net(speech_arr,lengths)
    
    activation = activation.detach().numpy()
    
    feature_matrix[start:(start+num_examples),:] = activation
    
    targets += labels
    
    start = start + num_examples

    print(start)

print(len(targets))
print(feature_matrix[-1,:])

np.save('ravdess/' + 'activations_rnn_1_ravdess.npy',feature_matrix)
f.close()


#%%
#Conv modified
from sklearn.preprocessing import normalize

data_loc = "prep_data/ravdess_features.h5"
untrained_net.eval()

#Loading features extracted from sound clips

f = h5py.File(data_loc, 'r')

len(list(f.keys())) #1440

lis = list(f.keys())

len(np.unique(lis)) #Just to make sure

batch_size = 100

batches = batcher(lis,batch_size)

feature_matrix = np.zeros((len(list(f.keys())),64))

targets = []

start = 0

for batch in batches:
    lengths = []
    num_examples = len(batch)
    labels = []
    
    for example in batch:
        s = "/".join([example,"mfcc",example])
        length = len([y for y in f[s]])
        lengths.append(length)
        labels.append(mapping[example.split("_")[3]])
    
    max_length = max(lengths)
    
    speech_arr = np.zeros((num_examples,max_length,39))   
    
    j = 0
    for example in batch:
        s = "/".join([example,"mfcc",example])
        
        i = 0
    
        for feature_frame in f[s]:
            speech_arr[j,i,:] = feature_frame
            i += 1
        
        j += 1

    speech_arr = np.swapaxes(speech_arr,2,1) 
    # We need to sort according to length first
    speech_arr = speech_arr[np.argsort(- np.array(lengths))]
    labels = list(np.array(labels)[np.argsort(- np.array(lengths))])     
    lengths = list(np.array(lengths)[np.argsort(- np.array(lengths))]) 
    
    speech_arr = torch.FloatTensor(speech_arr)
    
    activation = untrained_net(speech_arr,lengths)
    
    activation = activation.detach().numpy()
    
    feature_matrix[start:(start+num_examples),:] = normalize(np.mean(activation, axis = 2), norm = 'l2', axis = 1)
                                                            
    targets += labels
    
    start = start + num_examples

    print(start)


print(len(targets))
print(feature_matrix[-1,:])

np.save('ravdess/' + 'activations_conv_norm_ravdess.npy',feature_matrix)
f.close()

#%%

#Input features

data_loc = "prep_data/ravdess_features.h5"

f = h5py.File(data_loc, 'r')

len(list(f.keys())) #1440

feature_matrix = np.zeros((len(list(f.keys())),39))

labels = []
intensities = []
actor_ids = []

j = 0

for example in f:
    s = "/".join([example,"mfcc",example])
    length = len([y for y in f[s]])
    labels.append(mapping[example.split("_")[3]])
    intensities.append(example.split("_")[4])
    actor_ids.append(example.split("_")[-1])
    
    speech_arr_sum = np.zeros(39)   
    
    i = 0
    
    for feature_frame in f[s]:
        speech_arr_sum += feature_frame 
        i += 1

    speech_arr_avg = speech_arr_sum/length   
    
    #speech_arr_avg = normalize(speech_arr_avg, norm = 'l2')

    feature_matrix[j,:] = speech_arr_avg
    
    j += 1

feature_matrix = normalize(feature_matrix, norm = 'l2', axis = 1)

np.save('ravdess/' + "Average_MFCC_norm_ravdess.npy",feature_matrix)

print(len(labels),len(intensities),len(actor_ids))
print(np.unique(labels))
print(np.unique(intensities,return_counts =True))
print(np.unique(actor_ids,return_counts = True))

np.save('ravdess/' + 'input_layer_labels.npy',labels)
np.save('ravdess/' + 'input_layer_intensities.npy',intensities)
np.save('ravdess/' + 'input_layer_actors.npy',actor_ids)

del(feature_matrix)

f.close()

