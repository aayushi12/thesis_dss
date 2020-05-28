from __future__ import print_function
import os
import nltk
nltk.download('perluniprops')

import tables
#PyTables is a package for managing hierarchical datasets and designed to efficently cope with extremely large amounts of data. PyTables is #built on top of the HDF5 library and the NumPy package and features an object-oriented interface that, combined with C-code generated from #Cython sources, makes of it a fast, yet extremely easy to use tool for interactively save and retrieve large amounts of data.

import argparse
import torch
import numpy as np

from torch.optim import lr_scheduler
#torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs. #torch.optim.lr_scheduler.ReduceLROnPlateau allows dynamic learning rate reducing based on some validation measurements.

import sys
sys.path.append(os.getcwd()+'\\functions') 

from trainer_modified import flickr_trainer
from costum_loss import batch_hinge_loss, ordered_loss, attention_loss
from encoders import img_encoder, audio_rnn_encoder
from data_split import split_data_flickr

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = 'prep_data_trunc/flickr_features.h5',
                    help = 'location of the feature file, default: prep_data_trunc/flickr_features.h5')
parser.add_argument('-split_loc', type = str, default = 'data/databases/flickr/dataset.json', 
                    help = 'location of the json file containing the data split information')
parser.add_argument('-results_loc', type = str, default = 'results/',
                    help = 'location to save the trained models')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 32, help = 'batch size, default: 32')
parser.add_argument('-lr', type = float, default = 0.0002, help = 'learning rate, default:0.0002')
parser.add_argument('-n_epochs', type = int, default = 30, help = 'number of training epochs, default: 25')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-visual', type = str, default = 'resnet', help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'mfcc', help = 'name of the node containing the audio features, default: mfcc')
parser.add_argument('-gradient_clipping', type = bool, default = False, help ='use gradient clipping, default: False')

args = parser.parse_args(args=[])

audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 'kernel_size': 6, 'stride': 2,
               'padding': 0, 'bias': False}, 'rnn':{'input_size': 64, 'hidden_size': 1024, 
               'num_layers': 3, 'batch_first': True, 'bidirectional': True, 'dropout': 0}, 
               'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = audio_config['rnn']['hidden_size'] * 2**audio_config['rnn']['bidirectional'] * audio_config['att']['heads']
image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 'norm': True}


# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')

def iterate_data(h5_file):
    for x in h5_file.root:
        yield x
f_nodes = [node for node in iterate_data(data_file)] 

# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, test, val = split_data_flickr(f_nodes, args.split_loc)

############################### Neural network setup #################################################

# network modules
img_net = img_encoder(image_config)
cap_net = audio_rnn_encoder(audio_config)

# Adam optimiser. I found SGD to work terribly and could not find appropriate parameter settings for it.
optimizer = torch.optim.Adam(list(img_net.parameters())+list(cap_net.parameters()), 1)

#plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.9, patience = 100, 
#                                                   threshold = 0.0001, min_lr = 1e-8, cooldown = 100)

#step_scheduler = lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, last_epoch=-1)

def create_cyclic_scheduler(max_lr, min_lr, stepsize):
    lr_lambda = lambda iteration: (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1))+min_lr
    cyclic_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press between
    # min and max lr   
    return(cyclic_scheduler)

cyclic_scheduler = create_cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, stepsize = (int(len(train)/args.batch_size)*5)*4)

trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_loss(batch_hinge_loss)
trainer.set_optimizer(optimizer)
trainer.set_audio_batcher()
trainer.set_lr_scheduler(cyclic_scheduler, 'cyclic')
trainer.set_att_loss(attention_loss)

# optionally use cuda, gradient clipping and pretrained glove vectors
if cuda:
    trainer.set_cuda()
trainer.set_evaluator([1, 5, 10])
# gradient clipping with these parameters (based the avg gradient norm for the first epoch)
# can help stabilise training in the first epoch.
if args.gradient_clipping:
    trainer.set_gradient_clipping(0.0025, 0.05)

while trainer.epoch <= args.n_epochs: 
    # Train on the train set
    trainer.train_epoch(train, args.batch_size)
    
    #evaluate on the validation set
    #trainer.test_epoch(val, args.batch_size)
    
    # save network parameters
    #trainer.save_params(args.results_loc)  
    loc = args.results_loc
    torch.save(trainer.cap_embedder.state_dict(), os.path.join(loc, 'caption_model' + '.' +str(trainer.epoch)))
    torch.save(trainer.img_embedder.state_dict(), os.path.join(loc, 'image_model' + '.' +str(trainer.epoch)))
    torch.save({'epoch': trainer.epoch,'optimizer_state_dict': trainer.optimizer.state_dict(),'lr_scheduler_state_dict': trainer.lr_scheduler.state_dict(), "iteration": trainer.iteration, "scheduler": trainer.scheduler}, os.path.join(loc, 'other_params' + '.' +str(trainer.epoch)))
        
    # print some info about this epoch
    #trainer.report(args.n_epochs)
    #trainer.recall_at_n(val, args.batch_size, prepend = 'validation')    

    if args.gradient_clipping:
        # I found that updating the clip value at each epoch did not work well     
        # trainer.update_clip()
        trainer.reset_grads()
    #increase epoch#
    trainer.update_epoch()