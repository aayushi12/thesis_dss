import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

test_accuracy = []
train_accuracy = []

test_baseline_accuracy = []

fig_name = "TESS_all_classes_corrected_rer.jpg"
fig_name_1 = "TESS_all_classes_corrected_diff.jpg"


def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
    
#%%

labels_path = "../prep_features_classification/tess/correct_labels.npy"
features_path = "../prep_features_classification/tess/"
    
labels = np.load(labels_path)

dic = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy':3, 'neutral': 4, 'ps': 5, 'sad':6}

y = np.array([dic[str(i)] for i in labels])

#Distribution of y

n, bins, patches = plt.hist(y, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

np.unique(y,return_counts = True)

#Ordering of input labels if different

input_labels_path = "../prep_features_classification/tess/input_layer_labels.npy"

input_labels = np.load(input_labels_path)

y_input = np.array([dic[str(i)] for i in labels])

#Distribution of y

n, bins, patches = plt.hist(y_input, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

np.unique(y_input,return_counts = True)

#%%

penalties = ["l1","l2"]
C_values = [1]

dataset = "_tess"
files_results = [x + dataset + ".npy" for x in ["activations_attention", "activations_rnn_3","activations_rnn_2","activations_rnn_1","activations_conv_norm","Average_MFCC_norm"]]

#Evaluating performance of layer-wise activations
for file in files_results:
    X = np.load(features_path + file)
    
    if "Average_MFCC_norm" in file:
        target = y_input
    else:
        target = y
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size= 1/3, random_state=42, stratify = target)
    
    settings = []
    
    for penalty in penalties:
        for C in C_values:
            skf = StratifiedKFold(n_splits= 5) #Setting a random_state has no effect since shuffle is False
            
            results_val_f1_score = []
            results_val_accuracy = []
            results_train_accuracy = []
            results_train_f1_score = []
            
            for train_index, test_index in skf.split(X_train, y_train):
                x_fold_train, x_val = X_train[train_index], X_train[test_index]
                y_fold_train, y_val = y_train[train_index], y_train[test_index]
                
                
                model = LogisticRegression(random_state=42,solver = "liblinear",C = C,penalty = penalty)
                model.fit(x_fold_train, y_fold_train)
                
                y_val_pred = model.predict(x_val)
                
                results_val_accuracy.append(accuracy_score(y_val,y_val_pred))
                results_train_accuracy.append(accuracy_score(y_fold_train,model.predict(x_fold_train)))
                
                results_val_f1_score.append(f1_score(y_val,y_val_pred, average = "macro"))
                results_train_f1_score.append(f1_score(y_fold_train, model.predict(x_fold_train), average = "macro"))
                
            settings.append([penalty,C, np.mean(results_val_f1_score), np.mean(results_train_f1_score), np.mean(results_val_accuracy), np.mean(results_train_accuracy)])
                
    #Selecting the best combination of settings
    best_setting = max(settings, key = lambda x: x[-2])
    
    penalty = best_setting[0]
    C = best_setting[1]
    
    model = LogisticRegression(random_state=42,solver = "liblinear", C = C,penalty = penalty)
    
    model.fit(X_train, y_train)
    
    #Computing baseline
    y_test_baseline = np.ones((len(y_test)))
    
    accuracy_baseline = accuracy_score(y_test, y_test_baseline) 
    
    #Evaluating performance of test set
    
    y_test_pred = model.predict(X_test)
    
    test_accuracy.append(accuracy_score(y_test, y_test_pred))
    
    train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))

    print(test_accuracy)
    print(train_accuracy)
    print(penalty)

#%%

test_baseline_accuracy = []

files_baseline = ["untrained_" + x for x in files_results]

for file in files_baseline:
    
    if "Average_MFCC_norm" in file:
        continue
    else:
        target = y
    
    X = np.load(features_path + file)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size= 1/3, random_state=42, stratify = target)
    
    settings = []
    
    for penalty in penalties:
        for C in C_values:
            skf = StratifiedKFold(n_splits= 5) #Setting a random_state has no effect since shuffle is False
            
            results_val_f1_score = []
            results_val_accuracy = []
            results_train_accuracy = []
            results_train_f1_score = []
            
            for train_index, test_index in skf.split(X_train, y_train):
                x_fold_train, x_val = X_train[train_index], X_train[test_index]
                y_fold_train, y_val = y_train[train_index], y_train[test_index]
                
                
                model = LogisticRegression(random_state=42,solver = "liblinear",C = C,penalty = penalty)
                model.fit(x_fold_train, y_fold_train)
                
                y_val_pred = model.predict(x_val)
                
                results_val_accuracy.append(accuracy_score(y_val,y_val_pred))
                results_train_accuracy.append(accuracy_score(y_fold_train,model.predict(x_fold_train)))
                
                results_val_f1_score.append(f1_score(y_val,y_val_pred, average = "macro"))
                results_train_f1_score.append(f1_score(y_fold_train, model.predict(x_fold_train), average = "macro"))
                
            settings.append([penalty,C, np.mean(results_val_f1_score), np.mean(results_train_f1_score), np.mean(results_val_accuracy), np.mean(results_train_accuracy)])
                
    #Selecting the best combination of settings
    best_setting = max(settings, key = lambda x: x[-2])
    
    penalty = best_setting[0]
    C = best_setting[1]
    
    model = LogisticRegression(random_state=42,solver = "liblinear", C = C,penalty = penalty)
    
    model.fit(X_train, y_train)
    
 
    #Evaluating performance of test set
    
    y_test_pred = model.predict(X_test)
    
    test_baseline_accuracy.append(accuracy_score(y_test, y_test_pred))
    
    print(test_baseline_accuracy)
   
#%%    
#Plotting

layers = ["Attention","GRU 3rd","GRU 2nd","GRU 1st","Conv","Input"]

layers = layers[::-1]

#Model computation

test_accuracy = test_accuracy[::-1]

train_accuracy = train_accuracy[::-1]

test_error = [1 - x for x in test_accuracy]
train_error = [1 - y for y in train_accuracy]


majority_error = [1- accuracy_baseline]*len(layers)

rer_model = [(x-y)/x for x,y in zip(majority_error,test_error)]


#Random intialisation computation
test_baseline_accuracy = test_baseline_accuracy[::-1]

layers_trunc = layers[1:]

test_baseline_error = [1 - x for x in test_baseline_accuracy]

rer_random = [(x-y)/x for x,y in zip(majority_error[1:],test_baseline_error)]



plt.figure()
plt.plot(layers,rer_model, 'ro-',linewidth=1, markersize=2, label = "RER trained")
plt.plot(layers_trunc, rer_random, 'bo-',linewidth=1, markersize=2, label = "RER random" )
plt.grid(axis='both', color='0.95')
plt.legend(loc="best")

plt.savefig(fig_name)

#%%
#Difference 
diff = [x -y for x,y in zip(rer_model[1:],rer_random)]

plt.figure()
plt.plot(layers_trunc, diff, 'mo-',linewidth=1, markersize=2, label = "RER (trained - random)" )
plt.grid(axis='both', color='0.95')
plt.legend(loc="best")

plt.savefig(fig_name_1)

#%%

for x, y in zip(train_error,test_error):
    print(round(x,3),round(y,3))
    
for x in rer_model:
    print(round(x,3))
    
for x,y in zip(rer_random,diff):
    print(round(x,3),round(y,3))

    
#%%