import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE 
import h5py
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

test_accuracy = []
test_f1_score = []
train_f1_score = []
val_f1_score= []
train_accuracy =[]

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
#%%

labels_path = "../prep_features_classification/tess/correct_labels.npy"
features_path = "../prep_features_classification/tess/"
    
#%%
labels = np.load(labels_path)

dic = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy':3, 'neutral': 4, 'ps': 5, 'sad':6}

y = np.array([dic[str(i)] for i in labels])

#Distribution of y

n, bins, patches = plt.hist(y, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

np.unique(y,return_counts = True)

#%%
X = np.load(features_path + "activations_attention_tess.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=42, stratify = y)

plt.figure()
n, bins, patches = plt.hist(y_train, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

plt.figure()
n, bins, patches = plt.hist(y_test, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

#Want to use k fold cross validation to choose the best hyperparameter (penalty)

#Realized that grid search is not appropriate to be applied on the upsampled data, because the validation data needs to have a distribution close to test data

penalties = ["l1","l2"]
C_values = [1]
settings = []

for penalty in penalties:
    for C in C_values:
        skf = StratifiedKFold(n_splits= 5) #Setting a random_state has no effect since shuffle is False
        #skf.get_n_splits(X_train, y_train)
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
best_setting = max(settings, key = lambda x: x[2])

penalty = best_setting[0]
C = best_setting[1]

model = LogisticRegression(random_state=42,solver = "liblinear", C = C,penalty = penalty)

model.fit(X_train, y_train)

#Computing baseline
y_test_baseline = np.ones((len(y_test)))

f1_baseline = f1_score(y_test, y_test_baseline, average = "macro") #0.0358816441290668

#Evaluating performance of test set

y_test_pred = model.predict(X_test)

test_accuracy.append(accuracy_score(y_test, y_test_pred))
test_f1_score.append(f1_score(y_test,y_test_pred, average = "macro"))
train_f1_score.append(f1_score(y_train, model.predict(X_train), average = "macro"))
val_f1_score.append(best_setting[2])
train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))


print(test_accuracy)
print(test_f1_score)
print(train_f1_score)
print(val_f1_score)
print(penalty)

#%%

X = np.load(features_path + "activations_rnn_3_tess.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=42, stratify = y)

plt.figure()
n, bins, patches = plt.hist(y_train, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

plt.figure()
n, bins, patches = plt.hist(y_test, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

#Want to use k fold cross validation to choose the best hyperparameter (penalty)

#Realized that grid search is not appropriate to be applied on the upsampled data, because the validation data needs to have a distribution close to test data

penalties = ["l1","l2"]
C_values = [1]
settings = []

for penalty in penalties:
    for C in C_values:
        skf = StratifiedKFold(n_splits= 5) #Setting a random_state has no effect since shuffle is False
        #skf.get_n_splits(X_train, y_train)
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
best_setting = max(settings, key = lambda x: x[2])

penalty = best_setting[0]
C = best_setting[1]

model = LogisticRegression(random_state=42,solver = "liblinear", C = C,penalty = penalty)

model.fit(X_train, y_train)

#Computing baseline
y_test_baseline = np.ones((len(y_test)))

f1_baseline = f1_score(y_test, y_test_baseline, average = "macro") 

#Evaluating performance of test set

y_test_pred = model.predict(X_test)

test_accuracy.append(accuracy_score(y_test, y_test_pred))
test_f1_score.append(f1_score(y_test,y_test_pred, average = "macro"))
train_f1_score.append(f1_score(y_train, model.predict(X_train), average = "macro"))
val_f1_score.append(best_setting[2])
train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))

print(test_accuracy)
print(test_f1_score)
print(train_f1_score)
print(val_f1_score)
print(penalty)


#%%

X = np.load(features_path + "activations_rnn_2_tess.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=42, stratify = y)

plt.figure()
n, bins, patches = plt.hist(y_train, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

plt.figure()
n, bins, patches = plt.hist(y_test, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

#Want to use k fold cross validation to choose the best hyperparameter (penalty)

#Realized that grid search is not appropriate to be applied on the upsampled data, because the validation data needs to have a distribution close to test data

penalties = ["l1","l2"]
C_values = [1]
settings = []

for penalty in penalties:
    for C in C_values:
        skf = StratifiedKFold(n_splits= 5) #Setting a random_state has no effect since shuffle is False
        #skf.get_n_splits(X_train, y_train)
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
best_setting = max(settings, key = lambda x: x[2])

penalty = best_setting[0]
C = best_setting[1]

model = LogisticRegression(random_state=42,solver = "liblinear", C = C,penalty = penalty)

model.fit(X_train, y_train)

#Computing baseline
y_test_baseline = np.ones((len(y_test)))

f1_baseline = f1_score(y_test, y_test_baseline, average = "macro") 

#Evaluating performance of test set

y_test_pred = model.predict(X_test)

test_accuracy.append(accuracy_score(y_test, y_test_pred))
test_f1_score.append(f1_score(y_test,y_test_pred, average = "macro"))
train_f1_score.append(f1_score(y_train, model.predict(X_train), average = "macro"))
val_f1_score.append(best_setting[2])
train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))

print(test_accuracy)
print(test_f1_score)
print(train_f1_score)
print(val_f1_score)
print(penalty)

#%%

X = np.load(features_path + "activations_rnn_1_tess.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=42, stratify = y)

plt.figure()
n, bins, patches = plt.hist(y_train, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

plt.figure()
n, bins, patches = plt.hist(y_test, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

#Want to use k fold cross validation to choose the best hyperparameter (penalty)

#Realized that grid search is not appropriate to be applied on the upsampled data, because the validation data needs to have a distribution close to test data

penalties = ["l1","l2"]
C_values = [1]
settings = []

for penalty in penalties:
    for C in C_values:
        skf = StratifiedKFold(n_splits= 5) #Setting a random_state has no effect since shuffle is False
        #skf.get_n_splits(X_train, y_train)
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
best_setting = max(settings, key = lambda x: x[2])

penalty = best_setting[0]
C = best_setting[1]

model = LogisticRegression(random_state=42,solver = "liblinear", C = C,penalty = penalty)

model.fit(X_train, y_train)

#Computing baseline
y_test_baseline = np.ones((len(y_test)))

f1_baseline = f1_score(y_test, y_test_baseline, average = "macro") #0.027777777777777776

#Evaluating performance of test set

y_test_pred = model.predict(X_test)

test_accuracy.append(accuracy_score(y_test, y_test_pred))
test_f1_score.append(f1_score(y_test,y_test_pred, average = "macro"))
train_f1_score.append(f1_score(y_train, model.predict(X_train), average = "macro"))
val_f1_score.append(best_setting[2])
train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))

print(test_accuracy)
print(test_f1_score)
print(train_f1_score)
print(val_f1_score)
print(penalty)

#%%

#Conv Layer

X = np.load(features_path + "activations_conv_norm_tess.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=42, stratify = y)

plt.figure()
n, bins, patches = plt.hist(y_train, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

plt.figure()
n, bins, patches = plt.hist(y_test, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

penalties = ["l1","l2"]
C_values = [1]
settings = []

for penalty in penalties:
    for C in C_values:
        skf = StratifiedKFold(n_splits= 5) #Setting a random_state has no effect since shuffle is False
        #skf.get_n_splits(X_train, y_train)
        results_val_f1_score = []
        results_val_accuracy = []
        results_train_accuracy = []
        results_train_f1_score = []
        
        for train_index, test_index in skf.split(X_train, y_train):
            x_fold_train, x_val = X_train[train_index], X_train[test_index]
            y_fold_train, y_val = y_train[train_index], y_train[test_index]
            
            #Now, upsample the training data for the fold, and leave the val data unchanged
            
            model = LogisticRegression(random_state=42,solver = "liblinear",C = C,penalty = penalty)
            model.fit(x_fold_train, y_fold_train)
            
            y_val_pred = model.predict(x_val)
            
            results_val_accuracy.append(accuracy_score(y_val,y_val_pred))
            results_train_accuracy.append(accuracy_score(y_fold_train,model.predict(x_fold_train)))
            
            results_val_f1_score.append(f1_score(y_val,y_val_pred, average = "macro"))
            results_train_f1_score.append(f1_score(y_fold_train, model.predict(x_fold_train), average = "macro"))
            
        settings.append([penalty,C, np.mean(results_val_f1_score), np.mean(results_train_f1_score), np.mean(results_val_accuracy), np.mean(results_train_accuracy)])
            

#Selecting the best combination of settings
best_setting = max(settings, key = lambda x: x[2])

penalty = best_setting[0]
C = best_setting[1]

model = LogisticRegression(random_state=42,solver = "liblinear", C = C,penalty = penalty)

model.fit(X_train, y_train)

#Computing baseline
y_test_baseline = np.ones((len(y_test)))

f1_baseline = f1_score(y_test, y_test_baseline, average = "macro") #0.027777777777777776

#Evaluating performance of test set

y_test_pred = model.predict(X_test)

test_accuracy.append(accuracy_score(y_test, y_test_pred))
test_f1_score.append(f1_score(y_test,y_test_pred, average = "macro"))
train_f1_score.append(f1_score(y_train, model.predict(X_train), average = "macro"))
val_f1_score.append(best_setting[2])
train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))

print(test_accuracy)
print(test_f1_score)
print(train_f1_score)
print(val_f1_score)
print(penalty)


#As input layer features have observations in a different order

labels_path = "../prep_features_classification/tess/input_layer_labels.npy"
features_path = "../prep_features_classification/tess/"
    
labels = np.load(labels_path)

dic = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy':3, 'neutral': 4, 'ps': 5, 'sad':6}

y = np.array([dic[str(i)] for i in labels])

#Distribution of y

n, bins, patches = plt.hist(y, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

np.unique(y,return_counts = True)

#%%

X = np.load(features_path + "Average_MFCC_norm_tess.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=42, stratify = y)

plt.figure()
n, bins, patches = plt.hist(y_train, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

plt.figure()
n, bins, patches = plt.hist(y_test, 20, facecolor='blue', alpha=0.5)
plt.ylabel('Class distribution')
plt.show()  

#Want to use k fold cross validation to choose the best hyperparameter (penalty)

penalties = ["l1","l2"]
C_values = [1]
settings = []

for penalty in penalties:
    for C in C_values:
        skf = StratifiedKFold(n_splits= 5) #Setting a random_state has no effect since shuffle is False
        #skf.get_n_splits(X_train, y_train)
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
best_setting = max(settings, key = lambda x: x[2])

penalty = best_setting[0]
C = best_setting[1]

model = LogisticRegression(random_state=42,solver = "liblinear", C = C,penalty = penalty)

model.fit(X_train, y_train)

#Computing baseline
y_test_baseline = np.ones((len(y_test)))

f1_baseline = f1_score(y_test, y_test_baseline, average = "macro") #0.027777777777777776

#Evaluating performance of test set

y_test_pred = model.predict(X_test)

test_accuracy.append(accuracy_score(y_test, y_test_pred))
test_f1_score.append(f1_score(y_test,y_test_pred, average = "macro"))
train_f1_score.append(f1_score(y_train, model.predict(X_train), average = "macro"))
val_f1_score.append(best_setting[2])
train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))

print(test_accuracy)
print(test_f1_score)
print(train_f1_score)
print(val_f1_score)
print(penalty)


#%%

#Plotting


layers = ["Attention","GRU 3rd","GRU 2nd","GRU 1st","Conv","Input"]

layers = layers[::-1]
test_accuracy = test_accuracy[::-1]
test_f1_score = test_f1_score[::-1]
train_f1_score = train_f1_score[::-1]
val_f1_score = val_f1_score[::-1]

#Baseline
#Majority class in the training set

majority_baseline = f1_baseline
majority_baseline = [majority_baseline]*len(layers)

y_test_baseline = np.ones((len(y_test)))

accuracy_baseline = accuracy_score(y_test, y_test_baseline)

majority_baseline_accuracy = accuracy_baseline
majority_baseline_accuracy = [majority_baseline_accuracy]*len(layers)

plt.figure()
#plt.plot(layers,test_accuracy, 'bx--', linewidth=1, markersize=5, label = "accuracy score test" )
plt.plot(layers,test_accuracy, 'ro--',linewidth=0.5, markersize=2, label = "accuracy score test")
plt.plot(layers,test_f1_score, 'go-',linewidth=0.5, markersize=2, label = "f1 score test",alpha = 0.5)
plt.plot(layers,majority_baseline_accuracy,linewidth=1, label = "accuracy baseline =" +str(np.round(accuracy_baseline,2)))
plt.plot(layers,majority_baseline, color = 'm',linewidth=1, label = "f1 baseline =" + str(np.round(f1_baseline,2)))
plt.grid(axis='both', color='0.95')
plt.legend(loc="best")

plt.savefig("TESS_all_classes_corrected.jpg")

#%%
train_accuracy = train_accuracy[::-1]

for x,y in zip(train_accuracy,test_accuracy):
    print(round(x,3),round(y,3))
    
for x,y in zip(train_f1_score,test_f1_score):
    print(round(x,3),round(y,3))
