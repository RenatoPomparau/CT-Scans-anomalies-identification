import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tensorflow.keras.metrics import Precision, Recall 
import cv2
import os
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred) # true positives
    fp = K.sum((1-y_true) * y_pred) # false positives
    fn = K.sum(y_true * (1-y_pred)) # false negatives
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1_score


train=dict()
f = open('/kaggle/input/testeee/train_labels.txt')
for filename in f:
   
    linie=filename.strip()
    linie=linie.split(",")
    train[linie[0]]=int(linie[1])
f = open('/kaggle/input/testeee/validation_labels.txt')
for filename in f:
   
    linie=filename.strip()
    linie=linie.split(",")
    train[linie[0]]=int(linie[1])
sample=dict()
f = open('/kaggle/input/testeee/sample_submission.txt')
for filename in f:
   
    linie=filename.split(",")
    sample[linie[0]]=int(linie[1])
        

Imagini=[];
Rezultate=[]
for dirname, _, filenames in os.walk('/kaggle/input/pozeeee/data'):
    filenames.sort()
    for filename in filenames:
        image= cv2.imread(f'/kaggle/input/pozeeee/data/{filename}',0)
        if image.size!=0:
            Imagini.append(image)
        if filename[:-4] in train:
            Rezultate.append(train[filename[:-4]])

# print(Imagini[0])
# print(train)


X=np.array(Imagini)
Y=np.array(Rezultate)

xtrain=X[:15000]
xtest=X[15000:17000]
ytrain=Y[:15000]
ytest=Y[15000:17000]
X_train = np.reshape(xtrain, (15000, 224, 224, 1))  #CNN
# X_train = X_train.astype('float32')
# X_train /= 255
X_test = np.reshape(xtest, (2000, 224, 224, 1))  #CNN
# X_test = X_test.astype('float32')
# X_test /= 255
y_train=ytrain
xsubmit=X[17000:]
xsubmit = np.reshape(xsubmit, (5149, 224, 224, 1))  #CNN
# xsubmit = xsubmit.astype('float32')
# xsubmit /= 255
model=Sequential()
#covolution layer
model.add(Conv2D(32,(8,8),activation='relu',input_shape=(224,224,1)))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
#covolution layer
model.add(Conv2D(32,(3,3),activation='relu'))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
#covolution layer
model.add(Conv2D(64,(3,3),activation='relu'))
#pooling layer
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
# model.add(Conv2D(16,(4,4),activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(BatchNormalization())
# model.add(BatchNormalization())
#i/p layer
model.add(Flatten())
#o/p layer
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_score])
hist= model.fit(X_train, y_train, validation_data=(X_test,ytest) , epochs=10)

f1_scores = hist.history['val_f1_score']
plt.plot(range(1, len(f1_scores)+1), f1_scores)
plt.title('F1 Score vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.show()

predicted_labels=model.predict(xsubmit)
for i in range(len(predicted_labels)):
    if predicted_labels[i]>=0.5:
        predicted_labels[i]=1
    else:
        predicted_labels[i]=0
print(predicted_labels)

f = open("output.txt", "w")
contor=17001
index=0
f.write("id,class\n")
for i in predicted_labels:
        f.write("0"+"{},{}\n".format(contor,int(predicted_labels[index])))
        index+=1
        contor+=1
f.close()