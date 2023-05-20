import os
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
path = os.listdir(r'C:\Users\renat\PycharmProjects\IA\Proiect\data')
f = open("train_labels.txt", "r")
train={}
for i in f:
    linie=i.strip()
    linie=linie.split(",")
    train[linie[0]]=int(linie[1])
f = open("validation_labels.txt", "r")
for i in f:
    linie=i.strip()
    linie=linie.split(",")
    train[linie[0]]=int(linie[1])
f = open("sample_submission.txt", "r")
sample={}
contorr=0
for i in f:
    contorr+=1
    linie=i.strip()
    linie=linie.split(",")
    sample[linie[0]]=int(linie[1])
# print(contorr)
Imagini=[]
Rezultate=[]
for i in path:
    image= cv2.imread(r'C:\Users\renat\PycharmProjects\IA\Proiect\data/'+i,0)
    if image.size!=0:
        # plt.imshow(image, cmap='gray')
        # plt.show()
        Imagini.append(image)
        if i[:-4] in train:
            Rezultate.append(train[i[:-4]])
# print(len(Imagini))
# print(len(Rezultate))
X=np.array(Imagini)
Y=np.array(Rezultate)
de_sters=[]
X=X.reshape(len(X),-1)
print(X.shape)
print(X.shape)
print(Y.shape)
xtrain=X[:15000]
xtest=X[15000:17000]
ytrain=Y[:15000]
ytest=Y[15000:17000]
xsubmit=X[17000:]
# X_test = xtest.astype('float32')
# X_test /= 255
# X_train = xtrain.astype('float32')
# X_train /= 255
# X_submit = xsubmit.astype('float32')
# X_submit /= 255
scaler = StandardScaler()
# fit the scaler on the array and transform it
X_test = scaler.fit_transform(xtest)
X_train = scaler.fit_transform(xtrain)
X_submit = scaler.fit_transform(xsubmit)
lg=LogisticRegression(solver="saga",C=0.1)
lg.fit(X_train,ytrain)
predicted=lg.predict(X_test)
cm = confusion_matrix(ytest, predicted)
print('Confusion matrix:\n', cm)
score=lg.score(X_test,ytest)
print(score)
predicted= lg.predict(X_submit)
contor=0
f = open("output.txt", "w")
for i in sample:
    if contor<5149:
        f.write("{},{}\n".format(int(i), int(predicted[contor])))
        contor += 1
