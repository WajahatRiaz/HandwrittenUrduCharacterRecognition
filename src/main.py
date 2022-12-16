from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import matplotlib.image as im
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os 

PIXELS = 40
DIMENSIONS = PIXELS*PIXELS

dataset_alif=[]
images_of_alif = 0

for filename in os.listdir("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Alif\\"):
    
    img=cv2.imread("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Alif\\" + filename , cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Alif\\" + filename , img)
 
    img_50x50 = cv2.resize(img,(PIXELS,PIXELS))
    img_instance = img_50x50.flatten()  

    if DIMENSIONS != img_instance.size:
        print("image pixel error") 
    
    dataset_alif.append(img_instance) 
    images_of_alif = images_of_alif + 1

print("\nVector Table for Alif\n")
data1 = np.empty([images_of_alif, DIMENSIONS], dtype = list)
for i in range(images_of_alif):
    data1[i] = dataset_alif[i]
print(data1.shape)
print(data1)


dataset_bay=[]
images_of_bay = 0

for filename in os.listdir("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Bay\\"):
    
    img=cv2.imread("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Bay\\" + filename , cv2.IMREAD_GRAYSCALE) 
    cv2.imwrite("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Bay\\" + filename , img)

    img_50x50 = cv2.resize(img,(PIXELS,PIXELS))
    img_instance = img_50x50.flatten() 

    if DIMENSIONS != img_instance.size:
        print("image pixel error")
    
    dataset_bay.append(img_instance) 
    images_of_bay = images_of_bay + 1

print("\nVector Table for Bay\n")
data2 = np.empty([images_of_bay, DIMENSIONS], dtype = list)
for i in range(images_of_bay):
    data2[i] = dataset_bay[i]
print(data2.shape)
print(data2)

dataset_jeem=[]
images_of_jeem = 0 

for filename in os.listdir("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Jeem\\"):
    
    img=cv2.imread("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Jeem\\" + filename , cv2.IMREAD_GRAYSCALE) 
    cv2.imwrite("D:\\HandwrittenUrduCharacterRecognition\\dataset\\Jeem\\" + filename , img)

    img_50x50 = cv2.resize(img,(PIXELS,PIXELS))
    img_instance = img_50x50.flatten() 

    if DIMENSIONS != img_instance.size:
        print("image pixel error")
    
    dataset_jeem.append(img_instance) 
    images_of_jeem = images_of_jeem + 1

print("\nVector Table for Jeem\n")
data3 = np.empty([images_of_jeem, DIMENSIONS], dtype = list)
for i in range(images_of_jeem):
    data3[i] = dataset_jeem[i]
print(data3.shape)
print(data3)


dataset_daal=[]
images_of_daal = 0

for filename in os.listdir("D:\\HandwrittenUrduCharacterRecognition\\Dataset\\Daal\\"):
    
    img=cv2.imread("D:\\HandwrittenUrduCharacterRecognition\\Dataset\\daal\\" + filename , cv2.IMREAD_GRAYSCALE) 
    cv2.imwrite("D:\\HandwrittenUrduCharacterRecognition\\Dataset\\daal\\" + filename , img)

    img_50x50 = cv2.resize(img,(PIXELS,PIXELS))
    img_instance = img_50x50.flatten() 
    if DIMENSIONS != img_instance.size:
        print("image pixel error")
    
    dataset_daal.append(img_instance) 
    images_of_daal = images_of_daal + 1

print("\nVector Table for Daal\n")
data4 = np.empty([images_of_daal, DIMENSIONS], dtype = list)
for i in range(images_of_daal):
    data4[i] = dataset_daal[i]
print(data4.shape)
print(data4)

instances = images_of_alif + images_of_bay + images_of_jeem + images_of_daal

print("instances", instances)
print("dimension" , DIMENSIONS)

x = np.concatenate((data1,data2,data3,data4))

print("my x matrix", x)
print("dimensions of x ", x.shape)

tag_alif = np.full((images_of_alif,1), 1, dtype=int)
print(tag_alif.shape)
tag_bay = np.full((images_of_bay,1), 2, dtype=int)

tag_jeem = np.full((images_of_jeem,1), 3, dtype=int)

tag_daal = np.full((images_of_daal,1), 4, dtype=int)


out_vector = np.concatenate((tag_alif,tag_bay,tag_jeem,tag_daal))
print(out_vector.shape)

y = np.ravel(out_vector, order='A')

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

model_1 = RandomForestClassifier(n_estimators=500)
model_1.fit(X_train, y_train)
predictions = model_1.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))

model_2 = SVC()
model_2.fit(X_train, y_train)
predictions = model_2.predict(X_test)
print(accuracy_score(y_test, predictions))


#min_max_scaler = MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(X_train)

#model_3 = LogisticRegression()
#model_3.fit(X_train, y_train)
#predictions = model_3.predict(X_test)
#print(accuracy_score(y_test, predictions))