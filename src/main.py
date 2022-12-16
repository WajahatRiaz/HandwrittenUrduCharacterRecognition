# Author: Wajahat Riaz 
# License: Apache-2.0

# Import classifiers and performance metri
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Standard scientific Python imports
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

data1 = np.empty([images_of_alif, DIMENSIONS], dtype = list)
for i in range(images_of_alif):
    data1[i] = dataset_alif[i]


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

data2 = np.empty([images_of_bay, DIMENSIONS], dtype = list)
for i in range(images_of_bay):
    data2[i] = dataset_bay[i]

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

data3 = np.empty([images_of_jeem, DIMENSIONS], dtype = list)
for i in range(images_of_jeem):
    data3[i] = dataset_jeem[i]

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


data4 = np.empty([images_of_daal, DIMENSIONS], dtype = list)
for i in range(images_of_daal):
    data4[i] = dataset_daal[i]

instances = images_of_alif + images_of_bay + images_of_jeem + images_of_daal

print("instances", instances)
print("dimension" , DIMENSIONS)

x = np.concatenate((data1,data2,data3,data4))

print("My X matrix of order", x.shape ,"is given as follows: ", x)

tag_alif = np.full((images_of_alif,1), 'A', dtype=str)

tag_bay = np.full((images_of_bay,1), 'B', dtype=str)

tag_jeem = np.full((images_of_jeem,1), 'J', dtype=str)

tag_daal = np.full((images_of_daal,1), 'D', dtype=str)


tag_vector = np.concatenate((tag_alif,tag_bay,tag_jeem,tag_daal))
print("My tags are:", tag_vector)

y = np.ravel(tag_vector, order='A')

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

model_1 = RandomForestClassifier(n_estimators=500)
model_1.fit(X_train, y_train)
predictions = model_1.predict(X_test)
print("Accuracy score of Random Forest Classifier:" , accuracy_score(y_test, predictions))
print(
    f"Classification report for Random Forest Classifier {model_1}:\n"
    f"{classification_report(y_test, predictions)}\n"
)
disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
disp.figure_.suptitle("Confusion matrix for Random Forest Classifier")
print(f"Confusion matrix for Random Forest Classifier:\n{disp.confusion_matrix}")
plt.show()

model_2 = SVC()
model_2.fit(X_train, y_train)
predictions = model_2.predict(X_test)
print("Accuracy score of SVM Classifier:" , accuracy_score(y_test, predictions))
print(
    f"Classification report for SVM Classifier {model_2}:\n"
    f"{classification_report(y_test, predictions)}\n"
)

disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
disp.figure_.suptitle("Confusion matrix for SVM Classifier")
print(f"Confusion matrix for SVM Classifier:\n{disp.confusion_matrix}")
plt.show()

scalar = StandardScaler()
X_train_scalar = scalar.fit_transform(X_train)
X_test_scalar = scalar.transform(X_test)

model_3 = LogisticRegression(solver='newton-cg',multi_class="ovr", max_iter=500)
model_3.fit(X_train_scalar, y_train)
predictions = model_3.predict(X_test_scalar)
print("Accuracy score of Logistic Regression:" , accuracy_score(y_test, predictions))
print(
    f"Classification report for Logistic Regression {model_3}:\n"
    f"{classification_report(y_test, predictions)}\n"
)

disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
disp.figure_.suptitle("Confusion matrix for Logistic Regression")
print(f"Confusion matrix for Logistic Regression:\n{disp.confusion_matrix}")
plt.show()
