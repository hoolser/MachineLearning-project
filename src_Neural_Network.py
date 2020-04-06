import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from math import sin, cos, sqrt, atan2, radians
import datetime


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

dataset = pd.read_csv('C:/Users/tASO/Desktop/data/train.csv')

##########################################
R = 6373.0
dataset['dlon'] = np.radians(dataset['LongitudeArrival']) - np.radians(dataset.LongitudeDeparture)
dataset['dlat'] = np.radians(dataset['LatitudeArrival']) - np.radians(dataset.LatitudeDeparture)

dataset['a'] = np.sin(dataset["dlat"].astype(np.float64)/2)**2 + np.cos(np.radians(dataset.LatitudeDeparture).astype(np.float64)) * np.cos(np.radians(dataset['LatitudeArrival']).astype(np.float64)) * np.sin(dataset["dlon"].astype(np.float64)/2)**2
dataset['c'] = 2 *np.arctan2(np.sqrt(dataset['a']).astype(np.float64), np.sqrt(1 - dataset['a']).astype(np.float64)).astype(np.float64) #atan2 h arctan2
dataset['distance'] = R*dataset['c']

#print(dataset.head())
############################################

splitDate = dataset["DateOfDeparture"].str.split("-", n = 3, expand = True)
dataset['Day'] = splitDate[2]
dataset['Month'] = splitDate[1] 
dataset['Year'] = splitDate[0]
dataset['Month'] = dataset['Month'].astype(str).astype(int)
dataset['Day'] = dataset['Day'].astype(str).astype(int)
dataset['Year'] = dataset['Year'].astype(str).astype(int)

#dataset['dayOfmonth']
tempArr=np.zeros(dataset.shape[0])
for i in range(dataset.shape[0]):
    if(dataset.iloc[i]['Day']>10 and dataset.iloc[i]['Day']<=20  ):
        tempArr[i]=1
    elif(dataset.iloc[i]['Day']<=31 and dataset.iloc[i]['Day']>20):
        tempArr[i]=2
        
dataset['dayOfmonth']=tempArr        


Year = dataset['Year'].tolist()
Month = dataset['Month'].tolist()
Day = dataset['Day'].tolist() 
DOW = list(range(len(Year)))
for i in range(len(Year)):
    DOW[i] = datetime.datetime( Year[i], Month[i], Day[i] ).weekday()
DOWS = pd.Series(DOW)
dataset['DOW'] = DOWS.values

##############################################################################
tempweek = np.zeros(dataset.shape[0])

for i in range(dataset.shape[0]):
    if(dataset.iloc[i]['DOW']==5 or dataset.iloc[i]['DOW']==6 ):
        tempweek[i] = 1
dataset['WEEKDAY'] = tempweek


tempArr=np.zeros(dataset.shape[0])
for i in range(dataset.shape[0]):
    if(dataset.iloc[i]['Month']>=11 and dataset.iloc[i]['Month']<=1):
        tempArr[i]=0 #xeimwnas
    elif(dataset.iloc[i]['Month']>=2 and dataset.iloc[i]['Month']<=4):
        tempArr[i]=1 # anoiksh
    elif (dataset.iloc[i]['Month']>=5 and dataset.iloc[i]['Month']<=7):
        tempArr[i]=2 # kalokairi
    elif (dataset.iloc[i]['Month']>=8 and dataset.iloc[i]['Month']<=10):
        tempArr[i]=3 # fthiniporo
dataset['Season']=tempArr
##############################################################################

dataset['wtd/std_wtd'] = dataset['WeeksToDeparture'] / dataset['std_wtd']

#######################################################################################################################################################################################################
##### GIA SUUUUUUUUUUUUUUUUUUUUBBBBBBBBBBBBBBBBBBBBBBBBBB <------------------------------------------------
df_test = pd.read_csv('C:/Users/tASO/Desktop/data/test.csv')

#########################################
R = 6373.0
df_test['dlon'] = np.radians(df_test['LongitudeArrival']) - np.radians(df_test.LongitudeDeparture)
df_test['dlat'] = np.radians(df_test['LatitudeArrival']) - np.radians(df_test.LatitudeDeparture)

df_test['a'] = np.sin(df_test["dlat"].astype(np.float64)/2)**2 + np.cos(np.radians(df_test.LatitudeDeparture).astype(np.float64)) * np.cos(np.radians(df_test['LatitudeArrival']).astype(np.float64)) * np.sin(df_test["dlon"].astype(np.float64)/2)**2
df_test['c'] = 2 *np.arctan2(np.sqrt(df_test['a']).astype(np.float64), np.sqrt(1 - df_test['a']).astype(np.float64)).astype(np.float64) #atan2 h arctan2
df_test['distance'] = R*df_test['c']

#print(df_test.head())
###########################################

splitDate2 = df_test["DateOfDeparture"].str.split("-", n = 3, expand = True)
df_test['Day'] = splitDate2[2]
df_test['Month'] = splitDate2[1] 
df_test['Year'] = splitDate2[0]
df_test['Month'] = df_test['Month'].astype(str).astype(int)
df_test['Day'] = df_test['Day'].astype(str).astype(int)
df_test['Year'] = df_test['Year'].astype(str).astype(int)

tempArr=np.zeros(df_test.shape[0])
for i in range(df_test.shape[0]):
    if(df_test.iloc[i]['Day']>10 and df_test.iloc[i]['Day']<=20  ):
        tempArr[i]=1
    elif(df_test.iloc[i]['Day']<=31 and df_test.iloc[i]['Day']>20):
        tempArr[i]=2
        
df_test['dayOfmonth']=tempArr        

Year = df_test['Year'].tolist()
Month = df_test['Month'].tolist()
Day = df_test['Day'].tolist() 
DOW = list(range(len(Year)))
for i in range(len(Year)):
    DOW[i] = datetime.datetime( Year[i], Month[i], Day[i] ).weekday()
DOWS = pd.Series(DOW)
df_test['DOW'] = DOWS.values

##############################################################################
tempweek = np.zeros(df_test.shape[0])

for i in range(df_test.shape[0]):
    if(df_test.iloc[i]['DOW']==5 or df_test.iloc[i]['DOW']==6 ):
        tempweek[i] = 1
df_test['WEEKDAY'] = tempweek


tempArr=np.zeros(df_test.shape[0])
for i in range(df_test.shape[0]):
    if(df_test.iloc[i]['Month']>=11 and df_test.iloc[i]['Month']<=1):
        tempArr[i]=0 #xeimwnas
    elif(df_test.iloc[i]['Month']>=2 and df_test.iloc[i]['Month']<=4):
        tempArr[i]=1 # anoiksh
    elif (df_test.iloc[i]['Month']>=5 and df_test.iloc[i]['Month']<=7):
        tempArr[i]=2 # kalokairi
    elif (df_test.iloc[i]['Month']>=8 and df_test.iloc[i]['Month']<=10):
        tempArr[i]=3 # fthiniporo
df_test['Season']=tempArr
##############################################################################

df_test['wtd/std_wtd'] = df_test['WeeksToDeparture'] / df_test['std_wtd']

df_test['DateOfDeparture'] = splitDate2[0] + '-' + splitDate2[1] + '-' + splitDate2[2]

import holidays 
hol = holidays.CA() + holidays.US() + holidays.MX() + holidays.TAR()
h = np.zeros(df_test.shape[0])
for i in range(df_test.shape[0]):
    if(hol.get(df_test.iloc[i]['DateOfDeparture']) is not None):
        h[i] = 1

df_test['holiday'] = h 

df_test.drop(df_test.columns[[0,2,3,4,6,7,8,9,10]], axis=1, inplace=True)
df_test.drop(df_test.columns[[2,3]], axis=1, inplace=True) #gia smote   

df_test['dep-arr'] = df_test['Departure'].astype(str) + df_test['Arrival']


le = LabelEncoder()
df_test['Departure'] = le.fit_transform(df_test['Departure'])
df_test['Arrival'] = le.fit_transform(df_test['Arrival'])
df_test['dep-arr'] = le.fit_transform(df_test['dep-arr'])
#####################################################normalize
from sklearn import preprocessing
x = df_test[['distance']] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_test['distance'] = x_scaled
###################################################
#####################################################################################################################################################################################################

y_train = dataset[['PAX']]

dataset['DateOfDeparture'] = splitDate[0] + '-' + splitDate[1] + '-' + splitDate[2]

import holidays 
hol = holidays.CA() + holidays.US() + holidays.MX() + holidays.TAR()
h = np.zeros(dataset.shape[0])
for i in range(dataset.shape[0]):
    if(hol.get(dataset.iloc[i]['DateOfDeparture']) is not None):
        h[i] = 1

dataset['holiday'] = h       
dataset['dep-arr'] = dataset['Departure'].astype(str) + dataset['Arrival']
  

dataset.drop(dataset.columns[[0,2,3,4,6,7,8,9,10,11]], axis=1, inplace=True)
#dataset.drop(dataset.columns[[2,3,4,5,9]], axis=1, inplace=True) #gia smote
#dataset.drop(dataset.columns[[11]], axis=1, inplace=True) #gia smote
dataset.drop(dataset.columns[[2,3]], axis=1, inplace=True) #gia smote



le = LabelEncoder()
dataset['Departure'] = le.fit_transform(dataset['Departure'])
dataset['Arrival'] = le.fit_transform(dataset['Arrival'])
dataset['dep-arr'] = le.fit_transform(dataset['dep-arr'])


#####################################################normalize
from sklearn import preprocessing
x = dataset[['distance']] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dataset['distance'] = x_scaled
####################################################


###gia topika##########################################################################################
#dataset, df_test, y_train, y_test = train_test_split(dataset, y_train, test_size=0.2, random_state=42)


#print(dataset.head())

#df_test.drop(df_test.columns[[0,2,3,4,6,7,8,9,10]], axis=1, inplace=True) ############gia sub 
#df_test.drop(df_test.columns[[0,2,3,4,6,7,8,9,10,11]], axis=1, inplace=True) ##############gia topika 

#
#print(df_test.head())
#
#le = LabelEncoder()
#dataset['Departure'] = le.fit_transform(dataset['Departure'])
#dataset['Arrival'] = le.fit_transform(dataset['Arrival'])
#dataset['dep-arr'] = le.fit_transform(dataset['dep-arr'])
#df_test['Departure'] = le.fit_transform(df_test['Departure'])
#df_test['Arrival'] = le.fit_transform(df_test['Arrival'])
#df_test['dep-arr'] = le.fit_transform(df_test['dep-arr'])


########################## OVER SAMPLE MAZI ME UNDER SAMPLE        
#from imblearn.combine import SMOTETomek        
#smt = SMOTETomek(ratio='auto')
#dataset, y_train = smt.fit_sample(dataset, y_train)
#dataset = pd.DataFrame(dataset)
##df_train.columns = ['Departure', 'Arrival', 'dlon', 'dlat', 'a', 'c', 'distance', 'Day', 'Month', 'DOW', 'wtd/std_wtd'] 
##dataset.columns = ['Departure', 'Arrival', 'distance', 'Day', 'Month', 'DOW', 'wtd/std_wtd'] 
#dataset.columns = ['Departure', 'Arrival', 'distance', 'Day', 'Month', 'DOW', 'wtd/std_wtd', 'holiday', 'dep-arr'] 
#
#
#
################################ Round up
#for i in range(0 , dataset.shape[0]):
#    for j in range(0 , dataset.shape[1]):
#        if(j != 2 and j != 6):
#            dataset.iloc[i,j] = round(dataset.iloc[i,j])    
###########################

################################################################################################################################### SMOTE
#trainBounds = dataset.shape[0]
#
####################################### SMOTE
shapeTrain = dataset.shape[0]
cols = list(dataset)        
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
dataset, y_train = smote.fit_sample(dataset, y_train)

shapeSmote = dataset.shape[0]


for i in range(round((shapeSmote - shapeTrain) * 5/6)):
    
    dataset = dataset[:-1]
    y_train = y_train[:-1]
    
    
dataset = pd.DataFrame(dataset)
#df_train.columns = ['Departure', 'Arrival', 'dlon', 'dlat', 'a', 'c', 'distance', 'Day', 'Month', 'DOW', 'wtd/std_wtd'] 
#dataset.columns = ['Departure', 'Arrival', 'distance', 'Day', 'Month', 'DOW', 'wtd/std_wtd'] 
#dataset.columns = ['Departure', 'Arrival', 'distance', 'Day', 'Month', 'DOW', 'wtd/std_wtd', 'holiday', 'dep-arr']
#dataset.columns = ['Departure', 'Arrival', 'distance', 'Day', 'Month', 'dayOfmonth', 'DOW', 'wtd/std_wtd', 'holiday', 'dep-arr']  
#dataset.columns = ['distance', 'Day', 'Month', 'DOW', 'wtd/std_wtd', 'holiday', 'dep-arr'] 
dataset.columns = cols



############################### Round up
for i in range(0 , dataset.shape[0]):
    for j in range(0 , dataset.shape[1]):
#        if(j != 2 and j != 7):
#        print(dataset.columns[j] != 'distance')
        if(dataset.columns[j] != 'distance' and dataset.columns[j] != 'wtd/std_wtd' and dataset.columns[j] != 'a' and dataset.columns[j] != 'c'):            
            dataset.iloc[i,j] = round(dataset.iloc[i,j])    
###################################################################################################################################
X_train = dataset
X_test = df_test
y_train = np.ravel(y_train)



enc = OneHotEncoder(sparse=False)
enc.fit(dataset)  
X_train = enc.transform(dataset)
X_test = enc.transform(df_test)



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer  ##gia to outputDim (Units) = (2+8)/2 = 5
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 3))
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=X_train.shape[1], units=(190))) #120
#classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=55, units=40))



###gia to overfitting
from keras.layers import Dropout
classifier.add(Dropout(0.30))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=(80))) #50
#classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=25))


###gia to overfitting
classifier.add(Dropout(0.30))


#### Adding the third hidden layer
##classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
#classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=(30)))
##classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=15))
#
####gia to overfitting
#classifier.add(Dropout(0.30))

## Adding the output layer
##classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))
#classifier.add(Dense(kernel_initializer="uniform", activation="softmax", units=8))

# Adding the output layer
#classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))
classifier.add(Dense(kernel_initializer="uniform", activation="softmax", units=8))


# Compiling Neural Network
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(encoded_Y)

# Fitting our model 
classifier.fit(X_train, dummy_y,  epochs = 30)

# Predicting the Test set results
y_pred = classifier.predict(X_test) 

#####gia na dw ton y_pred se csv
##with open('kkkkkkkkkk.csv', 'w') as writeFile:
##    writer = csv.writer(writeFile)
##    writer.writerows(y_pred)
##writeFile.close()
###################################
##

#####################################
yPRED = np.zeros(y_pred.shape[0])
for i in range(y_pred.shape[0]):
#    print(y_pred[i])
    yPRED[i]=np.argmax(y_pred[i])
#    print(np.argmax(y_pred[i]))
#####################################
    
with open('C:/Users/tASO/Desktop/data/y_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Label'])
    for i in range(yPRED.shape[0]):
        writer.writerow([i, yPRED[i].astype(np.int)])    
        
#print(f1_score(y_test, yPRED, average='micro'))
#from sklearn.metrics import confusion_matrix 
#
#cm = confusion_matrix(y_test, yPRED)