import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('air_quality_index.csv')
df=df.drop_duplicates(keep='first')

#Checking for null values
df.isna().sum()
#We are dropping those rows where the status is null
df=df[df['Status'].isna()==False]

pollutants = ['PM2.5','PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene','Toluene', 'Xylene']
#Before handling the null values in the pollutants concentration with measure is of central,tendencies,lets check for outliers
df2=df[pollutants]

#Keeping a threshold value since there are many null values in each pollutants.Here we take 10%
limit=len(df)*0.01
for column in df.columns:
    if (df[column].isna().sum()<=limit) & (df[column].isna().sum())>0:
        df=df[df[column].isna()==False] 
        
#we are replacing null values with median.
df2=df[pollutants]
for i in df2.columns:
    df2[i]=df2[i].fillna(df2[i].median())
    
df=df.drop(['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene'],axis=1)
df=pd.concat([df,df2],axis=1)

#For handling outliers in pollutants
upper_limit=[]
lower_limit=[]
for i in df2.columns:
    Q1=np.percentile(df2[i],25)
    Q2=np.percentile(df2[i],50)
    Q3=np.percentile(df2[i],75)
    IQR=Q3-Q1
    UL=Q3+IQR*1.5
    LL=Q1-IQR*1.5
    upper_limit.append(UL)
    lower_limit.append(LL) 
    
j=0
for i in df2.columns:
    df[i]=np.where(df[i]>upper_limit[j],upper_limit[j],np.where(df[i]<lower_limit[j],lower_limit[j],df[i]))
    j=j+1
    
Q1=np.percentile(df['AQI'],25)
Q2=np.percentile(df['AQI'],50)
Q3=np.percentile(df['AQI'],75)
IQR=Q3-Q1
UL=Q3+IQR*1.5
LL=Q1-IQR*1.5
upper_limit.append(UL)
lower_limit.append(LL)
outliers_13=[]
for j in df['AQI']:
    if ((j>upper_limit[12]) |(j<lower_limit[12])):
        outliers_13.append(j)
        
ind_AQI = df[(df['AQI'] < lower_limit[12]) | (df['AQI'] > upper_limit[12])]
df['AQI']=np.where(df['AQI']>upper_limit[12],upper_limit[12],np.where(df['AQI']<lower_limit[12],lower_limit[12],df['AQI']))
    
#Lets drop the columns of StationId,Station Name,Status,Date
df=df.drop(['StationId','StationName','Status','Date','State','AQI_Bucket'],axis=1)

#From the correaltion matrix,NOx and NO shows a relationship value as 0.89,hence dropping NOx
df=df.drop(['NOx'],axis=1)

#Label encoding the categorical columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['City']=le.fit_transform(df['City'])
pickle.dump(le,open('city.pkl','wb') )

#Since the values are in different ranges,doing feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled=scaler.fit_transform(df)
col_names=df.columns
scaled=pd.DataFrame(scaled,columns=col_names)


#Splitting the data into test,train data
X=scaled.drop(['AQI'],axis=1)
y=scaled['AQI']

#Training the models with selected features
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

#Fitting with the models

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()
model=rf.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(X_test)
print(y_pred)
#Saving the model to disk
pickle.dump(rf,open('model.pkl','wb') )

        
