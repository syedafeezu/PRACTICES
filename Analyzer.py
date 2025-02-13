import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 

# params = { 'n_estimators': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#           'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#           'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],}


path = '/home/syedafeezu/PROJECTS/PRACTICES/titanic_train.csv'

data=pd.read_csv(path)

# sns.boxplot(x=data['Fare'])
# plt.show()
fare_cap=200
data['Fare']=data['Fare'].apply(lambda x: fare_cap if x>fare_cap else x)

# sns.boxplot(x=data['Fare'])
# plt.show()
data['Title']=data['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
 
data['Title']=data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
data['Title']=data['Title'].replace(['Mlle','Ms'],'Miss')
data['Title']=data['Title'].replace('Mme','Mrs')
 
data['Title']=data['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5})

data.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)
data['Age']=data['Age'].fillna(data['Age'].mean())
data['Embarked']=data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Sex']=data['Sex'].map({'male':1,'female':0})
data['Embarked']=data['Embarked'].map({'S':0,'C':1,'Q':2})
# data['Embarked']=pd.get_dummies(data['Embarked'],drop_first=True)


# scaler=StandardScaler()
# scaled_vals=scaler.fit_transform(data[['Fare','Age']])
# data['Fare']=scaled_vals[:,0]
# data['Age']=scaled_vals[:,1]
 
 
data['FamSize'] = data['SibSp'] + data['Parch'] + 1  
data['IsAlone']= 0
data.loc[data['FamSize'] == 1, ['IsAlone']] = 1  
# print(data.info())
 
X=data.drop('Survived',axis=1)
Y=data['Survived']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# model = GradientBoostingRegressor(n_estimators=100)
# model = GradientBoostingClassifier(n_estimators=100)
model = RandomForestClassifier()
# model = LogisticRegression( max_iter=2000)

# grid_search = GridSearchCV(model, params, cv=5)
# grid_search.fit(x_train, y_train)
# print(f'Best parameters:{grid_search.best_params_}')
model.fit(x_train,y_train)
 
pred=model.predict(x_test)
acc=accuracy_score(y_test,pred)
print(f'Accuracy: {acc*100}%')