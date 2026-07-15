import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Reading training and test data
train_file_path = './data/train.csv'
train_data = pd.read_csv(train_file_path)
test_file_path = './data/test.csv'
test_data = pd.read_csv(test_file_path)

# Cleaning data
# Encode genders by male to zero and female to one
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data = train_data.drop(['Cabin'],axis=1)
train_data = train_data.dropna()
test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})
# Filling missing age data with the median age value, which is 28
test_data['Age'] = test_data['Age'].fillna('28')

# Feature selection
chosen_features = ['Pclass','Sex','Age','SibSp','Parch']
X = train_data[chosen_features]
y = train_data['Survived']
X_test = test_data[chosen_features]

# Building and fitting model
model = DecisionTreeRegressor(random_state=1)
model.fit(X,y)

# Making predictions
y_pred = model.predict(X_test)
# Set cut-off
for i in range(len(y_pred)):
    y_pred[i] = 1 if (y_pred[i]>0.5) else 0

# Saving results to file
output_column_names = ['PassengerId','Survived']
output_data = pd.DataFrame(columns=output_column_names)
output_data['PassengerId'] = test_data['PassengerId']
output_data['Survived'] = y_pred.astype(int)
output_data.to_csv('./predictions/pred.csv', index=False)

