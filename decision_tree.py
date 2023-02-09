import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

FILE_NAME = 'suv.csv'

df = pd.DataFrame(data=pd.read_csv(FILE_NAME))
print(df)
print(df.corr(numeric_only=True))

# Check for dead columns
if df.isnull().any().any():
    print('Null columns: {}'.format(df.columns[df.isnull().any()]))
    exit()

y = df['Purchased']
X = df.drop(columns=['User ID', 'Purchased', 'Gender'])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

entropy_model = tree.DecisionTreeClassifier(
    criterion='entropy').fit(X_train_std, y_train)
y_pred = entropy_model.predict(X_test_std)
print('ENTROPY MODEL:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

gini_model = tree.DecisionTreeClassifier(
    criterion='gini').fit(X_train_std, y_train)
y_pred = gini_model.predict(X_test_std)
print('GINI MODEL:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
