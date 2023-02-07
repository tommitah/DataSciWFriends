import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, \
    classification_report

file_name = 'data_banknote_authentication.csv'
df = pd.DataFrame(data=pd.read_csv(file_name))
print(df)
print(df.corr(numeric_only=True))

# Check for dead columns
print('Null columns: {}'.format(df.columns[df.isnull().any()]))

# 'class' only holds binary values, that is 0 or 1
y = df['class']
X = df.drop(columns='class')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20)


sv_clf = SVC(kernel='linear').fit(X_train, y_train)
y_pred = sv_clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)
print('Confusion matrix:\n{}'.format(conf_matrix))
print('Classification report:\n{}'.format(clf_report))


# This seems to be perfectly accurate? IDK weirdge.
rbf_clf = SVC(kernel='rbf').fit(X_train, y_train)
y_pred = rbf_clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)
print('Confusion matrix:\n{}'.format(conf_matrix))
print('Classification report:\n{}'.format(clf_report))
