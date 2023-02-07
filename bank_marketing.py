import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, \
    ConfusionMatrixDisplay, accuracy_score

file_name = 'bank.csv'
df = pd.DataFrame(data=pd.read_csv(file_name, delimiter=';'))
print(df)
print(df.corr(numeric_only=True))

# Check for dead columns
print('Null columns: {}'.format(df.columns[df.isnull().any()]))

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
df3 = pd.get_dummies(
    df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

df3_corr = df3.corr(numeric_only=True)

sns.heatmap(df3_corr, xticklabels=df3_corr.columns,
            yticklabels=df3_corr.columns, annot=True)

plt.show()

y = df3['y']
X = df3.drop(columns='y')

# random state to keep things reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

########################
# Logistic regression model seems to yield better accuracy
# than k-nearest neighbors model, how much so depends on the n_neighbors.
# Lower value -> lower accuracy
#
# Confusion matrices per model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_y_pred = logistic_model.predict(X_test)
log_conf_matrix = confusion_matrix(y_test, logistic_y_pred)
print("Confusion Matrix:\n{}".format(log_conf_matrix))
disp = ConfusionMatrixDisplay(
    confusion_matrix=log_conf_matrix, display_labels=logistic_model.classes_)
disp.plot()
log_accuracy = accuracy_score(y_test, logistic_y_pred)
print("Accuracy: {}".format(log_accuracy))
plt.show()

kneighbors_model = KNeighborsClassifier(n_neighbors=3)
kneighbors_model.fit(X_train, y_train)
kneighbors_y_pred = kneighbors_model.predict(X_test)
kn_conf_matrix = confusion_matrix(y_test, kneighbors_y_pred)
print("Confusion Matrix:\n{}".format(kn_conf_matrix))
disp = ConfusionMatrixDisplay(
    confusion_matrix=kn_conf_matrix, display_labels=logistic_model.classes_)
disp.plot()
kn_accuracy = accuracy_score(y_test, kneighbors_y_pred)
print("Accuracy: {}".format(kn_accuracy))
plt.show()
print('Size of data: {}'.format(len(df)))
