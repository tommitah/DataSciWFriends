import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

file_name = 'Auto.csv'
df = pd.DataFrame(data=pd.read_csv(file_name))
print(df)
print(df.corr(numeric_only=True))

# Check for dead columns
print('Null columns: {}'.format(df.columns[df.isnull().any()]))

# NOTE: 2. multiple regression model
m_reg = LinearRegression().fit(df[['cylinders', 'displacement', 'horsepower',
                                   'weight', 'acceleration']], df.mpg)

# NOTE: 3. training and testing data
features = df[['cylinders', 'displacement',
               'horsepower', 'weight', 'acceleration']].values
target = df['mpg'].values
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2)

# NOTE: 4&5. ridge and lasso regression models
# values for alpha
# Note that here we should be using training data instead of the actual data
alphas = [1, 3, 5, 7, 10, 15, 20]
scores_ridge = []
scores_lasso = []
for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha).fit(X_train, y_train)
    lasso_reg = Lasso(alpha=alpha).fit(X_train, y_train)
    # r2 score is computed with testing data
    # because it is supposed to test the model.
    # model is trained on the training data -> naturally it
    # fits the training data.
    # r2 is computed with testing data to get a more realistic
    # estimate of the model's performance on unseen data.
    y_test_pred_ridge = ridge_reg.predict(X_test)
    y_test_pred_lasso = lasso_reg.predict(X_test)
    scores_ridge.append(r2_score(y_test, y_test_pred_ridge))
    scores_lasso.append(r2_score(y_test, y_test_pred_lasso))

best_index_ridge = np.argmax(scores_ridge)
best_index_lasso = np.argmax(scores_lasso)

best_alpha_ridge = alphas[best_index_ridge]
best_alpha_lasso = alphas[best_index_lasso]

# NOTE: 6.
plt.plot(alphas, scores_ridge)
plt.title('Ridge regression, training data R2 to alpha.')
plt.xlabel('alphas')
plt.ylabel('r2 score')
plt.savefig('ridge.pdf')
plt.show()

plt.plot(alphas, scores_lasso)
plt.title('Lasso regression, training data R2 to alpha.')
plt.xlabel('alphas')
plt.ylabel('r2 score')
plt.savefig('lasso.pdf')
plt.show()

print('Best alpha for ridge: {}, with R2: {}'.format(
    alphas[best_index_ridge], scores_ridge[best_index_ridge]))
print('Best alpha for lasso: {}, with R2: {}'.format(
    alphas[best_index_lasso], scores_lasso[best_index_lasso]))
