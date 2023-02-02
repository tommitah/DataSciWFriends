import pandas as pd
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
    y_train_pred_ridge = ridge_reg.predict(X_train)
    y_train_pred_lasso = lasso_reg.predict(X_train)
    scores_ridge.append(r2_score(y_train, y_train_pred_ridge))
    scores_lasso.append(r2_score(y_train, y_train_pred_lasso))

# NOTE: 6.
plt.plot(alphas, scores_ridge)
plt.title('Ridge regression, training data R2 to alpha.')
plt.xlabel('alphas')
plt.ylabel('r2 score')
plt.show()
plt.plot(alphas, scores_lasso)
plt.title('Lasso regression, training data R2 to alpha.')
plt.xlabel('alphas')
plt.ylabel('r2 score')
plt.show()


# lasso_reg = Lasso(alpha=alpha)
# lasso_reg.fit(X_train, y_train)

#####
# 2. Setup multiple regression X and y to predict 'mpg(miles per gallon)'
# of cars using all the variables except 'mpg', 'name' and 'origin'
#
# 3. split data into training and testing sets (80/20 split)
#
# 4. Implement both ridge regression and LASSO regression using
# several values for alpha.
#
# 5. Search optimal value for alpha (in terms of R2 score) by fitting the
# models with training data and computing the score using testing data
#
# 6. Plot the R2 scores for both regressons as functions of alpha
#
# 7. Identify, as accurately as you can, the value for alpha
# which gives the best score
#####
