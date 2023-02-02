import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

file_name = '50_Startups.csv'

df = pd.DataFrame(data=pd.read_csv(file_name))

#####
# Variables:
# R&D Spend
# Administration
# Marketing Spend
# State
# Profit
#
# R&D is the best predictor of Profit in this dataset.
# On the technical side, it is easy to lose track of what values you are
# supposed to be using and calculating for the linear regression.
# For the future: try to plot the data sets separately, don't rush
# with plotting all the scatter and lines for each one simultaneously
#####

print(df)
print(df.corr(numeric_only=True))

#####
# First plot the correlation between the variables with positive correlation.
df.plot.scatter(x='R&D Spend', y='Profit', color='red',
                title='Linear correlation between R&D spending and Profit')
plt.savefig('R&D_to_profit.pdf')
plt.show()
df.plot.scatter(x='Marketing Spend', y='Profit', color='blue',
                title='Linear correlation between '
                'Marketing spending and Profit')
plt.savefig('marketing_to_profit.pdf')
plt.show()

# Correlation between admin and profit seems negligible
df.plot.scatter(x='Administration', y='Profit', color='red',
                title='Linear correlation between Administration and Profit')
plt.show()
#####


#####
# Forming the training and testing data 80/20
# ... and plotting it
RD_SPEND_train, RD_SPEND_test, profit_train, profit_test = train_test_split(
    df['R&D Spend'].values, df['Profit'].values, test_size=0.2)
plt.scatter(RD_SPEND_train, profit_train, marker='+', color='green')
plt.scatter(RD_SPEND_test, profit_test, marker='+', color='blue')
plt.legend(['train', 'test'])
plt.title('R&D Spending to Profit 80/20 split')
plt.xlabel('R&D Spending ($)')
plt.ylabel('Profit ($)')
plt.savefig('R&D_spending_to_profit_datasplit.pdf')
plt.show()
#####


#####
# Linear regression
reg = LinearRegression().fit(RD_SPEND_train.reshape(-1, 1), profit_train)
profit_pred = reg.predict(RD_SPEND_test.reshape(-1, 1))

# plot the training and testing data
plt.scatter(RD_SPEND_train.reshape(-1, 1), profit_train, color='blue')
plt.scatter(RD_SPEND_test.reshape(-1, 1), profit_test, color='orange')
# plot the regression line
plt.plot(RD_SPEND_test, profit_pred, color='green')
plt.title('Linear regression for R&D spending to profit.')
plt.legend(['train', 'test', 'regression'])
plt.savefig('linear_regression_R&D_spending_to_profit.pdf')
plt.show()
#####


#####
# Fit for training and testing data
# need predictions for training set separately!
profit_train_pred = reg.predict(RD_SPEND_train.reshape(-1, 1))
R2_TRAIN = r2_score(profit_train, profit_train_pred)
R2_TEST = r2_score(profit_test, profit_pred)
RMSE_TRAIN = np.sqrt(mean_squared_error(profit_train, profit_train_pred))
RMSE_TEST = np.sqrt(mean_squared_error(profit_test, profit_pred))
print('R2 for training data: ', R2_TRAIN)
print('R2 for testing data: ', R2_TEST)
print('RMSE for training data: ', RMSE_TRAIN)
print('RMSE for testing data: ', RMSE_TEST)
#####
