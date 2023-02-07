import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# constants
FILE_NAME = 'weight-height.csv'
N_NEIGHBORS = 5

df = pd.DataFrame(data=pd.read_csv(FILE_NAME))
print(df)
print(df.corr(numeric_only=True))

# Check for dead columns
if df.isnull().any().any():
    print('Null columns: {}'.format(df.columns[df.isnull().any()]))
    exit()


# Unit conversions and plot
X = df['Height'] * 2.54
y = df['Weight'] / 2.20462
plt.scatter(X, y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=30)

X_train = X_train.to_frame()
X_test = X_test.to_frame()


# normalization is good for skewed distribution or if algorithm requires
# that the data is on the same scale like neural networks
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# standardization is good for data that has a Gaussian distribution,
# or if algorithm is sensitive to the mean and std deviation.
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# KNN regression model for ORIGINAL data
kneighbors_model_og = KNeighborsRegressor(
    n_neighbors=N_NEIGHBORS).fit(X_train, y_train)
kn_r2_og_train = kneighbors_model_og.score(X_test, y_test)
print('R2 for KNN with k=5, ORIGINAL data:\t {}'.format(kn_r2_og_train))

# KNN reg for normalized data
kneighbors_model_norm = KNeighborsRegressor(
    n_neighbors=N_NEIGHBORS).fit(X_train_norm, y_train)
kn_r2_norm_train = kneighbors_model_norm.score(X_test_norm, y_test)
print('R2 for KNN with k=5, NORMALIZED data:\t {}'.format(kn_r2_norm_train))

# KNN reg for std'd data
kneighbors_model_std = KNeighborsRegressor(
    n_neighbors=N_NEIGHBORS).fit(X_train_std, y_train)
kn_r2_std_train = kneighbors_model_std.score(X_test_std, y_test)
print('R2 for KNN with k=5, STANDARDIZED data:\t {}'.format(kn_r2_std_train))

# NOTE: No performance difference between the models!
