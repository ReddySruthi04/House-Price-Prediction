import pandas as pd
import numpy as np
import tensorflow as tf
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = 'IndAnnual.csv'
df = pd.read_csv(file_path)

X = df[['TIME']].values.astype(float)
y = df['Value'].values.astype(float)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_scaled, y_scaled, epochs=100, verbose=0)
future_years = np.arange(2023, 2033).reshape(-1, 1)
future_years_scaled = scaler_X.transform(future_years)
future_values_scaled = model.predict(future_years_scaled)
future_values = scaler_y.inverse_transform(future_values_scaled).flatten()
plt.scatter(df['TIME'], df['Value'], label='Actual')
plt.plot(df['TIME'], scaler_y.inverse_transform(model.predict(X_scaled)).flatten(), label='Predicted (Training)', color='red')
plt.scatter(future_years.flatten(), future_values, label='Predicted (Future)', color='green')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Actual, Predicted, and Future Values')
plt.legend()
plt.show()
print('Predicted values for the next 10 years:')
for year, value in zip(future_years.flatten(), future_values):
    print(f'Year {year}: {value}')