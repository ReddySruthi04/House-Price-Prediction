import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
model.fit(X_scaled, y_scaled, epochs=1000, verbose=0)
training_predictions_scaled = model.predict(X_scaled)
training_predictions = scaler_y.inverse_transform(training_predictions_scaled).flatten()
plt.scatter(df['TIME'], df['Value'], label='Actual')
plt.scatter(df['TIME'], training_predictions, label='Predicted (Training)', color='red')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values (Training)')
plt.legend()
plt.show()
future_years = np.arange(2023, 2033).reshape(-1, 1)
future_years_scaled = scaler_X.transform(future_years)
future_values_scaled = model.predict(future_years_scaled)
future_values = scaler_y.inverse_transform(future_values_scaled).flatten()
plt.scatter(df['TIME'], df['Value'], label='Actual')
plt.scatter(df['TIME'], training_predictions, label='Predicted (Training)', color='red')
plt.scatter(future_years.flatten(), future_values, label='Predicted (Future)', color='green')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Actual, Predicted, and Future Values (Exponential Growth)')
plt.legend()
plt.show()

print('Predicted values for the next 10 years:')
for year, value in zip(future_years.flatten(), future_values):
    print(f'Year {year}: {value}')