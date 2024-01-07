import pandas as pd
import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

filePath = 'house.csv'
df = pd.read_csv(filePath)

X = df[['TIME']].values.astype(float)
y = df['Value'].values.astype(float)
print(df[['TIME']].values.astype(float))
print(df['Value'].values.astype(float))
scalerX = MinMaxScaler()
XScaled = scalerX.fit_transform(X)
scalerY = MinMaxScaler()
YScaled = scalerY.fit_transform(y.reshape(-1, 1))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
model.fit(XScaled, YScaled, epochs=1000, verbose=0)
trainingPredictionsScaled = model.predict(XScaled)
trainingPredictions = scalerY.inverse_transform(trainingPredictionsScaled).flatten()
plt.scatter(df['TIME'], df['Value'], label='Actual')
plt.scatter(df['TIME'], trainingPredictions, label='Predicted (Training)', color='red')
plt.xlabel('Year')
plt.ylabel('₹ lakhs')
plt.title('Actual vs Predicted Values (Training)')
plt.legend()
plt.show()

futureYears = np.arange(2023, 2033).reshape(-1, 1)
futureYearsScaled = scalerX.transform(futureYears)
futureValuesScaled = model.predict(futureYearsScaled)
futureValues = scalerY.inverse_transform(futureValuesScaled).flatten()
plt.scatter(df['TIME'], df['Value'], label='Actual')
plt.scatter(df['TIME'], trainingPredictions, label='Predicted (Training)', color='red')
plt.scatter(futureYears.flatten(), futureValues, label='Predicted (Future)', color='green')
plt.xlabel('Year')
plt.ylabel('₹ lakhs')
plt.title('Actual, Predicted, and Future Values (Exponential Growth)')
plt.legend()
plt.show()

print('Predicted values for the next 10 years:')
for year, value in zip(futureYears.flatten(), futureValues):
    print(f'Year {year}: {value}')