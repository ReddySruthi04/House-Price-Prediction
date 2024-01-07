import pandas as pd
import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

filePath = 'house.csv'
# filePath = 'sampleData2SaketKapra.csv'
df = pd.read_csv(filePath)
X = df[['TIME']].values.astype(float)
y = df['Value'].values.astype(float)
print(df[['TIME']].values.astype(float))
print(df['Value'].values.astype(float))

print(filePath)
area_name = input("Enter the name of the area: ")
print("You entered:", area_name)
units = input("Enter the units (per sq ft, per sq meter, per 100 sq meter): ")
print(f"You entered units: {units}")

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
plt.scatter(df['TIME'], df['Value'].astype(float), label='Actual')
plt.scatter(df['TIME'], trainingPredictions, label='Predicted (Training)', color='red')
plt.xlabel('Year')
plt.ylabel('₹')
plt.title(area_name)
plt.legend()
plt.show()

futureYears = np.arange(2023, 2030).reshape(-1, 1)
futureYearsScaled = scalerX.transform(futureYears)
futureValuesScaled = model.predict(futureYearsScaled)
futureValues = scalerY.inverse_transform(futureValuesScaled).flatten()
plt.scatter(df['TIME'], df['Value'], label='Actual')
plt.scatter(df['TIME'], trainingPredictions, label='Predicted (Training)', color='red')
plt.scatter(futureYears.flatten(), futureValues, label='Predicted (Future)', color='green')
plt.xlabel('Year')
plt.ylabel('₹')
plt.title(area_name)
plt.legend()
plt.show()

print('Predicted values:')
for year, value in zip(futureYears.flatten(), futureValues):
    formatted_value = "{:.2f}".format(value)
    print(f'Year {year}, Predicted Price: ₹{formatted_value}{units}')