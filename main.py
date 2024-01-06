# Step 1: Import necessary libraries
# pip install pandas numpy scikit-learn matplotlib tensorflow
import pandas as pd

# Step 2: Load the dataset
file_path = 'Delhi.csv'
data = pd.read_csv(file_path)

# Select relevant features and target variable
X = data[['Area']]
y = data['Price']
print(data[['Area']])
