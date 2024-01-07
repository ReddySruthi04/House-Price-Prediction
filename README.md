
# House Price Prediction
---

## Overview

The "House Price Prediction" project is a Python-based application that utilizes TensorFlow for machine learning and artificial intelligence to predict house prices for upcoming years based on historical data. The model is compiled using Mean Squared Error as the loss function.

## Data Source

The project uses data from the `house.csv` file, where the 'TIME' (year) and 'Value' (price at that year) are extracted using the pandas library.

## Dependencies

- Python
- TensorFlow
- Pandas
- Matplotlib (Pyplot)
- Numpy
- Scikit-learn (Min Max Scaler)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ReddySruthi04/House-Price-Prediction.git
    ```

2. Install dependencies:

    ```bash
    pip install tensorflow pandas matplotlib numpy scikit-learn
    ```

## Usage

1. Ensure you have the necessary dependencies installed.
2. Run the script:

    ```bash
    python house_price_prediction.py
    ```

3. View the output in the console and the generated graph.

## File Structure

- `house.csv`: Input data file containing historical house prices.
- `house_price_prediction.py`: Python script for training the TensorFlow model and predicting house prices.
- `README.md`: Project documentation.

## Implementation Details

The machine learning model is trained using TensorFlow, with Mean Squared Error as the loss function. The predictions are visualized using Matplotlib's pyplot library, both in the console and through a generated graph.

## Example Output

```
Year: 2022, Predicted Price: ₹XXXXXX units
Year: 2023, Predicted Price: ₹XXXXXX units
Year: 2024, Predicted Price: ₹XXXXXX units
...
```

### Sample 1:

```
sampleData1Secunderabad.csv
Enter the name of the area: Secunderabad
You entered: Secunderabad
Enter the units (per sq ft, per sq meter, per 100 sq meter):  per 100 sq.M
You entered units:  per 100 sq.M

1/1 [==============================] - 0s 86ms/step
1/1 [==============================] - 0s 138ms/step
Predicted values:
Year 2023, Predicted Price: ₹15156913.00 per 100 sq.M
Year 2024, Predicted Price: ₹16218570.00 per 100 sq.M
Year 2025, Predicted Price: ₹17167322.00 per 100 sq.M
Year 2026, Predicted Price: ₹18084206.00 per 100 sq.M
Year 2027, Predicted Price: ₹19001094.00 per 100 sq.M
Year 2028, Predicted Price: ₹19917980.00 per 100 sq.M
Year 2029, Predicted Price: ₹20834864.00 per 100 sq.M
```
![sample Data1 Secunderabad](images/sample1.png)
### Sample 2:
```
sampleData2SaketKapra.csv
Enter the name of the area: Saket Kapra   
You entered: Saket Kapra
Enter the units (per sq ft, per sq meter, per 100 sq meter):  per 100 sq.M
You entered units:  per 100 sq.M
1/1 [==============================] - 0s 166ms/step
1/1 [==============================] - 0s 112ms/step
Predicted values:
Year 2023, Predicted Price: ₹6147490.00 per 100 sq.M
Year 2024, Predicted Price: ₹7249057.00 per 100 sq.M
Year 2025, Predicted Price: ₹8279807.50 per 100 sq.M
Year 2026, Predicted Price: ₹9302948.00 per 100 sq.M
Year 2027, Predicted Price: ₹10326090.00 per 100 sq.M
Year 2028, Predicted Price: ₹11349229.00 per 100 sq.M
```
![sample Data2 SaketKapra](images/sample2.png)
### Sample 3:
```
sampleData3Malkajgiri.csv
Enter the name of the area: Malkajgiri
You entered: Malkajgiri
Enter the units (per sq ft, per sq meter, per 100 sq meter):  per 100 sq.M
You entered units:  per 100 sq.M
1/1 [==============================] - 0s 87ms/step
1/1 [==============================] - 0s 149ms/step
Predicted values:
Year 2023, Predicted Price: ₹7021180.50 per 100 sq.M
Year 2024, Predicted Price: ₹7508031.00 per 100 sq.M
Year 2025, Predicted Price: ₹7971773.50 per 100 sq.M
Year 2026, Predicted Price: ₹8435518.00 per 100 sq.M
Year 2027, Predicted Price: ₹8899261.00 per 100 sq.M
Year 2028, Predicted Price: ₹9363004.00 per 100 sq.M
Year 2029, Predicted Price: ₹9826748.00 per 100 sq.M
```
![sample Data3 Malkajgiri](images/sample3.png)
---