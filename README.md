
# TSLA stock prediction algorithm

For this mandatory task, I chose to make a prediction algorithm for the TSLA stock. We were given the choice to either make a regression algorithm, or a classification algorithm. I chose to make a regression algorithm is because stock price prediction, fundamentally is a regression problem, since it estimates a linear trend line from the first price to the last. This wil increase the interpretability as the rise/sink will be constant. Lastly it is a simple model which may be sufficient to interpret stock price prediction on a simple level. 

###
First we import the necesarry modules for the program. Last line is only used on jupyter notebooks. If you copy this to a normal python program, you will have to comment/remove this line.
```Python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
%matplotlib inline
```

Now we want to load the raw data from the github to a new dataframe. I would also recommend to print the dataframe and generate the descriptive statistics.
```Python
url = 'https://raw.githubusercontent.com/atikagondal/Assignment-2-dave3625-202323/main/TSLA.csv'
Data = pd.read_csv(url, sep=',')
Data.head()
Data.describe()
```

Now we will convert the date column to a pandas datetime object. Then we calculate the number of days since first day of dataset, which we later will use to check how accurate the analysis is for a certain date.
``` Python
Data['Date'] = pd.to_datetime(Data['Date'])
Data['Days'] = (Data['Date'] - Data['Date'].min()).dt.days
```

I reccomend to create a plot for the close price according to the dataframe.
``` Python
Close = Data['Close']
ax = Close.plot(title='TSLA stock')
ax.set_xlabel('Date')
ax.set_ylabel('Close price')
plt.show()
```
Now we will start to model and train the data.
```Python
X = Data[['Days']].values
Y = Data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
```

We will now ask for a user input as a date. We wil then calculate the actual price at the closest date, the predicted price and finally the prection percentage score. Last, but not least we will neatly print the values mentioned above.

``` Python
user_date = input("Enter a date (yyyy-mm-dd) for stock price prediction: ")
user_date_datetime = pd.to_datetime(user_date)
closest_date = Data['Date'].iloc[(Data['Date'] - user_date_datetime).abs().argsort()[0]]
predicted_price = model.predict([[Data[Data['Date'] == closest_date]['Days'].values[0]]])[0]
actual_price = Data[Data['Date'] == closest_date]['Close'].values[0]
percentage_score = (predicted_price / actual_price) * 100
print(f"Predicted Price: ${predicted_price:.2f}")
print(f"Actual Price: ${actual_price:.2f}")
print(f"Prediction Percentage Score: {percentage_score:.2f}%")
```

## Author

- [HaavardAass](https://github.com/HaavardAass)

