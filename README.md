# Bitcoin-Predict
##### Project is currently under development
![Docker](https://github.com/Naman1997/Bitcoin-Predict/workflows/Docker/badge.svg)

Predict bitcoin valuation from older data.
Models currently working:
  - RNN
  - Linear Regression

Get data sources from:
[HERE](https://www.kaggle.com/mczielinski/bitcoin-historical-data)

## To run the project
First make the pickle as described in SP.ipynb and SP.ipynb
Next build the project again with docker build. This will add the required pickle to the image.
Next run the image as shown below:

$ sudo docker run -p 8080:8080  Naman1997/Bitcoin-Predict
then go to
http://localhost:8080/defaults
