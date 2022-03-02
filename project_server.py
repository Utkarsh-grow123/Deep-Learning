from flask import Flask, request, render_template
import dttest
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')

# @app.route('/on.html', methods=["GET","POST"])
# def input1():
#     if request.method == "POST":
#         id=request.form['value of Id'])
#     print(f"purchased: {}")
#
#     # name=request.form['Name']
#     # email=request.form['Email']
#     # password=request.form['Password']
#     #
#     # print(f"name : {name}")
#     # print(f"name : {name}")
#     # print(f"name : {name}")
#     return render_template('/predictions.html')

@app.route('/predictions.html', methods=["GET", "POST"])
def input():
    if request.method == 'GET':
        return render_template('/predictions.html')
    elif request.method == 'POST':
        algorithm = int(request.form['algorithm'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        dt = int(request.form['date'])
        # print(f"algorithm: {algorithm}")
        # print(f"year: {type(year)}")
        # print(f"month: {type(month)}")
        # print(f"date: {type(date)}")
        dateformat = f"{year}-{month}-{dt} 00:00:00"
        d = {'col1': [dateformat]}
        df = pd.DataFrame(data=d)
        df['col1'] = pd.to_datetime(df['col1'])
        # print(df.dtypes)
        # print(df)
        df['col1'] = df['col1'].values.astype(float)
        # print(df['col1'])
        input = df.iloc[:, 0].copy()
        input_value = input.values
        input0 = np.reshape(input_value, (len(input_value), 1))

        algorithm_name = ''
        mse = 0
        mae = 0
        y_pred = 0
        if algorithm == 0:
            algorithm_name = 'Linear regression'
            y_pred = dttest.regressor_LR.predict(input0)
            mse, mae = dttest.validation

        elif algorithm == 1:
            algorithm_name = 'XGboost'
            y_pred = dttest.regressor_XGB.predict(input0)
            mse, mae = dttest.validation2

        elif algorithm == 2:
            algorithm_name = 'Random Forest Regressor'
            y_pred = dttest.regressor_RF.predict(input0)
            mse, mae = dttest.validation3

        elif algorithm == 3:
            algorithm_name = 'RNN'
            y_pred = dttest.regressor_rnn.predict(input0)
            mse, mae = dttest.validation4

        # elif algorithm == 1:
        #     algorithm_name = 'LSTM'
        #     y_pred = dttest.XGBregressor.predict(input0)
        #     mse, mae = dttest.validation2
        #
        # elif algorithm == 1:
        #     algorithm_name = 'ARIMA'
        #     y_pred = dttest.XGBregressor.predict(input0)
        #     mse, mae = dttest.validation2

        # print(f"purchased: {purchased[0]}")

        return render_template("result.html", input0=input0, bitcoin_price=y_pred[0], Mean_square_error=mse,
                               Mean_absolute_error=mae,
                               algorithm=algorithm_name)

@app.route('/about.html', methods=["GET", "POST"])
def about():
    if request.method == 'GET':
        return render_template('about.html')

@app.route('/chartsanalysis.html',methods=["GET", "POST"])
def chart():
    if request.method == 'GET':
        return render_template('chartsanalysis.html')
    elif request.method == 'POST':
        choice = int(request.form['choice'])
        if choice == 10:
            return render_template('chart33.html')
        elif choice == 11:
            return render_template('chart44.html')
        elif choice == 12:
            return render_template('chart55.html')
        elif choice == 13:
            return render_template('chart66.html')
        elif choice == 14:
            return render_template('chart77.html')
        elif choice == 15:
            return render_template('chart11.html')
        elif choice == 16:
            return render_template('chart22.html')

app.run(port=8000, host='0.0.0.0', debug=True)
