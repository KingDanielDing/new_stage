# @title
# Retrieve stock data from yFinance
import yfinance as yf
import contextlib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from flask import Flask, jsonify, request
from datetime import date, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

def retrieve_data(filename, end_date, start_date, interval="1d"):
    def suppress_output(func, *args, **kwargs):
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            return func(*args, **kwargs)

    # Fetch and prepare stock data
    # input parameters: filename (stock symbol), end_date, start_date
    # output is a DataFrame [['Date', 'Open', 'High', 'Low', 'Close', 'Volume']] without index
    print("retrieve data from here")
    # Suppress the output of the yfinance download function
    df = suppress_output(yf.download, filename, start=start_date, end=end_date, interval=interval).reset_index()
    print("retrieve data successful")
    # Check if the column is named 'Datetime' and rename it to 'Date'
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})

    print("stop 2")

    # Ensure 'Date' is a column and not the index
    if df.index.name == 'Date':
        df = df.reset_index()

    print("stop3")

    # Ensure the DataFrame has the required columns and rearrange if needed
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    print(df.head())
    # df.columns = df.columns.droplevel(1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)


    print("stop 4")
    # Round the numeric columns to 3 decimal places
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(3)
    print("now we are alomost done before retun the stock data")

    return df

# Retrieve Stock data from AKShare
# import akshare as ak
import pandas as pd

def retrieve_ak_daily_data(mycode,end_date,start_date):
    stock_code = mycode.split('.')[0]  # Ping An Bank (A-share)
    adjust = "qfq"         # "qfq" for 前复权；"hfq": 后复权；"none": Raw prices.

    # Fetch daily historical stock data
    data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust=adjust)

    # Convert the '日期' column to datetime format for easier filtering
    data['日期'] = pd.to_datetime(data['日期'])

    # Filter the data by the specified date range
    filtered_data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]

    new_column_names = {filtered_data.columns[i]: f"col_{i+1}" for i in range(len(filtered_data.columns))}

    # Renaming columns
    filtered_data = filtered_data.rename(columns=new_column_names)
    filtered_data = filtered_data.drop(columns=['col_2','col_8','col_9','col_10','col_11','col_12'])
    filtered_data = filtered_data.rename(columns={'col_1':'Date','col_3':'Open','col_4':'Close','col_5':'High','col_6':'Low','col_7':'Volume'})

    # Return the filtered data
    return filtered_data.reset_index(drop=True)

def retrieve_ak_minute_data(mycode,end_date,start_date,period="15m"):
    stock_code = mycode.split('.')[0]  # Ping An Bank (A-share)
    adjust = "qfq"         # "qfq" for 前复权；"hfq": 后复权；"none": Raw prices.

    # Fetch daily historical stock data
    data = ak.stock_zh_a_hist_min_em(symbol=stock_code, period=period)

    new_column_names = {data.columns[i]: f"col_{i+1}" for i in range(len(data.columns))}

    # Renaming columns
    data = data.rename(columns=new_column_names)

    data = data.drop(columns=['col_6','col_7','col_9','col_10','col_11'])
    data = data.rename(columns={'col_1':'Date','col_2':'Open','col_3':'Close','col_4':'High','col_5':'Low','col_8':'Volume'})

    # Convert the '日期' column to datetime format for easier filtering
    data['Date'] = pd.to_datetime(data['Date'])

    # Filter the data by the specified date range
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    # Return the filtered data
    return filtered_data.reset_index(drop=True)

# Retrieve stock data from baostock
# import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime
def retrieve_bao_stock_data(mycode, end_minute_date, start_minute_date, freq="5"):
    def convert_time_format(time_string):
        # Ensure time_string is a string; if it's a pandas Timestamp, convert it first
        if isinstance(time_string, pd.Timestamp):
            time_string = time_string.strftime('%Y%m%d%H%M%S')
        else:
            # If it's already a string, ensure it is in the correct format
            time_string = str(time_string)

        # Take only the first 14 characters: "20240913144500"
        clean_time_string = time_string[:14]
        # Convert the string to a datetime object
        dt_object = datetime.strptime(clean_time_string, "%Y%m%d%H%M%S")
        # Convert it to a pandas Timestamp object
        timestamp = pd.Timestamp(dt_object)
        return timestamp

    # Step 1: Determine stock code format
    if mycode.split('.')[1] == "ss":
        stock_code = "sh." + mycode.split('.')[0]
    elif mycode.split('.')[1] == "sz":
        stock_code = "sz." + mycode.split('.')[0]
    else:
        return None

    # Step 2: Login to Baostock
    lg = bs.login()

    # Check if login was successful
    if lg.error_code != "0":
        print(f"Login failed: {lg.error_msg}")
        return None  # Exit function if login fails

    # Step 3: Query 5-minute stock data
    start_date = start_minute_date
    end_date = end_minute_date

    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,time,code,open,high,low,close,volume",
        start_date=start_date, end_date=end_date,
        frequency=freq,  # "5" for 5-minute data
        adjustflag="3"   # 1 前复权；2 后复权； 3 不复权
    )

    # Step 4: Check for query success
    if rs.error_code != "0":
        print(f"Query failed: {rs.error_msg}")
        bs.logout()
        return None  # Exit function if query fails

    # Step 5: Process the results
    data_list = []
    while rs.next():  # Go through each row of data
        data_list.append(rs.get_row_data())

    # Check if the data list is empty
    if not data_list:
        print("No data found for the given query.")
        bs.logout()
        return None

    # Convert the result to a Pandas DataFrame
    result = pd.DataFrame(data_list, columns=rs.fields)

    # Step 6: Logout from Baostock
    bs.logout()

    # Step 7: Apply the conversion function to the "time" column
    result['time'] = result['time'].apply(convert_time_format)

    # Step 8: Select and rename columns
    mystock = result[['time', 'open', 'high', 'low', 'close', 'volume']]
    mystock = mystock.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

    # Step 9: Convert data types to floats
    mystock['Close'] = mystock['Close'].astype(np.float64)
    mystock['Open'] = mystock['Open'].astype(np.float64)
    mystock['High'] = mystock['High'].astype(np.float64)
    mystock['Low'] = mystock['Low'].astype(np.float64)
    mystock['Volume'] = mystock['Volume'].astype(np.float64)

    return mystock.reset_index(drop=True)

def retrieve_bao_stock_daily(mycode,end_date,start_date):

    # Step 1: Determine stock code format
    if mycode.split('.')[1] == "ss":
        stock_code = "sh." + mycode.split('.')[0]
    elif mycode.split('.')[1] == "sz":
        stock_code = "sz." + mycode.split('.')[0]
    else:
        return None

    # Login to the system
    lg = bs.login()
    if lg.error_code != '0':
        print(f'Login failed, error_code: {lg.error_code}, error_msg: {lg.error_msg}')
    else:
        print(f'Login success, user_id: {lg.user_id}')

    # Retrieve daily stock data (e.g., stock code "sh.600000" for Shanghai stock exchange)
    data_fields = "date,open,high,low,close,preclose,volume"

    rs = bs.query_history_k_data_plus(stock_code,
                    data_fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",  # Daily data
                    adjustflag="3") # 1 前复权；2 后复权； 3 不复权

    # Convert data to a DataFrame
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())

    # Create DataFrame
    df = pd.DataFrame(data_list, columns=rs.fields)

    # Logout after retrieval
    bs.logout()

    # Step 8: Select and rename columns
    mystock = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    mystock = mystock.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

    # Step 9: Convert data types to floats
    mystock['Close'] = pd.to_numeric(mystock['Close'],errors='coerce')
    mystock['Open'] = pd.to_numeric(mystock['Open'],errors='coerce')
    mystock['High'] = pd.to_numeric(mystock['High'],errors='coerce')
    mystock['Low'] = pd.to_numeric(mystock['Low'],errors='coerce')
    mystock['Volume'] = pd.to_numeric(mystock['Volume'],errors='coerce')
    # mystock['Close'] = mystock['Close'].astype(np.float64)
    # mystock['Open'] = mystock['Open'].astype(np.float64)
    # mystock['High'] = mystock['High'].astype(np.float64)
    # mystock['Low'] = mystock['Low'].astype(np.float64)
    # mystock['Volume'] = mystock['Volume'].astype(np.float64)

    return mystock.reset_index(drop=True)

# @title
import sys
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import warnings
# warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

def new_find_fvgs(mystock):
    # with 1% buffer
    df = mystock.copy()
    # Ensure that df is sorted by date
    df = df.sort_values('Date').reset_index(drop=True)

    # Create lists to store FVG information
    bullish_fvg = []
    bearish_fvg = []

    # Loop through the DataFrame to identify FVGs
    for i in range(2, len(df)):
        prev_high = df.loc[i-2, 'High']
        prev_low = df.loc[i-2, 'Low']
        current_high = df.loc[i, 'High']
        current_low = df.loc[i, 'Low']

        # Check for Bullish FVG
        if prev_low / current_high > 1.01:  # 1% buffer
            bullish_fvg.append({
                'Date': df.loc[i, 'Date'],
                'FVG_Type': 'Bullish',
                'Prev_Low': prev_low,
                'Current_High': current_high
            })

        # Check for Bearish FVG
        if current_low / prev_high > 1.01:  # 1% buffer
            bearish_fvg.append({
                'Date': df.loc[i, 'Date'],
                'FVG_Type': 'Bearish',
                'Prev_High': prev_high,
                'Current_Low': current_low
            })

    # Combine the FVG data into a DataFrame
    fvg_df = pd.DataFrame(bullish_fvg + bearish_fvg)

    # Add 'Benchmark' and "Bench" column
    if not fvg_df.empty:  # Make sure fvg_df is not empty
        fvg_df['Bench'] = fvg_df.apply(
            lambda row: row['Current_High'] if row['FVG_Type'] == 'Bullish' else row['Current_Low'],
            axis=1
        )

    if not fvg_df.empty:  # Make sure fvg_df is not empty
        fvg_df['Benchmark'] = fvg_df.apply(
            lambda row: row['Prev_Low'] if row['FVG_Type'] == 'Bullish' else row['Prev_High'],
            axis=1
        )

    return fvg_df

def new_generate_finals(mystock,fvg_df,end_trade_date,start_trade_date="2022-01-01"):
    df = mystock.copy()

    # Convert 'Date' columns to timezone-naive to ensure compatibility
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    fvg_df['Date'] = pd.to_datetime(fvg_df['Date']).dt.tz_localize(None)

    # Now merge df and new_smi
    # merged_adf = pd.merge(df, new_smi, on='Date', how='left')[['Date','Close','High','Low','Open','SMI','Signal_Line']]
    merged_df = pd.merge(df, fvg_df, on='Date', how='left')[['Date','Close','High','Low','Open','Benchmark','Bench']]
    merged_df = merged_df[(merged_df['Date'] >= start_trade_date) & (merged_df['Date'] <= end_trade_date)].reset_index(drop=True)

    merged_df['Benchmark'] = merged_df['Benchmark'].ffill()
    merged_df['Bench'] = merged_df['Bench'].ffill()

    return merged_df

def generate_pure_stages(df):

      def stage_assessment(data_dictionary):
          benchmark = data_dictionary['benchmark']
          bench = data_dictionary['bench']
          close = data_dictionary['close']

          stage = 0

          if close <= bench <= benchmark:
            stage = 1
          elif close <= benchmark < bench:
            stage = 3
          elif benchmark < bench < close:
            stage = 6
          elif bench <= benchmark < close:
            stage = 4
          elif bench < close <= benchmark:
            stage = 2
          elif benchmark < close <= bench:
            stage = 5
          elif close == bench == benchmark:
            stage = 3.5

          return stage


      final = df.copy()
      merged_df = final[['Date','Close','Benchmark','Bench']]

      # Generate Signals for merged_df
      my_stage = [0]
      for i in range(1,len(merged_df)):
          the_date = merged_df['Date'].iloc[i]
          the_close = merged_df['Close'].iloc[i]
          the_bench = merged_df['Bench'].iloc[i]
          the_benchmark = merged_df['Benchmark'].iloc[i]

          summary_dict = {
              "close": the_close,
              "bench": the_bench,
              "benchmark": the_benchmark,
          }

          stage = stage_assessment(summary_dict)
          my_stage.append(stage)

      merged_df['Stage'] = my_stage

      return merged_df

def daily_knn_model(states,start_daily_date,end_daily_date):

    # Prepare the dataset for KNN
    X = []
    y = []

    # Define which states are "good" and "bad"
    good_states = [4, 5, 6]
    bad_states = [1, 2, 3]

    for i in range(len(states) - 10):
        last_10_states = states['Stage'][i:i + 10].to_numpy()
        next_state = states['Stage'][i + 10]

        # Ensure that the sliding window and next state are valid (not missing)
        if len(last_10_states) == 10 and not np.isnan(next_state):
            # Only append if the next_state is within the known categories
            if next_state in good_states or next_state in bad_states:
                X.append(last_10_states)

                # Label the next state as "good" or "bad"
                if next_state in good_states:
                    y.append("good")
                elif next_state in bad_states:
                    y.append("bad")

    # Convert X to numpy array and ensure y has matching length
    X = np.array(X)

    if len(X) != len(y):
        raise ValueError("Mismatch between features (X) and labels (y) lengths.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model (binary classification)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return knn,accuracy

def generate_knn_predictions(states,knn):

    def daily_knn_predictions(states,knn,i=0):

        current_date = states['Date'].iloc[-1-i]

        if i == 0:
            last_10_states = states['Stage'].iloc[-10:].to_numpy()
        else:
            last_10_states = states['Stage'].iloc[-10-i:0-i].to_numpy()

        if len(last_10_states) == 10 and not np.isnan(last_10_states).any():
            predicted_state = knn.predict([last_10_states])[0]
        else:
            predicted_state = "na"

        return current_date,predicted_state

    my_predictions = []
    for i in range(len(states)-10,-1,-1):
        current_date,predicted_state_d = daily_knn_predictions(states,knn,i)
        my_predictions.append([current_date,predicted_state_d])

    predictions_df = pd.DataFrame(my_predictions, columns=['Date', 'P_State'])

    merged_result = pd.merge(states, predictions_df, on='Date', how='left')

    return merged_result

def generate_daily_merged_df(mycode,start_daily_date,end_daily_date,choice="y"):
    if choice == "y":
        mystock_d = retrieve_data(mycode,end_daily_date,start_daily_date)
    elif choice == "a":
        mystock_d = retrieve_ak_daily_data(mycode,end_daily_date,start_daily_date)
    elif choice == "b":
        mystock_d = retrieve_bao_stock_daily(mycode,end_daily_date,start_daily_date)

    print("retrieve data successful")

    # print("Date=: ",mystock_d['Date'].iloc[-1],"Close=: ",mystock_d['Close'].iloc[-1])
    # close = mystock_d['Close'].iloc[-1]
    fvg_d = new_find_fvgs(mystock_d) # generate Benchmark and Bench with dis-continous data
    final_d = new_generate_finals(mystock_d,fvg_d,end_daily_date,start_daily_date) # combine mystock with fvg
    # benchmark = final_d['Benchmark'].iloc[-1]
    merged_df = generate_pure_stages(final_d) # [Date,Close,Benchmark,Bench,Stage]

    return merged_df

def generate_minute_merged_df(mycode,start_minute_date,end_minute_date,choice="y"):
    if choice == "y":
        mystock_m = retrieve_data(mycode,end_minute_date,start_minute_date,"15m")
    elif choice == "a":
        mystock_m = retrieve_ak_minute_data(mycode,end_minute_date,start_minute_date,period="15m")
    elif choice == "b":
        mystock_m = retrieve_bao_stock_data(mycode,end_minute_date,start_minute_date,freq="15")

    # print("Date=: ",mystock_m['Date'].iloc[-1],"Close=: ",mystock_m['Close'].iloc[-1])
    fvg_m = new_find_fvgs(mystock_m) # generate Benchmark and Bench with dis-continous data
    final_m = new_generate_finals(mystock_m,fvg_m,end_minute_date,start_minute_date) # combine mystock with fvg
    # bench = final_m['Benchmark'].iloc[-1]
    merged_df = generate_pure_stages(final_m) # [Date,Close,Benchmark,Bench,Stage]

    return merged_df,final_m

def batch_knn_individual(mycode,start_daily_date,end_daily_date,start_minute_date,end_minute_date,choice):

    def stage_grade(bench, benchmark, close):
        if bench > benchmark:
            if close < benchmark:
                grade_value = -2
            elif close < bench:
                grade_value = -1
            else:
                grade_value = 0
        else:
            if close < bench:
                grade_value = 0
            elif close < benchmark:
                grade_value = 1
            else:
                grade_value = 2

        return grade_value

    def predict_grade(minute_predict,daily_predict,bench,benchmark):
        if bench < benchmark:
            if minute_predict == "good" and daily_predict == "good":
                grade_value = 2
            elif minute_predict == "good" and daily_predict == "bad":
                grade_value = 1
            elif minute_predict == "bad" and daily_predict == "good":
                grade_value = 10
            elif minute_predict == "bad" and daily_predict == "bad":
                grade_value = 0
            else:
                pass
        else:
            if minute_predict == "good" and daily_predict == "good":
                grade_value = 0
            elif minute_predict == "good" and daily_predict == "bad":
                grade_value = -10
            elif minute_predict == "bad" and daily_predict == "good":
                grade_value = -1
            elif minute_predict == "bad" and daily_predict == "bad":
                grade_value = -2
            else:
                pass

        return grade_value

    print("code starts here")

    merged_df = generate_daily_merged_df(mycode,start_daily_date,end_daily_date,choice=choice) # [Date,Close,Benchmark,Bench,Stage]
    print("This is the second step")
    knn,accuracy = daily_knn_model(merged_df,start_daily_date,end_daily_date)
    print("This is the third step")
    stage_predict_d = generate_knn_predictions(merged_df,knn)
    # predicted_state_d = stage_predict_d['P_State'].iloc[-1]

    merged_df,_ = generate_minute_merged_df(mycode,start_minute_date,end_minute_date,choice=choice) # [Date,Close,Benchmark,Bench,Stage]
    knn,accuracy = daily_knn_model(merged_df,start_minute_date,end_minute_date)
    stage_predict_m = generate_knn_predictions(merged_df,knn)
    # predicted_state_m = stage_predict_m['P_State'].iloc[-1]

    # Prepare for minute_df
    minute_df = stage_predict_m[['Date','Close','Benchmark','Stage','P_State']].copy()
    minute_df = minute_df.rename(columns={'Benchmark': 'Bench', 'Stage': 'Stage_m', 'P_State': 'P_State_m'})
    minute_df['Date'] = pd.to_datetime(minute_df['Date'])
    minute_df['Date_m'] = minute_df['Date'].dt.floor('D').reset_index(drop=True)

    # Prepare for daily_df
    daily_df = stage_predict_d[['Date','Close','Benchmark','Stage','P_State']].copy()
    daily_df = daily_df.rename(columns={'Stage': 'Stage_d', 'P_State': 'P_State_d'}).reset_index(drop=True)

    print("Minute DataFrame:\n", minute_df.head())
    print("Daily DataFrame:\n", daily_df.head())

    # Merge daily_df with minute_df based on 'Date' and 'Date_m'
    combined_df = pd.merge(minute_df, daily_df, left_on='Date_m', right_on='Date', how='left')
    combined_df = combined_df.drop(columns=['Date_m', 'Date_y', 'Close_y'])
    combined_df = combined_df.rename(columns={'Date_x': 'Date', 'Close_x': 'Close'}).dropna(how='any')

    # Apply the stage_grade and predict_grade function to each row
    combined_df['Current_Grade'] = combined_df.apply(
        lambda row: stage_grade(row['Bench'], row['Benchmark'], row['Close']), axis=1)
    combined_df['Predict_Grade'] = combined_df.apply(
        lambda row: predict_grade(row['P_State_m'], row['P_State_d'], row['Bench'], row['Benchmark']), axis=1)
    combined_df['Potential'] = combined_df['Current_Grade'] - combined_df['Predict_Grade']

    return combined_df

def batch_stage_individual(mycode,start_daily_date,end_daily_date,start_minute_date,end_minute_date,choice):

    merged_df_d = generate_daily_merged_df(mycode,start_daily_date,end_daily_date,choice=choice) # [Date,Close,Benchmark,Bench,Stage]

    merged_df_m,final_m = generate_minute_merged_df(mycode,start_minute_date,end_minute_date,choice=choice) # [Date,Close,Benchmark,Bench,Stage]

    # Prepare for minute_df
    minute_df = merged_df_m[['Date','Close','Benchmark','Stage']].copy()
    minute_df = minute_df.rename(columns={'Benchmark': 'Bench', 'Stage': 'Stage_m'})
    minute_df['Date'] = pd.to_datetime(minute_df['Date'])
    minute_df['Date_m'] = minute_df['Date'].dt.floor('D')

    # Prepare for daily_df
    daily_df = merged_df_d[['Date','Close','Benchmark','Stage']].copy()
    daily_df = daily_df.rename(columns={'Stage': 'Stage_d'})

    # Merge daily_df with minute_df based on 'Date' and 'Date_m'
    combined_df = pd.merge(minute_df, daily_df, left_on='Date_m', right_on='Date', how='left')
    combined_df = combined_df.drop(columns=['Date_m', 'Date_y', 'Close_y'])
    combined_df = combined_df.rename(columns={'Date_x': 'Date', 'Close_x': 'Close'}).dropna(how='any') # [Date,Close,Bench,Stage_m,Benchmark,Stage_d]

    return combined_df,final_m

def generate_knn_signals(combined_stage_df):
      # Generate Signals for merged_df
      merged_df = combined_stage_df.copy()
      combined_signal = []
      for i in range(1,len(merged_df)):
          the_date = merged_df['Date'].iloc[i]
          the_close = merged_df['Close'].iloc[i]
          close_prev = merged_df['Close'].iloc[i-1]
          the_bench = merged_df['Bench'].iloc[i]
          bench_prev =  merged_df['Bench'].iloc[i-1]
          the_benchmark = merged_df['Benchmark'].iloc[i]
          benchmark_prev = merged_df['Benchmark'].iloc[i-1]
          the_stage = merged_df['Stage'].iloc[i]
          stage_prev = merged_df['Stage'].iloc[i-1]

          if ((the_close >= the_benchmark) and (close_prev <= benchmark_prev)) \
               or ((the_close >= the_bench) and (close_prev <= bench_prev)) \
                or (the_stage == 5 and stage_prev == 4):
              combined_signal.append([the_date,"Buy",the_close])
          elif ((the_close <= the_benchmark) and (close_prev >= benchmark_prev)) \
             or ((the_close <= the_bench) and (close_prev >= bench_prev)) \
                or (the_stage == 2 and stage_prev == 3):
              combined_signal.append([the_date,"Sell",the_close])

          else: continue

      original_signals_df = pd.DataFrame(combined_signal, columns=["Date","Signal","Signal_Value"])

      # Step 1: Convert 'Date' column to datetime
      original_signals_df['Date'] = pd.to_datetime(original_signals_df['Date'])

      # Step 2: Group by the date part (YYYY-MM-DD) and keep the last row for each group
      signals_by_day_df = original_signals_df.groupby(original_signals_df['Date'].dt.date).tail(1).reset_index(drop=True)
      compressed_df = signals_by_day_df[signals_by_day_df['Signal'] != signals_by_day_df['Signal'].shift(1)]

      return signals_by_day_df, compressed_df

def calculate_profit_and_change(df,signal,startDate,endDate):
    # Calculate the profitability and close price percentage change from two DataFrames.

    # Parameters:
    # df (DataFrame): DataFrame containing stock data with columns ['Date', 'Close', 'Open', 'High', 'Low', 'Volume'].
    # signal (DataFrame): DataFrame containing signals with columns ['Date', 'Signal', 'Signal_Value'].

    # Returns:
    # dict: A dictionary with keys 'profitability' and 'close_price_change' containing the respective calculations.

    # Ensure 'Date' columns are timezone-naive
    df['Date'] = df['Date'].dt.tz_localize(None)
    signal['Date'] = signal['Date'].dt.tz_localize(None)
    startDate = pd.Timestamp(startDate).normalize()
    endDate = pd.Timestamp(endDate).normalize()

    # # Step 1: Remove consecutive duplicate signals
    # signal = signal[signal['Signal'] != signal['Signal'].shift(1)]

    # Step 2: Merge the two DataFrames on the 'Date' column
    before_merged_df = pd.merge(df, signal[['Date', 'Signal', 'Signal_Value']], on='Date', how='left')
    # before_merged_df['Date'] = pd.to_datetime(df['Date'])

    # Filter the DataFrame for dates greater than "2024-01-01"
    pre_merged_df = before_merged_df[before_merged_df['Date'] >= startDate]
    merged_df = pre_merged_df[pre_merged_df['Date'] <= endDate]
    # print(merged_df)

    # Step 3: Calculate the percentage change in Close price from earliest to most recent date
    earliest_close = merged_df['Close'].iloc[0]
    most_recent_close = merged_df['Close'].iloc[-1]
    # earliest_date = merged_df['Date'].iloc[0]
    # most_recent_date = merged_df['Date'].iloc[-1]
    print(f"from {startDate.date()} to {endDate.date()}")

    # Step 4: Calculate profitability and track shares based on the signals
    initial_capital = 1000000
    capital = initial_capital
    holding = False
    total_profit = 0
    shares = 0  # To track the number of shares bought
    trading_times = 0
    for index, row in merged_df.iterrows():
        if row['Close'] == 0: continue
        if row['Signal'] == 'Buy' and not holding:
            # if first_buy:
            #     #earliest_close = row['Close']   # Close of the first trading(Buy) day is the earliest_close
            #     first_buy = False
            trading_times += 1
            shares = capital // row['Close']  # Calculate how many shares can be bought
            capital = shares * row['Close']  # Update capital to reflect the amount used
            holding = True
        elif row['Signal'] == 'Sell' and holding:
            # if shares == 0: print(row['Date'])
            total_profit += shares * row['Close'] - capital  # Calculate profit for this trade
            capital = initial_capital + total_profit  # Update capital after selling
            holding = False
            shares = 0  # Reset shares after selling
            trading_times += 1
    # If the last signal is 'Buy', assume selling at the last available Close price
    if holding:
        total_profit += shares * most_recent_close - capital

    # Calculate profitability as total profit divided by the initial capital
    profitability = (total_profit / initial_capital) * 100  # Express as percentage
    close_price_change = ((most_recent_close - earliest_close) / earliest_close) * 100
    # Return results as a dictionary
    print(f"Trading Times: {trading_times}")
    return {
        'profitability': round(profitability, 2),
        'close_price_change': round(close_price_change, 2),
        'shares': shares  # Return the last number of shares held, if any
    }
import numpy as np
# from scipy.stats import norm

# Black-Scholes function for a call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

# Function to calculate the call option price
def calculate_call_option_price(merged_df, past_days, risk_free_rate, maturity_days):

    # The current stock price (last closing price)
    S = merged_df['Close'].iloc[-1]

    # The exercise price is the same as the last close price
    K = S

    # Time to maturity (in years, given the input as days)
    T = maturity_days / 252  # Convert days to years

    # Annualized risk-free rate
    r = risk_free_rate / 100  # Keep r as an annual rate in decimal form

    # Calculate volatility using the last 20 days of data
    last_200_prices = merged_df['Close'].tail(past_days).values
    daily_returns = np.diff(last_200_prices) / last_200_prices[:-1]
    sigma = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility

    # Calculate the call option price using Black-Scholes formulamyenv
    call_price = black_scholes_call(S, K, T, r, sigma)

    return call_price

# # @title
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import date, timedelta

# def new_plot_stage_predict_grade(combined_df):

#     stage_of_grade = combined_df['Current_Grade'].iloc[-1]
#     predict_of_grade = combined_df['Predict_Grade'].iloc[-1]
#     close = combined_df['Close'].iloc[-1]
#     bench = combined_df['Bench'].iloc[-1]
#     benchmark = combined_df['Benchmark'].iloc[-1]
#     daily_state = combined_df['P_State_d'].iloc[-1]
#     minute_state = combined_df['P_State_m'].iloc[-1]

#     # Variables a and b (strings)
#     a = "minute: " + minute_state
#     b = "daily: " + daily_state
#     c = str(stage_of_grade - predict_of_grade)

#     # Calculate padding dynamically based on the range of values
#     min_value = min(bench, benchmark, close)
#     max_value = max(bench, benchmark, close) * 1.02
#     padding = (max_value - min_value) * 0.5  # Add 50% of the range as padding

#     # If padding is too small (when numbers are very close), set a minimum padding
#     if padding == 0:
#         padding = 0.1  # Minimum padding to ensure clear spacing

#     # Create the figure and axis
#     fig, ax = plt.subplots()

#     # Draw benchmark (solid line) and bench (dashed line) with small horizontal lengths
#     plt.hlines(benchmark, 0.95, 1.05, colors='black', linestyles='solid', linewidth=2)
#     plt.hlines(bench, 0.95, 1.05, colors='blue', linestyles='dashed', linewidth=2)

#     # Mark the 'close' value as a red dot
#     plt.scatter(1, close, color='red', zorder=5)

#     # Add labels for each value
#     plt.text(1.06, benchmark, f'Benchmark: {benchmark}', va='center')
#     plt.text(1.06, bench, f'Bench: {bench}', va='center', color='blue')
#     plt.text(1.06, close, f'Close: {close}', va='center', color='red')

#     # Add variables a and b at the top of the chart
#     plt.text(1.05, max_value + 0.75*padding, f'{a}, {b}, {c}', ha='center', fontsize=12, fontweight='normal')

#     # Dynamically adjust y-axis limits based on the values and padding
#     plt.xlim(0.9, 1.2)  # Tight x-axis limits
#     plt.ylim(min_value - padding, max_value + padding)  # Dynamic y-limits

#     # Remove x-axis and ticks
#     ax.get_xaxis().set_visible(False)

#     # # Show the plot
#     # plt.show()
#     return fig

from flask import Flask, request, send_file, jsonify
import io
from io import BytesIO
from datetime import date, timedelta
# import plotly.io as pio
import yfinance as yf

app = Flask(__name__)

@app.route('/get-state', methods=['POST'])
def get_state():
    try:
        data = request.get_json()
        if 'Code' not in data:
            return jsonify({"error": "Code parameter missing"}), 400
        
        code = data['Code']
        print(f"Received Code: {code}")  # This should print to the terminal

        mycode = code

        today = date.today()
        tomorrow = today + timedelta(days=1)
        oneandhalf_years_ago = tomorrow - timedelta(days=364*1.5)
        sixty_days_ago = tomorrow - timedelta(days=58)

        start_daily_date = oneandhalf_years_ago.strftime("%Y-%m-%d")
        end_daily_date = tomorrow.strftime("%Y-%m-%d")

        start_minute_date = sixty_days_ago.strftime("%Y-%m-%d")
        end_minute_date = tomorrow.strftime("%Y-%m-%d")

        print(today)
        
        # Create combined_df
        combined_df = batch_knn_individual(mycode, start_daily_date, end_daily_date, start_minute_date, end_minute_date, "y")

        # Debugging: Inspect the DataFrame
        print("Combined DataFrame Head:\n", combined_df.head())
        print("Combined DataFrame Index Levels:\n", combined_df.index)

        stage_of_grade = combined_df['Current_Grade'].iloc[-1]
        predict_of_grade = combined_df['Predict_Grade'].iloc[-1]
        close = combined_df['Close'].iloc[-1]
        bench = combined_df['Bench'].iloc[-1]
        benchmark = combined_df['Benchmark'].iloc[-1]
        daily_state = combined_df['P_State_d'].iloc[-1]
        minute_state = combined_df['P_State_m'].iloc[-1]

        # Variables a and b (strings)
        a = "minute: " + minute_state
        b = "daily: " + daily_state
        c = str(stage_of_grade - predict_of_grade)

        title = f"{a}, {b}, {c}"

        # Convert the last row to a dictionary
        last_row_dict = dict()
        last_row_dict['Bench'] = bench
        last_row_dict['Benchmark'] = benchmark
        last_row_dict['Close'] = close
        last_row_dict['Title'] = title
        
        # Return the response as a JSON object
        return jsonify(last_row_dict)

    except Exception as e:
        print("Error occurred:", e)  # Print error message to console
        return jsonify({"error": str(e)}), 500  # Return error message as JSON


if __name__ == '__main__':
    app.run(debug=True)
