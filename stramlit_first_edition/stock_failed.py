'''
project/
├── components/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   ├── visualization.py
│   └── utils.py
├── requirements.txt
└── main.py
'''
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time
import re


def format_number(number):
    """格式化數字"""
    return "{:,.2f}".format(number)
def validate_data(df, min_length=60):
    """驗證數據是否足夠，若不足則補全"""
    if df is None or df.empty:
        st.error("數據為空")
        return False, None
    
    required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"數據缺少必要的欄位: {', '.join(missing_columns)}")
        return False, None
    
    if len(df) < min_length:
        st.warning(f"數據量不足，只有 {len(df)} 筆，將嘗試補充至 {min_length} 筆")
        df = df.copy()
        while len(df) < min_length:
            df = pd.concat([df, df.iloc[-1:]], ignore_index=True)

    null_columns = df[required_columns].isnull().any()
    if null_columns.any():
        null_cols = null_columns[null_columns].index.tolist()
        st.warning(f"以下欄位包含空值，將進行自動處理: {', '.join(null_cols)}")
        for col in null_cols:
            df[col].fillna(method='ffill', inplace=True)
            df[col].fillna(method='bfill', inplace=True)

    inf_columns = np.isinf(df[required_columns]).any()
    if inf_columns.any():
        inf_cols = inf_columns[inf_columns].index.tolist()
        st.warning(f"以下欄位包含無限值，將進行自動處理: {', '.join(inf_cols)}")
        for col in inf_cols:
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            df[col].fillna(method='ffill', inplace=True)
            df[col].fillna(method='bfill', inplace=True)

    return True, df

def prepare_lstm_data(data, lookback=60, scaler=None):
    """準備 LSTM 模型的輸入數據"""
    try:
        if len(data) == 0:
            st.error("輸入數據為空")
            return None, None, None

        data = np.array(data, dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=None, neginf=None)

        mean = np.mean(data[np.isfinite(data)])
        std = np.std(data[np.isfinite(data)])
        data = np.clip(data, mean - 10 * std, mean + 10 * std)

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        else:
            scaled_data = scaler.transform(data.reshape(-1, 1))

        if len(scaled_data) < lookback + 1:
            st.error(f"數據長度不足，需要至少 {lookback + 1} 個數據點")
            return None, None, None

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0 or len(y) == 0:
            st.error("無法生成有效的訓練數據")
            return None, None, None

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaler

    except Exception as e:
        st.error(f"數據預處理過程中發生錯誤: {str(e)}")
        return None, None, None
    
def create_lstm_model(lookback):
    """創建LSTM模型"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_predict_lstm(df, epochs=50, batch_size=32, lookback=60):
    """改進的LSTM模型訓練和預測"""
    if df is None or df.empty:
        return None, None
    
    try:
        # 數據預處理
        data = df['Close'].values
        
        # 標準化數據
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        # 準備序列數據
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # 分割訓練和測試數據
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 創建改進的LSTM模型
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        # 使用Adam優化器並添加學習率衰減
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        # 添加早停機制
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # 學習率衰減
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001
        )
        
        # 訓練模型
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping, lr_scheduler],
            verbose=0
        )
        
        # 顯示訓練過程
        st.line_chart(pd.DataFrame(history.history['loss'], columns=['training loss']))
        
        # 模型摘要
        st.write("模型訓練摘要：")
        st.write(f"* 訓練輪數：{len(history.history['loss'])}")
        st.write(f"* 最終損失：{history.history['loss'][-1]:.4f}")
        st.write(f"* 最終驗證損失：{history.history['val_loss'][-1]:.4f}")
        
        # 預測
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # 反轉縮放
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_orig = scaler.inverse_transform([y_train]).T
        y_test_orig = scaler.inverse_transform([y_test]).T
        
        # 創建完整的預測序列
        predictions = np.zeros(len(df))
        predictions[:] = np.nan
        predictions[lookback:lookback+len(train_predict)] = train_predict.flatten()
        predictions[lookback+len(train_predict):] = test_predict.flatten()
        
        # 轉換為pandas Series
        predictions = pd.Series(predictions, index=df.index)
        
        return predictions, (y_test_orig.flatten(), test_predict.flatten())
        
    except Exception as e:
        st.error(f"LSTM模型訓練過程中發生錯誤: {str(e)}")
        return None, None

def prepare_features(df):
    """改進的特徵準備函數"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
        
    try:
        df_features = pd.DataFrame()
        
        # 基本特徵
        df_features['Close'] = df['Close']
        df_features['Volume'] = df['Volume']
        
        # 處理極端值和缺失值
        for col in ['Close', 'Volume']:
            # 處理無限值
            df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan)
            
            # 計算均值和標準差 (忽略NaN)
            mean = df_features[col].mean()
            std = df_features[col].std()
            
            # 極端值處理
            df_features[col] = df_features[col].clip(lower=mean - 3*std, upper=mean + 3*std)
            
            # 填充缺失值
            df_features[col] = df_features[col].fillna(mean)
        
        # 定義特徵列表
        features = ['MA5', 'MA10', 'MA20', 'MA60', 
                   'Price_Change', 'Price_Change_5', 'Price_Change_10',
                   'Volatility', 'Volume_MA5', 'Volume_MA20', 'Volume_Change',
                   'Price_Position', 'RSI']
        
        # 計算移動平均
        windows = [5, 10, 20, 60]
        for window in windows:
            df_features[f'MA{window}'] = (
                df['Close']
                .rolling(window=window, min_periods=1)
                .mean()
                .fillna(method='bfill')
            )
        
        # 價格變化率
        for period in [1, 5, 10]:
            col_name = 'Price_Change' if period == 1 else f'Price_Change_{period}'
            df_features[col_name] = (
                df['Close']
                .pct_change(periods=period)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
        
        # 波動率
        df_features['Volatility'] = (
            df['Close']
            .rolling(window=20, min_periods=1)
            .std()
            .fillna(method='bfill')
        )
        
        # 成交量特徵
        df_features['Volume_MA5'] = (
            df['Volume']
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(method='bfill')
        )
        df_features['Volume_MA20'] = (
            df['Volume']
            .rolling(window=20, min_periods=1)
            .mean()
            .fillna(method='bfill')
        )
        df_features['Volume_Change'] = (
            df['Volume']
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        
        # 價格位置
        min_price = df['Close'].rolling(window=20, min_periods=1).min()
        max_price = df['Close'].rolling(window=20, min_periods=1).max()
        price_range = max_price - min_price
        df_features['Price_Position'] = np.where(
            price_range != 0,
            (df['Close'] - min_price) / price_range,
            0.5  # 當範圍為0時使用0.5作為中間值
        )
        
        # RSI指標
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = np.where(loss != 0, gain/loss, 0)
        df_features['RSI'] = 100 - (100 / (1 + rs))
        df_features['RSI'] = df_features['RSI'].fillna(50)  # 使用中間值填充
        
        # 最後檢查確保沒有NaN值
        if df_features.isnull().any().any():
            df_features = df_features.fillna(df_features.mean())
        
        # 確保所有特徵都存在
        missing_features = set(features) - set(df_features.columns)
        if missing_features:
            st.warning(f"缺少以下特徵: {', '.join(missing_features)}")
            return None, None
            
        return df_features, features
        
    except Exception as e:
        st.error(f"特徵準備過程中發生錯誤: {str(e)}")
        return None, None

def train_predict(df_original, train_test_ratio=0.8):
    """改進的訓練預測函數"""
    if df_original is None or not isinstance(df_original, pd.DataFrame) or df_original.empty:
        st.error("輸入數據無效")
        return None, None
        
    try:
        # 使用原始數據準備特徵
        features_result = prepare_features(df_original)
        if features_result is None or len(features_result) != 2:
            st.error("特徵準備失敗")
            return None, None
            
        df_features, features = features_result
        
        # 再次檢查NaN值
        if df_features.isnull().any().any():
            st.warning("數據中存在空值，將進行填充")
            df_features = df_features.fillna(df_features.mean())
        
        X = df_features[features]
        y = df_features['Close']
        
        # 分割數據
        test_size = 1 - train_test_ratio
        split_index = int(len(df_features) * train_test_ratio)
        
        # 使用時間序列分割
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        # 訓練模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 計算特徵重要性
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': abs(model.coef_)
        }).sort_values('Importance', ascending=False)
        
        st.subheader('特徵重要性')
        st.write(feature_importance)
        
        # 模型評估
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        st.write(f'訓練集 R² 分數: {train_score:.4f}')
        st.write(f'測試集 R² 分數: {test_score:.4f}')
        
        # 預測
        predictions = model.predict(X)
        test_predictions = model.predict(X_test)
        
        # 將預測結果轉換為pandas Series
        predictions = pd.Series(predictions, index=df_original.index)
        
        return predictions, (y_test, test_predictions)
        
    except Exception as e:
        st.error(f"模型訓練過程中發生錯誤: {str(e)}")
        return None, None
    
def get_market_suffix(market):
    """改進的市場後綴獲取"""
    market_suffixes = {
        '台股': '.TW',
        '美股': ''
    }
    return market_suffixes.get(market.strip(), '')

def validate_stock_symbol(symbol, market):
    """改進的股票代碼驗證"""
    if market == '台股':
        # 移除任何空格
        symbol = symbol.strip()
        # 允許前導零
        if not symbol.isdigit():
            return False
        # 確保實際數字是4-6位
        num_val = int(symbol)
        return 1000 <= num_val <= 999999
    elif market == '美股':
        # 移除空格並轉換為大寫
        symbol = symbol.strip().upper()
        # 美股代碼規則：1-5位字母
        return bool(re.match(r'^[A-Z]{1,5}$', symbol))
    return False

def get_stock_data(symbol, market, start_date, end_date):
    """改進的股票數據獲取函數"""
    try:
        # 台股代碼處理
        if market == '台股':
            # 確保是4位數字的代碼
            symbol = symbol.zfill(4)
            full_symbol = f"{symbol}.TW"
        else:
            full_symbol = symbol  # 美股不需要後綴
            
        # 重試機制
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 添加延遲避免請求過快
                if retry_count > 0:
                    time.sleep(2)
                
                # 獲取數據
                df = yf.download(
                    full_symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                # 驗證數據
                if df is not None and not df.empty:
                    # 檢查是否有交易數據
                    if len(df) > 0:
                        st.success(f"成功獲取 {market} {symbol} 的股票數據")
                        return df
                    else:
                        st.warning("指定時間範圍內沒有交易數據")
                        return None
                
                retry_count += 1
                st.warning(f"第 {retry_count} 次嘗試獲取數據...")
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    st.warning(f"第 {retry_count} 次嘗試獲取數據失敗，正在重試...")
                else:
                    st.error(f"無法獲取股票數據: {str(e)}")
                    return None
        
        st.error(f"在 {max_retries} 次嘗試後仍無法獲取股票數據")
        return None
        
    except Exception as e:
        st.error(f"獲取股票數據時發生錯誤: {str(e)}")
        return None

def get_stock_info(symbol, market):
    """獲取股票基本信息"""
    try:
        full_symbol = symbol + get_market_suffix(market)
        stock = yf.Ticker(full_symbol)
        info = stock.info
        return info
    except:
        return None

def display_prediction_plot(df, predictions, market=None, stock_symbol=None):
    """展示預測股價的圖表"""
    try:
        # 檢查資料有效性
        if df is None or predictions is None or df.empty or predictions.empty:
            st.error("無法繪製圖表，數據或預測結果無效")
            return

        # 填補預測數據的空值
        predictions = predictions.fillna(method='ffill').fillna(method='bfill')

        # 確保索引一致
        predictions = predictions.reindex(df.index, method='nearest')

        # 初始化圖表
        fig = go.Figure()

        # 添加實際價格曲線
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='實際價格',
            line=dict(color='blue', width=2)
        ))

        # 添加預測價格曲線
        fig.add_trace(go.Scatter(
            x=df.index,
            y=predictions,
            mode='lines',
            name='預測價格',
            line=dict(color='red', dash='dash', width=2)
        ))

        # 設定圖表格式
        title_text = f'{market} {stock_symbol} 股價預測結果' if market and stock_symbol else '股價預測結果'
        fig.update_layout(
            title=title_text,
            xaxis=dict(title='日期', showgrid=True, gridwidth=0.5, gridcolor='LightGray'),
            yaxis=dict(title='價格', showgrid=True, gridwidth=0.5, gridcolor='LightGray'),
            legend=dict(title='圖例', bgcolor='rgba(255,255,255,0.5)'),
            hovermode='x unified',
            template='plotly_white'
        )

        # 顯示圖表
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"繪製圖表時發生錯誤: {str(e)}")



def display_prediction_metrics(last_actual, last_predicted, prediction_diff):
    """顯示預測結果指標"""
    st.subheader('預測結果')
    col1, col2, col3 = st.columns(3)
    
    # 格式化顯示
    col1.metric(
        "最新實際價格",
        f"${format_number(last_actual)}",
        delta=None
    )
    
    col2.metric(
        "預測價格",
        f"${format_number(last_predicted)}",
        delta=None
    )
    
    # 根據預測差異顯示不同顏色
    delta_color = "normal"
    if abs(prediction_diff) > 5:
        delta_color = "off"
    
    col3.metric(
        "預測差異",
        f"{format_number(prediction_diff)}%",
        delta=None,
        delta_color=delta_color
    )

def prepare_balanced_data(df):
    """準備平衡的訓練數據"""
    if df is None or len(df) == 0:
        st.error("輸入數據為空")
        return None
        
    try:
        # 只保留需要的列
        df = df[['Close', 'Open', 'High', 'Low', 'Volume']].copy()
        
        # 計算價格變動
        df['returns'] = df['Close'].pct_change()
        
        # 移除NaN值
        df = df.dropna()
        
        # 將數據分類為上漲、下跌和橫盤
        threshold = 0.001
        df['movement'] = np.where(df['returns'] > threshold, 'up',
                                np.where(df['returns'] < -threshold, 'down', 'neutral'))
        
        if len(df) == 0:
            st.error("處理後數據為空")
            return None
        
        # 顯示原始數據分布
        distribution = df['movement'].value_counts()
        st.write("原始數據分布：")
        st.write(distribution)
        
        # 對每類樣本進行平衡採樣
        min_samples = min(distribution)
        balanced_dfs = []
        
        for movement in ['down', 'neutral', 'up']:
            movement_data = df[df['movement'] == movement]
            if len(movement_data) > min_samples:
                balanced_dfs.append(movement_data.sample(n=min_samples, random_state=42))
            else:
                balanced_dfs.append(movement_data)
        
        balanced_df = pd.concat(balanced_dfs)
        
        # 顯示平衡後的分布
        st.write("平衡後的數據分布：")
        st.write(balanced_df['movement'].value_counts())
        
        return balanced_df
        
    except Exception as e:
        st.error(f"數據平衡處理時發生錯誤: {str(e)}")
        return None

def calculate_backtest_metrics(y_true, y_pred):
    """計算回測指標"""
    try:
        # 確保輸入是numpy數組
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 去除無效值
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("沒有有效的數據進行計算")
            
        # 計算基本誤差指標
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # 計算方向準確度
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        direction_accuracy = np.mean(true_direction == pred_direction) * 100
        
        # 計算價格準確度
        mape = mean_absolute_percentage_error(y_true, y_pred)
        price_accuracy = max(0, min(100, (1 - mape) * 100))
        
        # 計算R平方值
        r2 = r2_score(y_true, y_pred)
        r2_percentage = max(-100, min(100, r2 * 100))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'price_accuracy': price_accuracy,
            'r2_percentage': r2_percentage
        }
        
    except Exception as e:
        st.error(f"計算回測指標時發生錯誤: {str(e)}")
        return {
            'mse': 0,
            'rmse': 0,
            'direction_accuracy': 0,
            'price_accuracy': 0,
            'r2_percentage': 0
        }
        
def perform_time_series_cv(df, model_type='linear', n_splits=5):
    """執行時間序列交叉驗證"""
    if df is None or len(df) == 0:
        st.error("交叉驗證的輸入數據為空")
        return None
        
    try:
        # 確保數據量足夠進行交叉驗證
        if len(df) < n_splits * 2:
            st.warning(f"數據量不足以進行{n_splits}折交叉驗證，將調整為較小的折數")
            n_splits = max(2, len(df) // 2)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {
            'train_scores': [],
            'test_scores': [],
            'direction_accuracy': [],
            'price_accuracy': []
        }
        
        for train_idx, test_idx in tscv.split(df):
            try:
                train_data = df.iloc[train_idx]
                test_data = df.iloc[test_idx]
                
                if model_type == 'linear':
                    # 線性回歸
                    features_result_train = prepare_features(train_data)
                    features_result_test = prepare_features(test_data)
                    
                    if features_result_train is None or features_result_test is None:
                        continue
                        
                    df_features_train, features = features_result_train
                    df_features_test, _ = features_result_test
                    
                    X_train = df_features_train[features]
                    y_train = df_features_train['Close']
                    X_test = df_features_test[features]
                    y_test = df_features_test['Close']
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                else:
                    # LSTM
                    X_train, y_train, scaler = prepare_lstm_data(train_data['Close'].values)
                    if X_train is not None:
                        X_test, y_test, _ = prepare_lstm_data(
                            test_data['Close'].values, 
                            scaler=scaler
                        )
                        if X_test is not None:
                            model = create_lstm_model(X_train.shape[1])
                            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                            train_pred = model.predict(X_train)
                            test_pred = model.predict(X_test)
                            
                            # 反轉縮放
                            train_pred = scaler.inverse_transform(train_pred).flatten()
                            test_pred = scaler.inverse_transform(test_pred).flatten()
                            y_train = scaler.inverse_transform([y_train]).flatten()
                            y_test = scaler.inverse_transform([y_test]).flatten()
                
                # 計算性能指標
                metrics = calculate_backtest_metrics(y_test, test_pred)
                if metrics:
                    cv_scores['train_scores'].append(metrics['r2_percentage'])
                    cv_scores['test_scores'].append(metrics['r2_percentage'])
                    cv_scores['direction_accuracy'].append(metrics['direction_accuracy'])
                    cv_scores['price_accuracy'].append(metrics['price_accuracy'])
                    
            except Exception as e:
                st.warning(f"跳過一個分割: {str(e)}")
                continue
        
        # 檢查是否有有效的分數
        if not any(cv_scores.values()):
            st.error("交叉驗證未能產生有效的分數")
            return None
            
        return cv_scores
        
    except Exception as e:
        st.error(f"交叉驗證過程中發生錯誤: {str(e)}")
        return None

def calculate_direction_accuracy(y_true, y_pred):
    """計算方向準確度"""
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction) * 100

def display_cv_results(cv_scores):
    """顯示交叉驗證結果"""
    st.subheader('交叉驗證結果')
    
    col1, col2 = st.columns(2)
    
    # 訓練和測試性能
    col1.markdown("**模型性能 (R²)**")
    col1.write(f"訓練集平均: {np.mean(cv_scores['train_scores']):.4f} ± {np.std(cv_scores['train_scores']):.4f}")
    col1.write(f"測試集平均: {np.mean(cv_scores['test_scores']):.4f} ± {np.std(cv_scores['test_scores']):.4f}")
    
    # 方向和價格準確度
    col2.markdown("**預測準確度**")
    col2.write(f"方向準確度: {np.mean(cv_scores['direction_accuracy']):.2f}% ± {np.std(cv_scores['direction_accuracy']):.2f}%")
    col2.write(f"價格準確度: {np.mean(cv_scores['price_accuracy']):.2f}% ± {np.std(cv_scores['price_accuracy']):.2f}%")
    
    # 繪製CV scores分布
    fig.add_trace(go.Box(y=cv_scores['train_scores'], name='訓練集 R²'))
    fig.add_trace(go.Box(y=cv_scores['test_scores'], name='測試集 R²'))
    fig.add_trace(go.Box(y=cv_scores['direction_accuracy'], name='方向準確度'))
    fig.add_trace(go.Box(y=cv_scores['price_accuracy'], name='價格準確度'))
    
    fig.update_layout(title='交叉驗證性能分布',
                     yaxis_title='分數',
                     boxmode='group')
    


def main():
    # 初始化变量
    df = None
    predictions = None
    backtest_data = None

    st.title('全球股票預測與回測系統')
    
    # 側邊欄
    st.sidebar.header('參數設置')
    
    # 市場選擇
    market = st.sidebar.selectbox(
        '選擇市場',
        ['台股', '美股']
    )
    
    # 添加模型選擇
    model_type = st.sidebar.selectbox(
        '選擇預測模型',
        ['線性回歸', 'LSTM']
    )
    
    # 添加模型參數設置
    if model_type == 'LSTM':
        st.sidebar.subheader('LSTM模型參數')
        epochs = st.sidebar.slider('訓練輪數', 10, 100, 50)
        batch_size = st.sidebar.slider('批次大小', 16, 64, 32)
        lookback = st.sidebar.slider('歷史數據長度', 30, 90, 60)
    else:
        st.sidebar.subheader('線性回歸模型參數')
        train_test_split_ratio = st.sidebar.slider('訓練集比例', 0.5, 0.9, 0.8, 0.1)
    
    # 股票代碼輸入提示
    if market == '台股':
        placeholder = '2330'
        help_text = '請輸入台股代碼（例如：2330)'
    else:
        placeholder = 'AAPL'
        help_text = '請輸入美股代碼（例如：AAPL)'
    
    # 股票代碼輸入
    stock_symbol = st.sidebar.text_input('請輸入股票代碼', placeholder, help=help_text)
    
    if stock_symbol:
        if not validate_stock_symbol(stock_symbol, market):
            st.error(f'請輸入有效的{market}代碼格式')
            return
    
    # 日期選擇
    today = datetime.now()
    default_start = today - timedelta(days=365)
    
    start_date = st.sidebar.date_input("開始日期", value=default_start)
    end_date = st.sidebar.date_input("結束日期", value=today)
    
    # 驗證日期
    if start_date >= end_date:
        st.error('開始日期必須早於結束日期')
        return
    
    if st.sidebar.button('開始分析'):
        with st.spinner('正在獲取股票數據...'):
            df = get_stock_data(stock_symbol, market, start_date, end_date)
        
    if df is not None and not df.empty:
        if validate_data(df):
            try:
                # 數據預處理
                st.subheader('數據分析與預處理')
                
                # 保存原始數據
                df_original = df.copy()
                
                # 數據平衡和交叉驗證
                with st.spinner('正在進行數據分析...'):
                    balanced_df = prepare_balanced_data(df)
                    
                    if balanced_df is not None:
                        predictions = None
                        backtest_data = None
                        cv_results = None
                        
                        # 執行交叉驗證
                        cv_results = perform_time_series_cv(
                            balanced_df,
                            model_type='lstm' if model_type == 'LSTM' else 'linear'
                        )
                        
                        with st.spinner('正在訓練最終模型...'):
                            if model_type == '線性回歸':
                                # 使用原始數據進行預測
                                predictions, backtest_data = train_predict(
                                    df_original,
                                    train_test_ratio=train_test_split_ratio
                                )
                            else:
                                predictions, backtest_data = train_predict_lstm(
                                    balanced_df,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    lookback=lookback
                                )
                
                if predictions is not None and backtest_data is not None:
                    try:
                        # 確保預測結果有效
                        predictions = predictions.fillna(method='ffill').fillna(method='bfill')
                        if predictions.empty or predictions.isna().all():
                            st.error("無法生成有效的預測結果")
                        else:
                            # 顯示預測趨勢圖
                            display_prediction_plot(df, predictions, market, stock_symbol)

                            # 顯示預測結果指標
                            last_actual = float(df['Close'].iloc[-1])
                            last_predicted = float(predictions.iloc[-1])
                            prediction_diff = ((last_predicted - last_actual) / last_actual) * 100

                            st.subheader('預測結果')
                            st.write(f"最新實際價格：${format_number(last_actual)}")
                            st.write(f"預測價格：${format_number(last_predicted)}")
                            st.write(f"預測差異：{format_number(prediction_diff)}%")
                    except Exception as e:
                        st.error(f"處理預測結果時發生錯誤: {str(e)}")

                
                else:
                    st.warning('模型預測失敗，請檢查輸入數據')
                
            except Exception as e:
                st.error(f'處理過程中發生錯誤: {str(e)}')
                st.error('請檢查輸入數據和參數設置')
                return 
        
        else:
            st.error('無法獲取股票資料，請確認股票代碼是否正確')

if __name__ == '__main__':
    main()