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
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def format_number(number):
    """格式化數字"""
    return "{:,.2f}".format(number)

def prepare_lstm_data(data, lookback=60):
    """準備LSTM模型的輸入數據"""
    try:
        # 檢查並處理無效值
        data = np.array(data, dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=None, neginf=None)
        
        # 將極端值限制在合理範圍內
        mean = np.mean(data[np.isfinite(data)])
        std = np.std(data[np.isfinite(data)])
        data = np.clip(data, mean - 10*std, mean + 10*std)
        
        # 標準化數據
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
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
    """準備特徵數據"""
    if df is None or df.empty:
        return None
        
    try:
        df_features = pd.DataFrame()
        
        # 基本特徵
        df_features['Close'] = df['Close']
        df_features['Volume'] = df['Volume']
        
        # 處理極端值
        for col in ['Close', 'Volume']:
            mean = df_features[col].mean()
            std = df_features[col].std()
            df_features[col] = df_features[col].clip(lower=mean - 3*std, upper=mean + 3*std)
        
        # 價格特徵
        df_features['MA5'] = df['Close'].rolling(window=5).mean()
        df_features['MA10'] = df['Close'].rolling(window=10).mean()
        df_features['MA20'] = df['Close'].rolling(window=20).mean()
        df_features['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 價格變化率 - 添加數據驗證
        df_features['Price_Change'] = df['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
        df_features['Price_Change_5'] = df['Close'].pct_change(periods=5).replace([np.inf, -np.inf], np.nan)
        df_features['Price_Change_10'] = df['Close'].pct_change(periods=10).replace([np.inf, -np.inf], np.nan)
        
        # 波動率特徵
        df_features['Volatility'] = df['Close'].rolling(window=20).std()
        
        # 成交量特徵
        df_features['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df_features['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df_features['Volume_Change'] = df['Volume'].pct_change().replace([np.inf, -np.inf], np.nan)
        
        # 價格位置 - 添加數據驗證
        min_price = df['Close'].rolling(window=20).min()
        max_price = df['Close'].rolling(window=20).max()
        price_range = max_price - min_price
        df_features['Price_Position'] = np.where(
            price_range != 0,
            (df['Close'] - min_price) / price_range,
            0  # 當範圍為0時的默認值
        )
        
        # RSI指標 (14天) - 添加數據驗證
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # 避免除以0
        rs = np.where(loss != 0, gain/loss, 0)
        df_features['RSI'] = 100 - (100 / (1 + rs))
        
        # 填充NaN值
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        
        # 最後檢查是否有無限值
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(df_features.mean())
        
        return df_features
        
    except Exception as e:
        st.error(f"特徵準備過程中發生錯誤: {str(e)}")
        return None

# 修改 train_predict 函数，添加 train_test_ratio 参数
def train_predict(df, train_test_ratio=0.8):
    """訓練線性回歸模型並進行預測"""
    if df is None or df.empty:
        return None, None
        
    # 準備特徵
    features = ['MA5', 'MA10', 'MA20', 'MA60', 
               'Price_Change', 'Price_Change_5', 'Price_Change_10',
               'Volatility', 'Volume_MA5', 'Volume_MA20', 'Volume_Change',
               'Price_Position', 'RSI']
    
    try:
        X = df[features]
        y = df['Close']
        
        # 檢查數據有效性
        if X.isnull().any().any() or y.isnull().any():
            st.warning("數據中包含空值，將自動處理")
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
        
        # 檢查是否有無限值
        if np.any(np.isinf(X.values)) or np.any(np.isinf(y.values)):
            st.warning("數據中包含無限值，將自動處理")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
            y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
        
    except KeyError:
        st.error("特徵準備過程中出現錯誤，請確保所有必要的特徵都已計算")
        return None, None
    
    # 檢查數據是否足夠
    if len(df) < 60:
        st.warning("數據量不足，無法進行可靠的預測")
        return None, None
    
    try:
        # 修改分割比例
        test_size = 1 - train_test_ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 訓練模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 預測
        predictions = model.predict(X)
        test_predictions = model.predict(X_test)
        
        # 檢查預測結果是否有無效值
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            st.warning("預測結果中包含無效值，將進行修正")
            predictions = np.nan_to_num(predictions, nan=y.mean())
            test_predictions = np.nan_to_num(test_predictions, nan=y_test.mean())
        
        # 計算並顯示特徵重要性
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': abs(model.coef_)
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        st.subheader('特徵重要性')
        st.write(feature_importance)
        
        # 模型評估指標
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        
        st.write(f'訓練集 R² 分數: {train_score:.4f}')
        st.write(f'測試集 R² 分數: {test_score:.4f}')
        
        # 將預測結果轉換為pandas Series
        predictions = pd.Series(predictions, index=df.index)
        
        return predictions, (y_test, test_predictions)
    except Exception as e:
        st.error(f"模型訓練過程中發生錯誤: {str(e)}")
        return None, None
    
def get_market_suffix(market):
    """根據市場返回股票代碼後綴"""
    market_suffixes = {
        '台股': '.TW',
        '美股': '',
    }
    return market_suffixes.get(market, '')

def validate_stock_symbol(symbol, market):
    """驗證股票代碼格式"""
    if market == '台股':
        # 台股通常是4-6位數字
        if not symbol.isdigit() or len(symbol) < 4 or len(symbol) > 6:
            return False
    elif market == '美股':
        # 美股通常是1-5位字母
        if not symbol.isalpha() or len(symbol) > 5:
            return False
    return True

def get_stock_data(symbol, market, start_date, end_date):
    """獲取股票數據"""
    try:
        # 添加市場後綴
        full_symbol = symbol + get_market_suffix(market)
        
        # 獲取數據
        df = yf.download(full_symbol, start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"無法獲取{market}股票代碼 {symbol} 的數據，請確認代碼是否正確")
            return None
            
        return df
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

def calculate_backtest_metrics(y_true, y_pred):
    """改進的回測指標計算"""
    try:
        # 將數據轉換為numpy數組並確保為float類型
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        
        # 移除任何無效值
        valid_indices = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]
        
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("沒有有效的數據進行計算")
        
        # 計算基本誤差指標
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # 計算相對誤差（以百分比表示）
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 計算方向準確度
        y_true_direction = np.diff(y_true) > 0
        y_pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
        
        # 計算價格準確度（相對誤差的補數）
        price_accuracy = 100 - mape
        
        # 計算R平方值並轉換為百分比
        y_true_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_true_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        r2_percentage = r2 * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'price_accuracy': max(0, min(100, price_accuracy)),  # 限制在0-100範圍內
            'r2_percentage': max(-100, min(100, r2_percentage))  # 限制在合理範圍內
        }
        
    except Exception as e:
        st.error(f"計算指標時發生錯誤: {str(e)}")
        return {
            'mse': 0,
            'rmse': 0,
            'direction_accuracy': 0,
            'price_accuracy': 0,
            'r2_percentage': 0
        }
# 在顯示預測結果時的代碼修改
def display_prediction_results(last_actual, last_predicted, prediction_diff):
    """改進的預測結果顯示"""
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
    
def main():
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
        # [獲取股票信息部分保持不變]
        info = get_stock_info(stock_symbol, market)
        if info:
            try:
                # 顯示股票基本信息
                st.subheader('股票基本信息')
                
                # 公司名稱
                company_name = info.get('longName', '未知')
                st.write(f"**公司名稱:** {company_name}")
                
                # 行業資訊
                sector = info.get('sector', '未知')
                industry = info.get('industry', '未知')
                st.write(f"**行業:** {sector}")
                st.write(f"**產業:** {industry}")
            except:
                st.warning('無法獲取詳細的股票信息')
        
        # 顯示載入狀態
        with st.spinner('正在獲取股票數據...'):
            df = get_stock_data(stock_symbol, market, start_date, end_date)
        
        if df is not None and not df.empty:
            try:
                # 顯示基本資訊
                st.subheader('股票基本資訊')
                col1, col2, col3 = st.columns(3)
                
                try:
                    current_price = float(df['Close'].iloc[-1])
                    prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
                    price_change = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
                    volume = int(df['Volume'].iloc[-1])
                    
                    col1.metric("當前價格", f"${format_number(current_price)}")
                    col2.metric("日漲跌幅", f"{format_number(price_change)}%")
                    col3.metric("成交量", format_number(volume))
                except Exception as e:
                    st.error(f"處理價格數據時發生錯誤: {str(e)}")
                    return
                
                # 準備特徵和訓練模型
                with st.spinner('正在準備特徵數據...'):
                    df_features = prepare_features(df)
                
                if df_features is not None and not df_features.empty:
                    with st.spinner(f'正在訓練{model_type}模型...'):
                        if model_type == '線性回歸':
                            predictions, backtest_data = train_predict(df_features, train_test_ratio=train_test_split_ratio)
                        else:
                            predictions, backtest_data = train_predict_lstm(df, epochs=epochs, batch_size=batch_size, lookback=lookback)
                    
                            # 新的預測結果顯示部分
                        if predictions is not None and backtest_data is not None:
                            try:
                                # 確保預測結果有效
                                if isinstance(predictions, pd.Series):
                                    predictions = predictions.fillna(method='ffill').fillna(method='bfill')
                                    if predictions.empty or predictions.isna().all():
                                        st.error("無法生成有效的預測結果")
                                        return
                                elif isinstance(predictions, np.ndarray):
                                    predictions = pd.Series(predictions, index=df_features.index)
                                
                                # 獲取最新價格
                                last_actual = float(df_features['Close'].iloc[-1])
                                last_predicted = float(predictions.iloc[-1])
                                
                                # 檢查預測值是否在合理範圍內
                                price_std = df_features['Close'].std()
                                price_mean = df_features['Close'].mean()
                                if abs(last_predicted - price_mean) > 3 * price_std:
                                    st.warning("預測價格可能不夠準確，請謹慎參考")
                                
                                prediction_diff = ((last_predicted - last_actual) / last_actual) * 100
                                
                                # 顯示預測結果
                                st.subheader('預測結果')
                                col1, col2, col3 = st.columns(3)
                                col1.metric("最新實際價格", f"${format_number(last_actual)}")
                                col2.metric("預測價格", f"${format_number(last_predicted)}")
                                col3.metric("預測差異", f"{format_number(prediction_diff)}%",
                                        delta_color="normal" if abs(prediction_diff) < 10 else "off")
                                

                                # 計算並顯示可信度
                                confidence_score = max(0, min(100, 100 - abs(prediction_diff)))
                                
                                if confidence_score > 90:
                                    st.success(f"預測可信度: {confidence_score:.1f}% (高)")
                                elif confidence_score > 70:
                                    st.info(f"預測可信度: {confidence_score:.1f}% (中)")
                                else:
                                    st.warning(f"預測可信度: {confidence_score:.1f}% (低)")
                                
                                # 繪製股價圖
                                st.subheader('股價走勢與預測')
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=df_features.index,
                                    y=df_features['Close'],
                                    mode='lines',
                                    name='實際股價',
                                    line=dict(color='blue')
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=df_features.index,
                                    y=predictions,
                                    mode='lines',
                                    name='預測股價',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title=f'{market} {stock_symbol} 股價預測結果',
                                    xaxis_title='日期',
                                    yaxis_title='價格',
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig)
                                
                                # 回測結果
                                y_test, test_predictions = backtest_data
                                try:
                                    metrics = calculate_backtest_metrics(y_test, test_predictions)
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    # 左欄顯示誤差指標
                                    col1.markdown("**誤差指標**")
                                    col1.write(f"均方誤差 (MSE): {format_number(metrics['mse'])}")
                                    col1.write(f"均方根誤差 (RMSE): {format_number(metrics['rmse'])}")
                                    
                                    # 右欄顯示準確度指標
                                    col2.markdown("**準確度指標**")
                                    col2.write(f"方向準確度: {format_number(metrics['direction_accuracy'])}%")
                                    col2.write(f"價格準確度: {format_number(metrics['price_accuracy'])}%")
                                    col2.write(f"模型解釋能力: {format_number(metrics['r2_percentage'])}%")
                                    
                                    # 準確度解釋
                                    st.markdown("""
                                    **準確度指標說明：**
                                    - 方向準確度：預測價格變動方向（漲/跌）的準確程度
                                    - 價格準確度：預測價格與實際價格的接近程度
                                    - 模型解釋能力：模型對價格變動的解釋程度（R²）
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"計算回測指標時發生錯誤: {str(e)}")
                                
                                # 顯示原始資料
                                if st.checkbox('顯示原始資料'):
                                    st.subheader('原始資料')
                                    st.write(df)
                                    
                            except Exception as e:
                                st.error(f"處理預測結果時發生錯誤: {str(e)}")
                                
                        else:
                            st.warning('模型預測失敗，請檢查輸入數據')
                    
                else:
                    st.warning('無法生成有效的特徵數據')
        
            except Exception as e:
                st.error(f'處理數據時發生錯誤: {str(e)}')
                st.error('請檢查股票代碼是否正確，或是否有足夠的歷史數據')
        
        else:
            st.error('無法獲取股票資料，請確認股票代碼是否正確')

if __name__ == '__main__':
    main()