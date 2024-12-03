import yfinance as yf
from django.shortcuts import render
import matplotlib.pyplot as plt
import matplotlib
import os
from django.conf import settings


def format_number(number):
    """格式化數字"""
    return "{:,.2f}".format(number)

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

def fetch_stock_data(request):
    if request.method == 'POST':
        stock_code = request.POST.get('stock_code')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        market = request.POST.get('market')

        try:
            # 驗證股票代碼
            if not validate_stock_symbol(stock_code, market):
                raise ValueError(f"無效的股票代碼格式: {stock_code} ({market})")

            # 添加市場後綴
            full_symbol = stock_code + get_market_suffix(market)

            # 獲取股票數據
            stock_data = yf.download(full_symbol, start=start_date, end=end_date)

            if stock_data.empty:
                raise ValueError(f"無法獲取 {market} 股票代碼 {stock_code} 的數據，請確認代碼是否正確")

            # 獲取最新價格和日漲跌幅
            latest_price = float(stock_data['Close'].iloc[-1])
            daily_change = float(((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]) * 100)




            # 傳遞數據到模板
            context = {
                'stock_code': stock_code,
                'latest_price': f"${format_number(latest_price)}",
                'daily_change': f"{format_number(daily_change)}%",
                'start_date': start_date,
                'end_date': end_date,
            }

            # 生成圖表
            plot_path = os.path.join(settings.BASE_DIR, 'stocks/static/stock_plot.png')
            stock_data['Close'].plot(title=f"{stock_code} Stock Price")
            plt.savefig(plot_path)
            plt.close()

            return render(request, 'stocks/result.html', context)

        except Exception as e:
            return render(request, 'stocks/query.html', {'error': f"Error: {e}"})

    return render(request, 'stocks/query.html')
matplotlib.use('Agg')  # 使用非交互式後端
plt.savefig('stocks/static/stock_plot.png')
plt.close()