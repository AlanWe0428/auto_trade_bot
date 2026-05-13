import streamlit as st
import google.generativeai as genai
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time
import re
from binance.client import Client
from binance.enums import *

# --- 1. 頁面配置 ---
st.set_page_config(page_title="Crypto AI 短線狙擊儀", layout="wide", page_icon="🎯")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #00ffcc; color: #0e1117; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #00d1ad; color: white; }
    .stInfo { background-color: #1e2630; border-left: 5px solid #00ffcc; color: #ffffff; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 幣安核心功能模組 ---

def get_binance_client():
    """初始化幣安客戶端"""
    try:
        api_key = st.secrets.get("BINANCE_API_KEY")
        api_secret = st.secrets.get("BINANCE_API_SECRET")
        if not api_key or not api_secret: return None
        return Client(api_key, api_secret)
    except:
        return None

def get_futures_balance():
    """取得 U 本位合約帳戶可用 USDT 餘額"""
    client = get_binance_client()
    if not client: return 0.0
    try:
        balance = client.futures_account_balance()
        for asset in balance:
            if asset['asset'] == 'USDT':
                return float(asset['withdrawAvailable'])
        return 0.0
    except:
        return 0.0

def place_futures_order(symbol, side, leverage, usdt_amount, price):
    """執行下單：輸入 USDT 金額，自動換算數量並處理精度"""
    client = get_binance_client()
    if not client: return None
    try:
        # 1. 設定槓桿
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        
        # 2. 計算數量 (Qty = 投入保證金 * 槓桿 / 價格)
        raw_qty = (usdt_amount * leverage) / price
        # 簡易精度處理：大多數幣種取小數點後 2 位較安全
        qty = round(raw_qty, 2) 
        
        # 3. 執行限價單 (LIMIT)
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=qty,
            price=str(round(price, 4))
        )
        return order
    except Exception as e:
        st.error(f"❌ 幣安下單失敗: {str(e)}")
        return None

# --- 3. 數據抓取邏輯 ---

def get_crypto_data(coin_symbol):
    """抓取市場數據與技術指標"""
    try:
        ticker = f"{coin_symbol}-USD"
        data = yf.download(ticker, period="1mo", interval="1h", progress=False, timeout=15)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        data = data.dropna().astype(float)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        
        price_trend = data['Close'].tail(20).tolist()
        price_trend_str = ", ".join([f"{p:.2f}" for p in price_trend])
        
        high_30d, low_30d = float(data['High'].max()), float(data['Low'].min())
        latest = data.iloc[-1]
        
        return {
            "symbol": coin_symbol,
            "price": float(latest['Close']),
            "rsi": float(latest['RSI']),
            "atr": float(latest['ATR']),
            "price_trend": price_trend_str,
            "fib_0618": high_30d - 0.618 * (high_30d - low_30d),
            "change": float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100)
        }
    except Exception as e:
        st.error(f"數據抓取錯誤 ({coin_symbol}): {str(e)}")
        return None

# --- 4. 側邊欄與模型設定 ---

with st.sidebar:
    st.header("🎯 狙擊手控制台")
    api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
    selected_coins = st.multiselect("追蹤幣種", ["BTC", "ETH", "SOL", "BNB", "API3", "XRP"], default=["BTC", "ETH", "SOL"])
    
    st.divider()
    st.subheader("🛠️ 技術組合")
    indicators = []
    if st.checkbox("FVG 缺口分析", value=True): indicators.append("FVG 缺口回補")
    if st.checkbox("斐波那契匯流", value=True): indicators.append("0.618 關鍵位")
    if st.checkbox("ATR 波動止損", value=True): indicators.append("ATR 動態止損")

# --- 5. 主程式邏輯 ---

st.title("🎯 AI 短線高勝率狙擊儀")

# 初始化 Session State
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {"symbol": "BTCUSDT", "price": 0.0}
if "messages" not in st.session_state:
    st.session_state.messages = []

# 掃描按鈕
if st.button("🚀 開始掃描短線狙擊機會"):
    if not api_key:
        st.error("請在 Secrets 或側邊欄提供 Gemini API Key")
    else:
        try:
            genai.configure(api_key=api_key)
            
            # --- 動態模型偵測 (修復 404 錯誤) ---
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            target_model = next((m for m in models if "gemini-1.5-flash" in m), "models/gemini-1.5-flash")
            model = genai.GenerativeModel(target_model)
            
            all_data = {}
            p_bar = st.progress(0)
            for i, coin in enumerate(selected_coins):
                p_bar.progress((i+1)/len(selected_coins), text=f"正在分析 {coin} 盤面...")
                res = get_crypto_data(coin)
                if res: all_data[coin] = res
                time.sleep(0.5)

            if all_data:
                # 建立 AI Prompt
                data_content = ""
                for c, d in all_data.items():
                    data_content += f"【{c}】現價:{d['price']:.2f}, RSI:{d['rsi']:.1f}, ATR:{d['atr']:.2f}\n"
                
                prompt = f"你是一位精通 ICT 的狙擊交易員。數據：{data_content}\n請給出狙擊報告，格式包含：💎 [幣種]、進場 [數值]、止盈、止損。"
                
                with st.spinner("AI 正在計算匯流點..."):
                    response = model.generate_content(prompt)
                    st.info(response.text)
                    
                    # 自動解析建議點位
                    p_match = re.search(r"進場\s*\[?([\d\.]+)", response.text)
                    c_match = re.search(r"💎\s*\**\[?(\w+)", response.text)
                    if p_match: st.session_state.last_analysis["price"] = float(p_match.group(1))
                    if c_match: st.session_state.last_analysis["symbol"] = c_match.group(1).upper() + "USDT"
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"分析失敗: {str(e)}")

# --- 6. 幣安合約下單面板 ---

st.divider()
st.subheader("🤖 幣安合約快速執行面板")

# 實時餘額
usdt_balance = get_futures_balance()

with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        trade_symbol = st.text_input("下單幣種", value=st.session_state.last_analysis["symbol"])
    with col2:
        trade_price = st.number_input("建議點位", value=st.session_state.last_analysis["price"], format="%.4f")
    with col3:
        leverage = st.select_slider("槓桿倍數", options=list(range(1, 21)), value=5)

    col4, col5 = st.columns(2)
    with col4:
        input_usdt = st.number_input("投入保證金 (USDT)", min_value=0.0, step=10.0)
        st.caption(f"💰 合約可用餘額：**{usdt_balance:.2f} USDT**")
    with col5:
        side = st.radio("交易方向", ["做多 (BUY)", "做空 (SELL)"], horizontal=True)

    if st.button("🔥 執行幣安自動掛單", type="primary"):
        if input_usdt <= 0:
            st.warning("請輸入預計投入金額")
        elif input_usdt > usdt_balance:
            st.error("❌ 餘額不足")
        else:
            order_side = SIDE_BUY if "BUY" in side else SIDE_SELL
            with st.spinner("訂單發送中..."):
                res = place_futures_order(trade_symbol, order_side, leverage, input_usdt, trade_price)
                if res:
                    st.success(f"✅ 掛單成功！訂單 ID: {res['orderId']}")
                    st.balloons()

# --- 7. 對話紀錄顯示 ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])
