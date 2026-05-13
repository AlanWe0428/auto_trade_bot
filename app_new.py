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

# --- 2. 幣安核心功能模組 (強化版) ---
def get_binance_client():
    try:
        api_key = st.secrets.get("BINANCE_API_KEY")
        api_secret = st.secrets.get("BINANCE_API_SECRET")
        if not api_key or not api_secret: return None
        return Client(api_key, api_secret)
    except: return None

def get_futures_balance():
    client = get_binance_client()
    if not client: return 0.0
    try:
        # 確保抓取的是 U 本位合約帳戶
        balance = client.futures_account_balance()
        for asset in balance:
            if asset['asset'] == 'USDT':
                return float(asset['withdrawAvailable'])
        return 0.0
    except Exception as e:
        st.sidebar.error(f"餘額讀取異常: {e}")
        return 0.0

def place_futures_order(symbol, side, leverage, usdt_amount, price, tp_price=None, sl_price=None):
    client = get_binance_client()
    if not client: return None
    try:
        # 1. 設定槓桿
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        
        # 2. 計算數量 (Qty = 保證金 * 槓桿 / 價格)
        qty = round((usdt_amount * leverage) / price, 3) 
        
        # 3. 主訂單 (限價單)
        main_order = client.futures_create_order(
            symbol=symbol, side=side, type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC, quantity=qty, price=str(round(price, 4))
        )
        
        # 4. 止盈單 (Take Profit)
        if tp_price and tp_price > 0:
            tp_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
            client.futures_create_order(
                symbol=symbol, side=tp_side, type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=str(round(tp_price, 4)), closePosition=True
            )
            
        # 5. 止損單 (Stop Loss)
        if sl_price and sl_price > 0:
            sl_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
            client.futures_create_order(
                symbol=symbol, side=sl_side, type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=str(round(sl_price, 4)), closePosition=True
            )
            
        return main_order
    except Exception as e:
        st.error(f"❌ 幣安下單失敗: {str(e)}")
        return None

# --- 3. 核心數據處理 (完整保留原版指標) ---
def get_crypto_data(coin_symbol):
    try:
        ticker = f"{coin_symbol}-USD"
        data = yf.download(ticker, period="1mo", interval="1h", progress=False, timeout=15)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        data = data.dropna().astype(float)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data['EMA20'] = ta.ema(data['Close'], length=20)
        price_trend = data['Close'].tail(20).tolist()
        price_trend_str = ", ".join([f"{p:.2f}" for p in price_trend])
        high_30d, low_30d = float(data['High'].max()), float(data['Low'].min())
        recent_7d = data.tail(24 * 7)
        res_7d, sup_7d = float(recent_7d['High'].max()), float(recent_7d['Low'].min())
        latest = data.iloc[-1]
        return {
            "price": float(latest['Close']), "rsi": float(latest['RSI']), "atr": float(latest['ATR']),
            "price_trend": price_trend_str, "fib_0618": high_30d - 0.618 * (high_30d - low_30d),
            "res_7d": res_7d, "sup_7d": sup_7d,
            "change": float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100)
        }
    except Exception as e:
        st.error(f"數據抓取錯誤: {str(e)}")
        return None

def get_sniper_prompt(all_tech_data, indicators):
    indicator_text = "、".join(indicators) if indicators else "綜合技術指標"
    data_content = ""
    for coin, d in all_tech_data.items():
        data_content += f"【{coin}】現價:{d['price']:.2f}, RSI:{d['rsi']:.1f}, ATR:{d['atr']:.2f}, 7D支撐:{d['sup_7d']:.2f}\n"
    return f"""你是一位精通 ICT 與 SMC 的「短線狙擊交易員」。分析手法：{indicator_text}。
    數據：{data_content}
    請嚴格按此格式輸出：
    💎 **[幣種] 短線狙擊報告**
    ● **匯流辨識**：[描述]
    ● **策略評級**：[⭐ 強勢狙擊 / ✅ 觀察等待 / ❌ 放棄]
    ● **掛單區間**：進場 [數值] | 止盈 [數值] | 止損 [數值]
    ● **核心理由**：[內容]
    ---
    🏆 **今日最優狙擊機會**：[幣種名稱]"""

# --- 4. 側邊欄 (保留 6 個選項) ---
with st.sidebar:
    st.header("🎯 狙擊手控制台")
    api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
    selected_coins = st.multiselect("追蹤幣種", ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP"], default=["BTC", "ETH", "SOL"])
    st.divider()
    st.subheader("🛠️ 勝率提升組合 (多維匯流)")
    indicators = []
    c1, c2 = st.columns(2)
    with c1:
        if st.checkbox("FVG 缺口分析", value=True): indicators.append("FVG 缺口回補")
        if st.checkbox("斐波那契匯流", value=True): indicators.append("0.618 關鍵位")
        if st.checkbox("ATR 波動止損", value=True): indicators.append("ATR 動態止損")
    with c2:
        if st.checkbox("RSI 背離辨識", value=True): indicators.append("RSI 背離")
        if st.checkbox("7D 支撐壓力", value=True): indicators.append("7日關鍵位")
        if st.checkbox("形態學分析", value=True): indicators.append("形態學辨識")

# --- 5. 主程式 ---
st.title("🎯 AI 短線高勝率狙擊儀")

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {"symbol": "BTCUSDT", "price": 0.0, "tp": 0.0, "sl": 0.0}

if st.button("🚀 開始掃描短線狙擊機會"):
    if not api_key: st.error("請提供 API Key")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            all_data = {}
            p_bar = st.progress(0)
            for i, coin in enumerate(selected_coins):
                p_bar.progress((i+1)/len(selected_coins))
                res = get_crypto_data(coin)
                if res: all_data[coin] = res
                time.sleep(0.5)

            if all_data:
                response = model.generate_content(get_sniper_prompt(all_data, indicators))
                st.info(response.text)
                st.session_state.setdefault("messages", []).append({"role":"assistant", "content": response.text})
                
                # --- 強化版自動抓取點位 ---
                text = response.text
                p_match = re.search(r"進場\s*\[?([\d\.]+)", text)
                tp_match = re.search(r"止盈\s*\[?([\d\.]+)", text)
                sl_match = re.search(r"止損\s*\[?([\d\.]+)", text)
                c_match = re.search(r"💎\s*\**\[?(\w+)", text)
                
                if p_match: st.session_state.last_analysis["price"] = float(p_match.group(1))
                if tp_match: st.session_state.last_analysis["tp"] = float(tp_match.group(1))
                if sl_match: st.session_state.last_analysis["sl"] = float(sl_match.group(1))
                if c_match: st.session_state.last_analysis["symbol"] = c_match.group(1).upper() + "USDT"
                st.rerun() # 強制刷新介面以顯示抓取到的點位
        except Exception as e: st.error(f"分析失敗: {e}")

# --- 6. 幣安下單面板 (新增止盈止損) ---
st.divider()
st.subheader("🤖 快速執行幣安合約下單")
usdt_balance = get_futures_balance()

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trade_symbol = st.text_input("下單幣種", value=st.session_state.last_analysis["symbol"])
    with col2:
        trade_price = st.number_input("進場點位", value=st.session_state.last_analysis["price"], format="%.2f")
    with col3:
        tp_input = st.number_input("止盈點位 (TP)", value=st.session_state.last_analysis["tp"], format="%.2f")
    with col4:
        sl_input = st.number_input("止損點位 (SL)", value=st.session_state.last_analysis["sl"], format="%.2f")

    col5, col6, col7 = st.columns(3)
    with col5:
        leverage = st.select_slider("槓桿倍數", options=list(range(1, 21)), value=5)
    with col6:
        input_usdt = st.number_input("保證金 (USDT)", min_value=0.0, step=10.0)
        st.caption(f"💰 合約可用餘額：**{usdt_balance:.2f} USDT**")
    with col7:
        side_opt = st.radio("交易方向", ["做多 (BUY)", "做空 (SELL)"], horizontal=True)

    if st.button("🔥 確認下單", type="primary"):
        if input_usdt <= 0: st.warning("請輸入金額")
        elif input_usdt > usdt_balance: st.error("餘額不足")
        else:
            final_side = SIDE_BUY if "BUY" in side_opt else SIDE_SELL
            with st.spinner("正在執行複合訂單..."):
                res = place_futures_order(trade_symbol, final_side, leverage, input_usdt, trade_price, tp_input, sl_input)
                if res: st.success("✅ 下單、止盈、止損掛單成功！")

# --- 7. 對話紀錄 ---
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])
