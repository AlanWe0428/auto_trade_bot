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

# --- 新增：幣安核心功能函數 (不影響原功能) ---
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
        balance = client.futures_account_balance()
        for asset in balance:
            if asset['asset'] == 'USDT':
                return float(asset['withdrawAvailable'])
        return 0.0
    except: return 0.0

def place_futures_order(symbol, side, leverage, usdt_amount, price):
    client = get_binance_client()
    if not client: return None
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        # 自動換算 Qty (保證金 * 槓桿 / 價格)
        qty = round((usdt_amount * leverage) / price, 2) 
        order = client.futures_create_order(
            symbol=symbol, side=side, type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC, quantity=qty, price=str(round(price, 4))
        )
        return order
    except Exception as e:
        st.error(f"❌ 幣安下單失敗: {str(e)}")
        return None

# --- 2. 核心數據處理 (完整保留原版) ---
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
            "price": float(latest['Close']),
            "rsi": float(latest['RSI']),
            "atr": float(latest['ATR']),
            "price_trend": price_trend_str,
            "fib_0618": high_30d - 0.618 * (high_30d - low_30d),
            "res_7d": res_7d, "sup_7d": sup_7d,
            "change": float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100)
        }
    except Exception as e:
        st.error(f"數據抓取錯誤 ({coin_symbol}): {str(e)}")
        return None

# --- 3. 整合狙擊 Prompt (完整保留原版) ---
def get_sniper_prompt(all_tech_data, indicators):
    indicator_text = "、".join(indicators) if indicators else "綜合技術指標"
    data_content = ""
    for coin, d in all_tech_data.items():
        data_content += f"【{coin}】現價:{d['price']:.2f} ({d['change']:.2f}%), RSI:{d['rsi']:.1f}, ATR波动:{d['atr']:.2f}, 7D支撐:{d['sup_7d']:.2f}, 趨勢:[{d['price_trend']}]\n"

    return f"""
    你是一位精通 ICT 策略與 SMC 核心的「短線狙擊交易員」。
    請利用以下手法進行【匯流分析】：【{indicator_text}】。
    
    分析邏輯：
    1. **FVG 辨識**：根據序列找出價格失衡區。
    2. **匯流(Confluence)**：尋找價格回補 FVG 且觸及 0.618 斐波那契位的重疊區。
    3. **風控**：以 ATR 的 1.5-2 倍設定止損，確保止損位在近期雜訊之外。
    4. **情緒**：判斷 RSI 是否與價格存在背離。

    數據：
    {data_content}

    請輸出格式：
    💎 **[幣種] 短線狙擊報告**
    ● **匯流辨識**：[描述 FVG、斐波那契、支撐位的重疊情況]
    ● **策略評級**：[⭐ 強勢狙擊 / ✅ 觀察等待 / ❌ 放棄]
    ● **掛單區間**：進場 [數值] | 止盈 [數值] | 止損 [數值]
    ● **核心理由**：[限 30 字內，指出關鍵支撐或背離點]
    ---
    🏆 **今日最優狙擊機會**：[幣種名稱 + 簡述原因]
    """

# --- 4. 側邊欄控制 (完整保留原版 6 個選項) ---
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

    if st.button("🗑️ 清除紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- 5. 主程式執行 ---
st.title("🎯 AI 短線高勝率狙擊儀")

# 初始化自動填單所需的 Session State
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {"symbol": "BTCUSDT", "price": 0.0}

if st.button("🚀 開始掃描短線狙擊機會"):
    if not api_key:
        st.error("請提供 API Key")
    else:
        try:
            genai.configure(api_key=api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            target_model = "models/gemini-1.5-flash"
            if target_model not in available_models:
                if "models/gemini-flash-latest" in available_models:
                    target_model = "models/gemini-flash-latest"
                else:
                    target_model = available_models[0] if available_models else "gemini-1.5-flash"
            
            model = genai.GenerativeModel(target_model)
            all_data = {}
            p_bar = st.progress(0)
            for i, coin in enumerate(selected_coins):
                p_bar.progress((i+1)/len(selected_coins), text=f"正在狙擊 {coin} 盤面...")
                res = get_crypto_data(coin)
                if res: all_data[coin] = res
                time.sleep(1)

            if all_data:
                with st.spinner(f"AI ({target_model}) 正在計算高勝率匯流點..."):
                    prompt = get_sniper_prompt(all_data, indicators)
                    response = model.generate_content(prompt)
                    st.info(response.text)
                    st.session_state.setdefault("messages", []).append({"role":"assistant", "content": response.text})
                    
                    # --- 自動抓取分析內容填入下單面板 ---
                    p_match = re.search(r"進場\s*\[?([\d\.]+)", response.text)
                    c_match = re.search(r"💎\s*\**\[?(\w+)", response.text)
                    if p_match: st.session_state.last_analysis["price"] = float(p_match.group(1))
                    if c_match: st.session_state.last_analysis["symbol"] = c_match.group(1).upper() + "USDT"
            else:
                st.warning("數據抓取失敗，請檢查網路。")
        except Exception as e:
            st.error(f"分析失敗: {str(e)}")

# --- 6. 新增功能：幣安合約快速下單面板 (位於掃描結果與對話區之間) ---
st.divider()
st.subheader("🤖 快速執行幣安合約下單")
usdt_balance = get_futures_balance()

with st.container(border=True):
    col_sym, col_pri, col_lev = st.columns(3)
    with col_sym:
        trade_symbol = st.text_input("下單幣種", value=st.session_state.last_analysis["symbol"])
    with col_pri:
        trade_price = st.number_input("掛單進場點位", value=st.session_state.last_analysis["price"], format="%.4f")
    with col_lev:
        leverage = st.selectbox("槓桿倍數 (1-20x)", options=list(range(1, 21)), index=4)

    col_cost, col_side, col_btn = st.columns([3, 3, 2])
    with col_cost:
        input_usdt = st.number_input("下單保證金 (USDT)", min_value=0.0, step=10.0)
        st.caption(f"💰 合約可用餘額：**{usdt_balance:.2f} USDT**")
    with col_side:
        side_opt = st.radio("交易方向", ["做多 (BUY)", "做空 (SELL)"], horizontal=True)
    with col_btn:
        st.write("") # 調整按鈕對齊
        if st.button("🔥 確認下單", type="primary"):
            if input_usdt <= 0: st.warning("請輸入金額")
            elif input_usdt > usdt_balance: st.error("餘額不足")
            else:
                final_side = SIDE_BUY if "BUY" in side_opt else SIDE_SELL
                with st.spinner("正在掛單..."):
                    res = place_futures_order(trade_symbol, final_side, leverage, input_usdt, trade_price)
                    if res: st.success(f"✅ 下單成功！ID: {res['orderId']}")

# --- 7. 對話區 (完整保留原版) ---
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if inp := st.chat_input("詢問有關進場時機的細節..."):
    st.session_state.messages.append({"role": "user", "content": inp})
    with st.chat_message("user"): st.markdown(inp)
    with st.chat_message("assistant"):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            r = model.generate_content(inp)
            st.markdown(r.text)
            st.session_state.messages.append({"role": "assistant", "content": r.text})
        except:
            st.error("對話助手目前無法回應，請檢查 API 狀態。")
