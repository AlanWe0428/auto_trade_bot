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

# --- 2. 幣安核心功能模組 (強化餘額抓取) ---
def get_binance_client():
    try:
        api_key = st.secrets.get("BINANCE_API_KEY")
        api_secret = st.secrets.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            return None
        return Client(api_key, api_secret)
    except:
        return None

def get_futures_balance():
    client = get_binance_client()
    if not client:
        return "API 未設定"
    try:
        # 使用 futures_account 獲取 U 本位合約可用餘額
        acc_info = client.futures_account()
        for asset in acc_info.get('assets', []):
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
        return 0.0
    except Exception as e:
        return f"讀取失敗: {str(e)}"

def place_futures_order(symbol, side, leverage, usdt_amount, price, tp_price=None, sl_price=None):
    client = get_binance_client()
    if not client: return None
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        qty = round((usdt_amount * leverage) / price, 3) 
        
        main_order = client.futures_create_order(
            symbol=symbol, side=side, type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC, quantity=qty, price=str(round(price, 4))
        )
        # 止盈掛單
        if tp_price and tp_price > 0:
            tp_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
            client.futures_create_order(
                symbol=symbol, side=tp_side, type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=str(round(tp_price, 4)), closePosition=True
            )
        # 止損掛單
        if sl_price and sl_price > 0:
            sl_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
            client.futures_create_order(
                symbol=symbol, side=sl_side, type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=str(round(sl_price, 4)), closePosition=True
            )
        return main_order
    except Exception as e:
        st.error(f"下單失敗: {str(e)}")
        return None

# --- 3. 核心數據處理 (保留原版) ---
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
        
        return {
            "price": float(data.iloc[-1]['Close']), "rsi": float(data.iloc[-1]['RSI']), 
            "atr": float(data.iloc[-1]['ATR']), "sup_7d": float(data['Low'].tail(168).min()),
            "change": float((data.iloc[-1]['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100)
        }
    except Exception as e:
        st.error(f"數據抓取錯誤: {str(e)}")
        return None

# --- 4. 側邊欄 ---
with st.sidebar:
    st.header("🎯 狙擊手控制台")
    api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
    selected_coins = st.multiselect("追蹤幣種", ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP"], default=["BTC", "ETH", "SOL"])
    st.divider()
    st.subheader("🛠️ 技術指標")
    indicators = ["FVG 缺口回補", "0.618 關鍵位", "ATR 動態止損", "RSI 背離", "7日關鍵位", "形態學辨識"]

# --- 5. 主程式 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {"symbol": "BTCUSDT", "price": 0.0, "tp": 0.0, "sl": 0.0}

if st.button("🚀 開始掃描短線狙擊機會"):
    if not api_key: st.error("請提供 API Key")
    else:
        try:
            genai.configure(api_key=api_key)
            
            # --- 嚴格還原：您指定的模型連線邏輯 ---
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
                p_bar.progress((i+1)/len(selected_coins))
                res = get_crypto_data(coin)
                if res: all_data[coin] = res
                time.sleep(0.5)

            if all_data:
                data_content = ""
                for coin, d in all_data.items():
                    data_content += f"【{coin}】現價:{d['price']:.2f}, RSI:{d['rsi']:.1f}, ATR:{d['atr']:.2f}\n"
                
                prompt = f"你是一位 ICT 狙擊員。數據：{data_content}。請輸出格式：💎 [幣種] 報告，● 掛單區間：進場 [數值] | 止盈 [數值] | 止損 [數值]"
                
                response = model.generate_content(prompt)
                st.info(response.text)
                st.session_state.messages.append({"role":"assistant", "content": response.text})
                
                # 自動填充邏輯
                def clean_val(t):
                    c = re.sub(r'[^\d.]', '', t)
                    return float(c) if c else 0.0

                text = response.text
                p_m = re.search(r"進場\s*[:：]?\s*[\$]?\s*([\d,.]+)", text)
                tp_m = re.search(r"止盈\s*[:：]?\s*[\$]?\s*([\d,.]+)", text)
                sl_m = re.search(r"止損\s*[:：]?\s*[\$]?\s*([\d,.]+)", text)
                c_m = re.search(r"💎\s*\[?(\w+)", text)
                
                if p_m: st.session_state.last_analysis["price"] = clean_val(p_m.group(1))
                if tp_m: st.session_state.last_analysis["tp"] = clean_val(tp_m.group(1))
                if sl_m: st.session_state.last_analysis["sl"] = clean_val(sl_m.group(1))
                if c_m: st.session_state.last_analysis["symbol"] = c_m.group(1).upper() + "USDT"
                
                st.rerun() 
        except Exception as e: st.error(f"分析失敗: {e}")

# --- 6. 幣安下單面板 ---
st.divider()
st.subheader("🤖 快速執行幣安合約下單")
balance_res = get_futures_balance()

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trade_symbol = st.text_input("下單幣種", value=st.session_state.last_analysis["symbol"])
    with col2:
        trade_price = st.number_input("進場點位", value=float(st.session_state.last_analysis["price"]), format="%.2f")
    with col3:
        tp_input = st.number_input("止盈點位 (TP)", value=float(st.session_state.last_analysis["tp"]), format="%.2f")
    with col4:
        sl_input = st.number_input("止損點位 (SL)", value=float(st.session_state.last_analysis["sl"]), format="%.2f")

    col5, col6, col7 = st.columns(3)
    with col5:
        leverage = st.select_slider("槓桿倍數", options=list(range(1, 21)), value=8)
    with col6:
        input_usdt = st.number_input("保證金 (USDT)", min_value=0.0, step=10.0)
        if isinstance(balance_res, float):
            st.caption(f"💰 合約可用餘額：**{balance_res:.2f} USDT**")
        else:
            st.markdown(f"⚠️ <span style='color:red'>{balance_res}</span>", unsafe_allow_html=True)
    with col7:
        side_opt = st.radio("交易方向", ["做多 (BUY)", "做空 (SELL)"], horizontal=True)

    if st.button("🔥 確認下單", type="primary"):
        if input_usdt <= 0: st.warning("請輸入金額")
        else:
            final_side = SIDE_BUY if "BUY" in side_opt else SIDE_SELL
            res = place_futures_order(trade_symbol, final_side, leverage, input_usdt, trade_price, tp_input, sl_input)
            if res: st.success("✅ 訂單已發送！")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])
