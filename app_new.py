import streamlit as st
import google.generativeai as genai
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="Crypto AI 綜合交易儀", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; font-weight: bold; }
    .stInfo { background-color: #1e2630; border-left: 5px solid #00ffcc; color: #ffffff; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據處理函數 (針對雲端環境優化) ---
def get_crypto_data(coin_symbol):
    try:
        ticker = f"{coin_symbol}-USD"
        # 抓取數據
        data = yf.download(ticker, period="3mo", interval="1h", progress=False, timeout=15)
        if data.empty: return None
        
        # 【關鍵修正】處理 yfinance v0.2.x 可能產生的多重索引欄位
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        # 確保數據為浮點數且移除空值
        data = data.dropna().astype(float)

        # 計算技術指標
        data['EMA20'] = ta.ema(data['Close'], length=20)
        data['EMA50'] = ta.ema(data['Close'], length=50)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        # MACD 計算
        macd = ta.macd(data['Close'])
        if macd is not None:
            data = pd.concat([data, macd], axis=1)
        
        # 取得最近趨勢
        price_trend = data['Close'].tail(20).tolist()
        price_trend_str = ", ".join([f"{p:.2f}" for p in price_trend])
        
        # 斐波那契
        high_3m, low_3m = float(data['High'].max()), float(data['Low'].min())
        diff = high_3m - low_3m
        fib = {
            "0.618": high_3m - 0.618 * diff,
            "0.786": high_3m - 0.786 * diff
        }

        # 支撐壓力
        recent_7d = data.tail(24 * 7)
        res_7d, sup_7d = float(recent_7d['High'].max()), float(recent_7d['Low'].min())
        
        latest = data.iloc[-1]
        
        return {
            "price": float(latest['Close']),
            "rsi": float(latest['RSI']),
            "price_trend": price_trend_str,
            "fib": fib,
            "res_7d": res_7d, "sup_7d": sup_7d,
            "change": float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100)
        }
    except Exception as e:
        st.error(f"數據抓取錯誤 ({coin_symbol}): {str(e)}")
        return None

# --- 3. 打包分析提示詞 ---
def get_bulk_analysis_prompt(all_tech_data, active_indicators):
    indicator_text = ", ".join(active_indicators)
    data_content = ""
    for coin, d in all_tech_data.items():
        data_content += f"【{coin}】價格:{d['price']:.2f}, RSI:{d['rsi']:.1f}, 7D支撐:{d['sup_7d']:.2f}, 趨勢:[{d['price_trend']}]\n"

    return f"""你是一位量化交易官。請根據數據進行【{indicator_text}】綜合分析。
    {data_content}
    請為每個幣種提供：型態辨識、策略評級(⭐/✅/❌)、具體掛單區間、止盈止損、核心理由(30字內)。
    最後選出一個「今日最佳機會」。"""

# --- 4. 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 設置")
    # 支援從 Secrets 讀取，若無則顯示輸入框
    api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
    selected_coins = st.multiselect("分析幣種", ["BTC", "ETH", "SOL", "BNB", "XRP"], default=["BTC", "ETH"])
    indicators = ["斐波那契", "型態學", "指標分析"]
    
    if st.button("🗑️ 清除紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- 5. 執行診斷 ---
st.title("📈 Crypto AI 打包分析助理")

if st.button("🚀 執行一鍵綜合診斷"):
    if not api_key:
        st.error("請提供 API Key")
    else:
        try:
            genai.configure(api_key=api_key)
            # 優先嘗試 gemini-1.5-flash
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            all_data = {}
            p_bar = st.progress(0)
            for i, coin in enumerate(selected_coins):
                p_bar.progress((i+1)/len(selected_coins), text=f"抓取 {coin}...")
                res = get_crypto_data(coin)
                if res: all_data[coin] = res
                time.sleep(1)

            if all_data:
                with st.spinner("AI 分析中..."):
                    prompt = get_bulk_analysis_prompt(all_data, indicators)
                    response = model.generate_content(prompt)
                    st.info(response.text)
                    st.session_state.setdefault("messages", []).append({"role":"assistant", "content": response.text})
        except Exception as e:
            st.error(f"分析失敗: {str(e)}")

# --- 6. 對話助手 ---
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if inp := st.chat_input("詢問細節..."):
    st.session_state.messages.append({"role": "user", "content": inp})
    with st.chat_message("user"): st.markdown(inp)
    with st.chat_message("assistant"):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        r = model.generate_content(inp)
        st.markdown(r.text)
        st.session_state.messages.append({"role": "assistant", "content": r.text})
