import streamlit as st
import google.generativeai as genai
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="Crypto AI 綜合交易儀", layout="wide", page_icon="📈")

# 自定義 CSS 樣式
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; font-weight: bold; }
    .stInfo { background-color: #1e2630; border-left: 5px solid #00ffcc; color: #ffffff; padding: 15px; border-radius: 5px; }
    .stSuccess { background-color: #1e2630; border-left: 5px solid #ffcc00; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據處理函數 ---
def get_crypto_data(coin_symbol):
    try:
        ticker = f"{coin_symbol}-USD"
        # 抓取數據
        data = yf.download(ticker, period="3mo", interval="1h", progress=False, timeout=15)
        if data.empty: return None
        
        # 處理多重索引問題
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        data = data.astype(float)

        # 計算指標
        data['EMA20'] = ta.ema(data['Close'], length=20)
        data['EMA50'] = ta.ema(data['Close'], length=50)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        macd = ta.macd(data['Close'])
        data = pd.concat([data, macd], axis=1)
        
        # 價格序列 (用於 AI 辨識型態)
        price_trend = data['Close'].tail(20).tolist()
        price_trend_str = ", ".join([f"{p:.2f}" for p in price_trend])
        
        # 斐波那契計算
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
        data_content += f"""
        【{coin} 數據】
        當前價格: {d['price']:.2f} ({d['change']:.2f}%) | RSI: {d['rsi']:.1f}
        20H價格序列: [{d['price_trend']}]
        斐波那契 0.618: {d['fib']['0.618']:.2f}
        7D壓力/支撐: {d['res_7d']:.2f} / {d['sup_7d']:.2f}
        ---"""

    prompt = f"""
    你是一位專業的量化交易員。請根據以下數據進行【{indicator_text}】綜合分析。
    
    {data_content}

    請嚴格遵守格式回覆，為每個幣種提供獨立區塊：

    💎 **[幣種名稱] 策略**
    ● **型態辨識**：[辨識結果]
    ● **策略評級**：[⭐ 強勢進場 / ✅ 分批建倉 / ❌ 絕對觀望]
    ● **建議區間**：進場 [數值] | 止盈 [數值] | 止損 [數值]
    ● **核心理由**：[限30字]

    ---
    🏆 **今日最佳機會**：[選出一個幣種並簡述理由]
    """
    return prompt

# --- 4. 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 設置")
    api_key = st.text_input("Gemini API Key", type="password")
    selected_coins = st.multiselect("分析幣種", ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP"], default=["BTC", "ETH"])
    
    st.divider()
    st.subheader("🛠️ 分析手法")
    indicators = []
    if st.checkbox("斐波那契回撤", value=True): indicators.append("斐波那契")
    if st.checkbox("型態學辨識", value=True): indicators.append("型態學")
    if st.checkbox("RSI/均線指標", value=True): indicators.append("指標分析")
    
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- 5. 主程式執行 ---
st.title("📈 Crypto AI 打包分析助理")

if st.button("🚀 執行一鍵綜合診斷"):
    if not api_key:
        st.error("請在側邊欄輸入 API Key")
    else:
        try:
            genai.configure(api_key=api_key)
            
            # --- 核心修復：動態偵測可用模型 ---
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # 優先順序：1.5-flash -> flash-latest -> flash
            target_model = "gemini-1.5-flash" 
            if "models/gemini-1.5-flash" not in available_models:
                if "models/gemini-flash-latest" in available_models:
                    target_model = "gemini-flash-latest"
            
            model = genai.GenerativeModel(target_model)
            
            all_data = {}
            progress_bar = st.progress(0, text="準備抓取數據...")
            
            for i, coin in enumerate(selected_coins):
                progress_bar.progress((i + 1) / len(selected_coins), text=f"正在獲取 {coin} 市場數據...")
                res = get_crypto_data(coin)
                if res:
                    all_data[coin] = res
                time.sleep(1) # 保護 yfinance 頻率

            if all_data:
                with st.spinner(f"AI ({target_model}) 正在分析中..."):
                    prompt = get_bulk_analysis_prompt(all_data, indicators)
                    response = model.generate_content(prompt)
                    
                    st.success("✅ 診斷報告已生成")
                    st.info(response.text)
                    
                    # 儲存對話紀錄
                    if "messages" not in st.session_state: st.session_state.messages = []
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
            else:
                st.warning("未能獲取任何有效數據。")
                
        except Exception as e:
            st.error(f"發生錯誤: {str(e)}")
            st.write("建議檢查：1. API Key 是否正確 2. 網路是否能連接 Google 服務 3. 執行 pip install -U google-generativeai")

# --- 6. 對話助手 ---
st.divider()
st.subheader("💬 策略深談")
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if inp := st.chat_input("詢問有關上方報告的細節..."):
    st.session_state.messages.append({"role": "user", "content": inp})
    with st.chat_message("user"): st.markdown(inp)
    
    with st.chat_message("assistant"):
        try:
            # 這裡同樣使用修正後的模型宣告方式
            model = genai.GenerativeModel("gemini-1.5-flash")
            r = model.generate_content(inp, stream=True)
            full_response = ""
            msg_placeholder = st.empty()
            for chunk in r:
                full_response += chunk.text
                msg_placeholder.markdown(full_response + "▌")
            msg_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"對話出錯: {str(e)}")