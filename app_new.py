import streamlit as st
import google.generativeai as genai
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time

# --- 1. 頁面配置 ---
st.set_page_config(page_title="Crypto AI 短線狙擊儀", layout="wide", page_icon="🎯")

# 自定義專業風格 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #00ffcc; color: #0e1117; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #00d1ad; color: white; }
    .stInfo { background-color: #1e2630; border-left: 5px solid #00ffcc; color: #ffffff; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心數據處理 (短線優化) ---
def get_crypto_data(coin_symbol):
    try:
        ticker = f"{coin_symbol}-USD"
        # 短線交易抓取 1H 數據 (可手動改為 15m)
        data = yf.download(ticker, period="1mo", interval="1h", progress=False, timeout=15)
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        data = data.dropna().astype(float)

        # 短線核心指標
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data['EMA20'] = ta.ema(data['Close'], length=20)
        
        # 取得最近 20H 價格序列
        price_trend = data['Close'].tail(20).tolist()
        price_trend_str = ", ".join([f"{p:.2f}" for p in price_trend])
        
        # 斐波那契 & 7D 關鍵位
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
        st.error(f"數據錯誤: {str(e)}")
        return None

# --- 3. 整合勝率 Prompt ---
def get_sniper_prompt(all_tech_data, indicators):
    indicator_text = "、".join(indicators)
    
    data_content = ""
    for coin, d in all_tech_data.items():
        data_content += f"【{coin}】價格:{d['price']:.2f}, RSI:{d['rsi']:.1f}, ATR(波動):{d['atr']:.2f}, 7D支撐:{d['sup_7d']:.2f}, 序列:[{d['price_trend']}]\n"

    return f"""
    你是一位身經百戰的「短線狙擊交易員」。你的目標是在 1-24 小時內完成獲利。
    
    請使用以下匯流(Confluence)手法進行分析：【{indicator_text}】。
    
    分析邏輯指令：
    1. **尋找 FVG (流動性缺口)**：觀察價格序列，判斷是否存在未回補的跳空區域。
    2. **匯流檢查**：若價格同時觸及 0.618 斐波那契位且 RSI 超賣，則為「高勝率」訊號。
    3. **短線止損**：利用 ATR 的 1.5 倍設定動態止損，嚴格控制風險。
    4. **量價背離**：若價格創新低但 RSI 未創新低，提示潛在反轉。

    數據如下：
    {data_content}

    請輸出：
    💎 **[幣種] 短線診斷**
    ● **型態辨識**：[如：FVG 回補中、二探底、高位橫盤等]
    ● **勝率評級**：[⭐ 狙擊進場 / ✅ 觀察等待 / ❌ 放棄]
    ● **掛單建議**：進場 [精確數值] | 止盈 [精確數值] | 止損 [精確數值]
    ● **關鍵理由**：[限 30 字，需說明匯流點]
    ---
    🏆 **今日最優短線標的**：[幣種 + 原因]
    """

# --- 4. 側邊欄 ---
with st.sidebar:
    st.header("🎯 狙擊手面板")
    api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
    selected_coins = st.multiselect("追蹤幣種", ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP"], default=["BTC", "ETH", "SOL"])
    
    st.divider()
    st.subheader("🛠️ 勝率提升組合")
    indicators = []
    c1, c2 = st.columns(2)
    with c1:
        if st.checkbox("FVG 缺口回補", value=True): indicators.append("FVG 缺口")
        if st.checkbox("斐波那契 0.618", value=True): indicators.append("斐波那契匯流")
        if st.checkbox("ATR 動態止損", value=True): indicators.append("ATR 波動止損")
    with c2:
        if st.checkbox("RSI 背離辨識", value=True): indicators.append("RSI 背離")
        if st.checkbox("7D 關鍵位", value=True): indicators.append("支撐壓力位")
        if st.checkbox("型態學辨識", value=True): indicators.append("形態學")

# --- 5. 主程式 ---
st.title("🎯 Crypto AI 短線高勝率分析儀")

if st.button("🚀 開始掃描短線機會"):
    if not api_key:
        st.error("請提供 API Key")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            all_data = {}
            p_bar = st.progress(0)
            for i, coin in enumerate(selected_coins):
                p_bar.progress((i+1)/len(selected_coins), text=f"掃描 {coin} 盤面...")
                res = get_crypto_data(coin)
                if res: all_data[coin] = res
                time.sleep(1)

            if all_data:
                with st.spinner("AI 狙擊手正在計算匯流點..."):
                    prompt = get_sniper_prompt(all_data, indicators)
                    response = model.generate_content(prompt)
                    st.info(response.text)
                    st.session_state.setdefault("messages", []).append({"role":"assistant", "content": response.text})
        except Exception as e:
            st.error(f"分析失敗: {str(e)}")

# --- 6. 對話助手 ---
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if inp := st.chat_input("詢問具體進場時機..."):
    st.session_state.messages.append({"role": "user", "content": inp})
    with st.chat_message("user"): st.markdown(inp)
    with st.chat_message("assistant"):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        r = model.generate_content(inp)
        st.markdown(r.text)
        st.session_state.messages.append({"role": "assistant", "content": r.text})
