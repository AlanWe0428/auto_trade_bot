import streamlit as st
import google.generativeai as genai
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time

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

# --- 2. 核心數據處理 (短線優化) ---
def get_crypto_data(coin_symbol):
    try:
        ticker = f"{coin_symbol}-USD"
        # 短線交易抓取 1H 數據
        data = yf.download(ticker, period="1mo", interval="1h", progress=False, timeout=15)
        if data.empty: return None
        
        # 修正 yfinance 多重索引問題
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        data = data.dropna().astype(float)

        # 短線核心指標
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data['EMA20'] = ta.ema(data['Close'], length=20)
        
        # 取得最近 20 小時價格序列 (用於 AI 辨識 FVG 缺口)
        price_trend = data['Close'].tail(20).tolist()
        price_trend_str = ", ".join([f"{p:.2f}" for p in price_trend])
        
        # 斐波那契與 7D 關鍵位
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

# --- 3. 整合狙擊 Prompt ---
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

# --- 4. 側邊欄控制 ---
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

if st.button("🚀 開始掃描短線狙擊機會"):
    if not api_key:
        st.error("請提供 API Key")
    else:
        try:
            genai.configure(api_key=api_key)
            
            # --- 核心修復：動態偵測並選擇完整模型名稱 ---
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            
            # 優先順序匹配
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
            else:
                st.warning("數據抓取失敗，請檢查網路。")
                
        except Exception as e:
            st.error(f"分析失敗: {str(e)}")

# --- 6. 對話區 ---
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if inp := st.chat_input("詢問有關進場時機的細節..."):
    st.session_state.messages.append({"role": "user", "content": inp})
    with st.chat_message("user"): st.markdown(inp)
    with st.chat_message("assistant"):
        try:
            genai.configure(api_key=api_key)
            # 對話端也使用穩定的模型名稱
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            r = model.generate_content(inp)
            st.markdown(r.text)
            st.session_state.messages.append({"role": "assistant", "content": r.text})
        except:
            st.error("對話助手目前無法回應，請檢查 API 狀態。")
