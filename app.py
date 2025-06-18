import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import requests

# --- CONFIGURATION (LISANT LES SECRETS ET PARAMÈTRES) ---
try:
    API_KEY = st.secrets["TWELVEDATA_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Secret 'TWELVEDATA_API_KEY' non trouvé. Veuillez le configurer dans les paramètres de votre application.")
    st.stop()

TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
INTERVAL = "1h" 
OUTPUT_SIZE = 200 # Augmenté pour avoir assez de données pour les calculs complexes

# --- FONCTIONS HELPER POUR INDICATEURS COMPLEXES ---
def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = pd.Series(range(1, length + 1))
    return series.rolling(length).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average"""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    return wma(2 * wma(series, half_length) - wma(series, length), sqrt_length)

# --- FETCH DATA ---
@st.cache_data(ttl=900)
def get_data(symbol):
    try:
        params = {"symbol": symbol, "interval": INTERVAL, "outputsize": OUTPUT_SIZE, "apikey": API_KEY, "timezone": "UTC"}
        r = requests.get(TWELVE_DATA_API_URL, params=params)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "error":
            st.error(f"Erreur API pour {symbol}: {j.get('message', 'Format de réponse inconnu.')}")
            return None
        df = pd.DataFrame(j["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        df = df.astype(float)
        df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
        return df[['Open','High','Low','Close']]
    except Exception as e:
        st.error(f"Erreur de traitement pour {symbol}: {e}")
        return None

# --- PAIRS ---
FOREX_PAIRS_TD = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP", "XAU/USD"]

# --- INDICATEURS (Fonctions de calcul) ---
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()
def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def adx(h, l, c, di_len, adx_len):
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    atr = rma(tr, di_len)
    up = h.diff(); down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di = 100 * rma(pd.Series(plus_dm, index=h.index), di_len) / atr.replace(0, 1e-9)
    minus_di = 100 * rma(pd.Series(minus_dm, index=h.index), di_len) / atr.replace(0, 1e-9)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)
    return rma(dx, adx_len)

def rsi(src, p):
    d = src.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    rs = rma(g, p) / rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

# --- CALCUL DES SIGNAUX (TRADUCTION EXACTE DU PINE SCRIPT) ---
def calculate_signals(df):
    if df is None or len(df) < 100: # Besoin de plus de données pour HMA, etc.
        return None

    # --- INPUTS from Pine Script ---
    hmaLength = 20
    adxThreshold = 20
    rsiLength = 10
    adxLength = 14
    diLength = 14
    ichimokuTenkan = 9
    sha_len1 = 10
    sha_len2 = 10
    
    signals = {}
    bullConfluences = 0
    bearConfluences = 0

    # 1. HMA
    hma_series = hma(df['Close'], hmaLength)
    hmaSlope = 1 if hma_series.iloc[-1] > hma_series.iloc[-2] else -1
    signals['HMA'] = "▲" if hmaSlope == 1 else "▼"
    
    # 2. Heiken Ashi (Standard)
    ha_close = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    ha_open = pd.Series(np.nan, index=df.index)
    ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    haSignal = 1 if ha_close.iloc[-1] > ha_open.iloc[-1] else -1
    signals['Heikin Ashi'] = "▲" if haSignal == 1 else "▼"

    # 3. Smoothed Heiken Ashi
    o = ema(df['Open'], sha_len1)
    c = ema(df['Close'], sha_len1)
    h = ema(df['High'], sha_len1)
    l = ema(df['Low'], sha_len1)
    haclose_s = (o + h + l + c) / 4
    haopen_s = pd.Series(np.nan, index=df.index)
    haopen_s.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(df)):
        haopen_s.iloc[i] = (haopen_s.iloc[i-1] + haclose_s.iloc[i-1]) / 2
    o2 = ema(haopen_s, sha_len2)
    c2 = ema(haclose_s, sha_len2)
    smoothedHaSignal = 1 if o2.iloc[-1] <= c2.iloc[-1] else -1 # Replicates o2 > c2 ? -1 : 1
    signals['Smoothed HA'] = "▲" if smoothedHaSignal == 1 else "▼"
    
    # 4. RSI
    ohlc4 = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    rsi_series = rsi(ohlc4, rsiLength)
    rsiSignal = 1 if rsi_series.iloc[-1] > 50 else -1
    signals['RSI'] = f"{int(rsi_series.iloc[-1])} {'▲' if rsiSignal == 1 else '▼'}"

    # 5. ADX
    adx_series = adx(df['High'], df['Low'], df['Close'], diLength, adxLength)
    adxHasMomentum = adx_series.iloc[-1] >= adxThreshold
    signals['ADX'] = f"{int(adx_series.iloc[-1])} {'💪' if adxHasMomentum else '💤'}"
    
    # 6. Ichimoku
    tenkan = (df['High'].rolling(ichimokuTenkan).max() + df['Low'].rolling(ichimokuTenkan).min()) / 2
    kijun = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    senkouA = (tenkan + kijun) / 2
    senkouB = (df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2
    cloudTop = np.maximum(senkouA.iloc[-1], senkouB.iloc[-1])
    cloudBottom = np.minimum(senkouA.iloc[-1], senkouB.iloc[-1])
    price = df['Close'].iloc[-1]
    ichimokuSignal = 1 if price > cloudTop else -1 if price < cloudBottom else 0
    signals['Ichimoku'] = "▲" if ichimokuSignal == 1 else "▼" if ichimokuSignal == -1 else "─"

    # --- CONFLUENCE COUNTING (Exact Replica) ---
    bullConfluences += 1 if hmaSlope == 1 else 0
    bullConfluences += 1 if haSignal == 1 else 0
    bullConfluences += 1 if smoothedHaSignal == 1 else 0
    bullConfluences += 1 if rsiSignal == 1 else 0
    bullConfluences += 1 if adxHasMomentum else 0 # Adds to both if true
    bullConfluences += 1 if ichimokuSignal == 1 else 0

    bearConfluences += 1 if hmaSlope == -1 else 0
    bearConfluences += 1 if haSignal == -1 else 0
    bearConfluences += 1 if smoothedHaSignal == -1 else 0
    bearConfluences += 1 if rsiSignal == -1 else 0
    bearConfluences += 1 if adxHasMomentum else 0 # Adds to both if true
    bearConfluences += 1 if ichimokuSignal == -1 else 0

    # --- Final Result ---
    if bullConfluences > bearConfluences:
        direction = "HAUSSIER"
        confluence = bullConfluences
    elif bearConfluences > bullConfluences:
        direction = "BAISSIER"
        confluence = bearConfluences
    else:
        direction = "NEUTRE"
        confluence = bullConfluences # or bear, they are equal

    stars = "⭐" * confluence
    return {"confluence": confluence, "direction": direction, "stars": stars, "signals": signals}

# --- INTERFACE UTILISATEUR ---
st.set_page_config(layout="wide", page_title="Scanner Canadian Confluence")
st.title("Scanner Canadian Confluence Premium")

st.sidebar.header("Paramètres")
min_conf = st.sidebar.slider("Confluence minimale", 1, 6, 3, help="Filtrez les paires qui ont au moins ce nombre de signaux concordants.")
show_all = st.sidebar.checkbox("Afficher toutes les paires", value=False)

if st.sidebar.button("Lancer le scan"):
    results = []
    total_pairs = len(FOREX_PAIRS_TD)
    progress_bar = st.progress(0, text="Lancement du scan...")

    for i, symbol in enumerate(FOREX_PAIRS_TD):
        progress_text = f"Scan en cours... {symbol} ({i+1}/{total_pairs})"
        progress_bar.progress((i + 1) / total_pairs, text=progress_text)
        df = get_data(symbol)
        time.sleep(8)
        
        if df is not None:
            res = calculate_signals(df)
            if res and (show_all or res['confluence'] >= min_conf):
                row = {"Paire": symbol, "Confluences": res['stars'], "Direction": res['direction'], "confluence_score": res['confluence']}
                row.update(res['signals'])
                results.append(row)
    
    progress_bar.empty()

    if results:
        df_res = pd.DataFrame(results).sort_values(by="confluence_score", ascending=False)
        # Reorder columns to match the indicator logic
        column_order = ["Paire", "Confluences", "Direction", "HMA", "Heikin Ashi", "Smoothed HA", "RSI", "ADX", "Ichimoku"]
        df_display = df_res.drop(columns=['confluence_score'])[column_order]
        
        def style_direction(direction):
            if direction == 'HAUSSIER': return 'color: #2ECC71'
            elif direction == 'BAISSIER': return 'color: #E74C3C'
            return 'color: #888888'

        st.dataframe(df_display.style.applymap(style_direction, subset=['Direction']), use_container_width=True)
        
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(label="📂 Exporter en CSV", data=csv, file_name="resultats_confluence.csv", mime="text/csv")
    else:
        st.warning("Aucun résultat correspondant aux critères.")

st.caption(f"Données basées sur l'intervalle {INTERVAL}. Dernière mise à jour : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
