import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet 


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Superstore Dashboard", layout="wide")

st.title("📊 Superstore Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Sample - Superstore.csv", encoding='windows-1254')
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['ship_date'] = pd.to_datetime(df['ship_date'])
    return df

df = load_data()

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("🔍 Filter")

region = st.sidebar.multiselect(
    "Region",
    df["region"].unique(),
    default=df["region"].unique()
)

category = st.sidebar.multiselect(
    "Category",
    df["category"].unique(),
    default=df["category"].unique()
)

df = df[(df["region"].isin(region)) & (df["category"].isin(category))]

# =========================
# KPI
# =========================
total_sales = df["sales"].sum()
total_profit = df["profit"].sum()
total_quantity = df["quantity"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Sales", f"${total_sales:,.0f}")
col2.metric("📈 Profit", f"${total_profit:,.0f}")
col3.metric("📦 Quantity", f"{total_quantity:,}")

# =========================
# AUTO INSIGHT FUNCTION
# =========================
def generate_insight(df):
    insight = []

    state_sales = df.groupby("state")["sales"].sum().sort_values(ascending=False)
    top_state = state_sales.idxmax()
    top_value = state_sales.max()

    insight.append(f"🏆 Top Sales State: {top_state} (${top_value:,.0f})")

    state_profit = df.groupby("state")["profit"].sum()
    low_state = state_profit.idxmin()
    low_value = state_profit.min()

    insight.append(f"⚠️ Lowest Profit State: {low_state} (${low_value:,.0f})")

    cat = df.groupby("category")["profit"].sum()
    best_cat = cat.idxmax()

    insight.append(f"📦 Most Profitable Category: {best_cat}")

    if df["discount"].mean() > 0.2:
        insight.append("⚠️ High discount detected, may reduce profit")

    return insight


def generate_recommendation(df):
    rec = []

    if df["discount"].mean() > 0.2:
        rec.append("Reduce discount strategy to improve profit")

    low_states = df.groupby("state")["profit"].sum().nsmallest(3).index.tolist()
    rec.append(f"Focus improvement on: {', '.join(low_states)}")

    return rec


def narrative(df):
    top_state = df.groupby("state")["sales"].sum().idxmax()
    top_cat = df.groupby("category")["profit"].sum().idxmax()

    return f"""
    Penjualan tertinggi berasal dari {top_state}. 
    Category {top_cat} memberikan kontribusi profit terbesar.
    Beberapa wilayah dengan profit rendah perlu evaluasi strategi bisnis.
    """

# =========================
# Modeling & Evaluation Function
# =========================

def compute_best_model(df_ts, horizon=30, step=30):

    results = []

    for start in range(0, len(df_ts) - horizon, step):
        train = df_ts[:start + horizon]
        test = df_ts[start + horizon:start + horizon + step]

        if len(test) == 0:
            continue

        try:
            # =========================
            # PROPHET
            # =========================
            model = Prophet()
            model.fit(train)

            future = model.make_future_dataframe(periods=len(test))
            forecast = model.predict(future)

            y_true = test["y"].values
            y_pred_prophet = forecast["yhat"].iloc[-len(test):].values

            if len(y_true) != len(y_pred_prophet):
                continue

            mae_p, rmse_p = evaluate(y_true, y_pred_prophet)

            # =========================
            # MOVING AVERAGE
            # =========================
            train_ma = train.copy()
            train_ma["ma"] = train_ma["y"].rolling(7).mean()

            # handle NaN
            if train_ma["ma"].dropna().empty:
                continue

            last_ma = train_ma["ma"].dropna().iloc[-1]
            y_pred_ma = np.full(len(test), last_ma)

            if len(y_true) != len(y_pred_ma):
                continue

            mae_ma, rmse_ma = evaluate(y_true, y_pred_ma)

            # =========================
            # SAVE RESULT
            # =========================
            results.append({
                "window": start,
                "mae_prophet": mae_p,
                "rmse_prophet": rmse_p,
                "mae_ma": mae_ma,
                "rmse_ma": rmse_ma
            })

        except Exception:
            continue  # skip kalau error

    df_res = pd.DataFrame(results)

    # =========================
    # PILIH MODEL TERBAIK
    # =========================
    if df_res.empty:
        return "No Model", df_res

    if df_res["rmse_prophet"].mean() < df_res["rmse_ma"].mean():
        best_model = "Prophet"
    else:
        best_model = "Moving Average"

    return best_model, df_res

# =========================
# GLOBAL COMPUTATION
# =========================
df_ts = df.groupby("order_date")["sales"].sum().reset_index()

df_ts = df_ts.rename(columns={
    "order_date": "ds",
    "sales": "y"
})

if "best_model" not in st.session_state:
    best_model, df_res = compute_best_model(df_ts)
    st.session_state["best_model"] = best_model
    st.session_state["df_res"] = df_res

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧠 Business & Data",
    "📊 Overview",
    "🗺️ Region",
    "📦 Category",
    "🔍 Drilldown",
    "📈 Forecast",
    "🤖 Modeling & Evaluation"
])

with tab1:
    st.subheader("🧠 Business & Data Understanding")

    # =========================
    # BUSINESS OBJECTIVE
    # =========================
    st.markdown("## 🎯 Business Objective")

    st.write("""
    Fokus utama adalah memahami faktor-faktor yang memengaruhi performa penjualan serta hubungan antara sales, profit, dan discount untuk mendukung pengambilan keputusan bisnis.

    Beberapa area utama yang menjadi perhatian meliputi:
    - Identifikasi wilayah dan kategori dengan kontribusi penjualan tertinggi dan terendah
    - Deteksi area dengan profit rendah sebagai dasar evaluasi strategi pricing dan operasional
    - Pemahaman pola penjualan untuk meningkatkan efektivitas distribusi dan pemasaran

    Data historis kemudian dimanfaatkan untuk membangun model forecasting guna memproyeksikan penjualan dalam jangka pendek (± 3 bulan ke depan).

    Hasil prediksi menunjukkan arah tren penjualan yang dapat digunakan untuk:
    - Mengoptimalkan perencanaan inventory dan supply chain
    - Menentukan waktu promosi dan campaign secara lebih tepat
    - Mengantisipasi risiko overstock maupun understock berdasarkan proyeksi demand

    Dengan menggabungkan pemahaman historis dan hasil forecasting, keputusan bisnis dapat diambil secara lebih terukur, adaptif, dan proaktif terhadap perubahan tren pasar.
    """)

    # =========================
    # TECHNICAL OBJECTIVE
    # =========================
    st.markdown("## ⚙️ Technical Objective")

    st.write("""
    Dari sisi teknis, pendekatan yang digunakan berfokus pada pembangunan model time series forecasting untuk memprediksi penjualan di masa depan berdasarkan data historis.

    Tahapan yang dilakukan meliputi:
    - Eksplorasi data untuk mengidentifikasi tren dan pola musiman (seasonality)
    - Pembangunan model menggunakan Prophet untuk menangkap komponen trend dan seasonality
    - Evaluasi performa model menggunakan metrik seperti MAE, dan RMSE
    - Analisis residual untuk memastikan model tidak bias dan memiliki error yang bersifat acak

    Model forecasting yang dibangun menghasilkan:
    - Estimasi penjualan jangka pendek (± 3 bulan ke depan)
    - Indikasi arah tren (peningkatan atau penurunan)
    - Insight pola musiman yang dapat dimanfaatkan dalam strategi bisnis

    Hasil prediksi ini digunakan untuk:
    - Perencanaan inventory dan supply chain
    - Penentuan waktu promosi dan campaign secara lebih tepat
    - Mendukung pengambilan keputusan strategis berbasis data prediktif
    """)

    # =========================
    # DATA PREVIEW
    # =========================
    st.markdown("## 📊 Data Preview")

    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", len(df))
    col2.metric("Jumlah Kolom", df.shape[1])
    min_date = df["order_date"].min().strftime("%d %b %Y")
    max_date = df["order_date"].max().strftime("%d %b %Y")

    st.caption(f"Periode data: {min_date} hingga {max_date}")

    # =========================
    # DATA TRANSFORMATION
    # =========================
    st.markdown("## 🔧 Data Transformation")

    # ---------- DATE FIX ----------
    st.markdown("### 📅 Perbaikan Format Tanggal")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Before")
        st.dataframe(df[["order_date"]].head())
        st.caption(f"Tipe data: {df['order_date'].dtype}")

    # convert
    df["order_date"] = pd.to_datetime(df["order_date"])

    with col2:
        st.write("After")
        st.dataframe(df[["order_date"]].head())
        st.caption(f"Tipe data: {df['order_date'].dtype}")

    # ---------- AGGREGATION ----------
    st.markdown("### 📊 Agregasi Penjualan per Hari")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Data Mentah")
        st.dataframe(df[["order_date", "sales"]].head())

    with col2:
        st.write("Setelah Agregasi (Daily)")
        st.dataframe(df_ts.head())

    # =========================
    # DATA SIZE COMPARISON
    # =========================
    st.markdown("### 📉 Perbandingan Jumlah Data")

    col1, col2 = st.columns(2)
    col1.metric("Data Awal", len(df))
    col2.metric("Data Setelah Agregasi", len(df_ts))

    # =========================
    # TRANSFORMATION INSIGHT
    # =========================
    st.markdown("## 🧠 Insight Transformasi")

    st.write("""
    - Format tanggal diperbaiki untuk memastikan konsistensi analisis berbasis waktu
    - Data diagregasi per hari untuk mengurangi noise dan mempermudah analisis tren
    - Hasil agregasi digunakan sebagai input utama dalam model forecasting
    """)

# =========================
# TAB 2: OVERVIEW
# =========================
with tab2:
    st.subheader("📊 Business Overview")

    # =========================
    # SALES TREND
    # =========================
    st.markdown("## 📈 Sales Trend")

    sales_time = df.groupby("order_date")["sales"].sum().reset_index()

    fig = px.line(sales_time, x="order_date", y="sales")
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # QUICK PERFORMANCE
    # =========================
    st.markdown("## ⚡ Quick Performance")

    col1, col2, col3 = st.columns(3)

    top_state = df.groupby("state")["sales"].sum().idxmax()
    top_category = df.groupby("category")["sales"].sum().idxmax()
    worst_state = df.groupby("state")["profit"].sum().idxmin()

    col1.success(f"🏆 Top State\n{top_state}")
    col2.info(f"📦 Top Category\n{top_category}")
    col3.error(f"📉 Worst Profit\n{worst_state}")

    # =========================
    # TOP STATE CHART
    # =========================
    st.markdown("## 🌍 Top 5 States by Sales")

    top_states = df.groupby("state")["sales"].sum().sort_values(ascending=False).head(5)

    fig_top = px.bar(
        top_states,
        x=top_states.values,
        y=top_states.index,
        orientation='h'
    )

    st.plotly_chart(fig_top, use_container_width=True)

    # =========================
    # CATEGORY CHART
    # =========================
    st.markdown("## 📦 Category Performance")

    cat = df.groupby("category")["sales"].sum()

    fig_cat = px.bar(cat, x=cat.index, y=cat.values)
    st.plotly_chart(fig_cat, use_container_width=True)

    # =========================
    # SUMMARY
    # =========================
    st.markdown("## 🧠 Executive Summary")

    st.write(f"""
    - {top_state} menjadi kontributor utama penjualan
    - Category {top_category} mendominasi performa bisnis
    - {worst_state} memiliki profit rendah dan perlu evaluasi
    """)

# =========================
# TAB 3: MAP
# =========================
with tab3:
    st.subheader("🗺️ Sales & Profit Map")

    # =========================
    # MAP
    # =========================
    kode_state = {
        'Alabama': 'AL','Alaska': 'AK','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO',
        'Connecticut': 'CT','Delaware': 'DE','Florida': 'FL','Georgia': 'GA','Hawaii': 'HI',
        'Idaho':'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY',
        'Louisiana':'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota':'MN',
        'Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH',
        'New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Ohio': 'OH',
        'Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Rhode Island': 'RI','South Carolina': 'SC',
        'South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virginia': 'VA',
        'Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'
    }

    df['kode_state'] = df['state'].map(kode_state)

    state_map = df.groupby('kode_state').agg({
        'sales':'sum',
        'profit':'sum',
        'quantity':'sum'
    }).reset_index()

    fig = go.Figure(data=go.Choropleth(
        locations=state_map['kode_state'],
        z=state_map['profit'],
        locationmode='USA-states',
        colorscale='RdYlGn',
        colorbar_title="Profit",
        hovertemplate=
        "<b>State:</b> %{location}<br>" +
        "Sales: %{customdata[0]:,.0f}<br>" +
        "Profit: %{customdata[1]:,.0f}<br>" +
        "Quantity: %{customdata[2]:,.0f}<extra></extra>",
        customdata=state_map[['sales','profit','quantity']].values
    ))

    fig.update_layout(geo_scope='usa')

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # SALES ANALYSIS
    # =========================
    st.markdown("## 📊 Top & Bottom States (Sales)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟢 Top 10 Sales")

        top_sales = df.groupby("state")["sales"].sum().sort_values(ascending=False).head(10)

        fig_top = px.bar(
            top_sales,
            x=top_sales.values,
            y=top_sales.index,
            orientation='h'
        )

        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.markdown("### 🔴 Bottom 10 Sales")

        low_sales = df.groupby("state")["sales"].sum().sort_values().head(10)

        fig_low = px.bar(
            low_sales,
            x=low_sales.values,
            y=low_sales.index,
            orientation='h'
        )

        st.plotly_chart(fig_low, use_container_width=True)

    # =========================
    # QUANTITY ANALYSIS
    # =========================
    st.markdown("## 📦 Top & Bottom States (Quantity)")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🟢 Top 10 Quantity")

        top_qty = df.groupby("state")["quantity"].sum().sort_values(ascending=False).head(10)

        fig_top_q = px.bar(
            top_qty,
            x=top_qty.values,
            y=top_qty.index,
            orientation='h'
        )

        st.plotly_chart(fig_top_q, use_container_width=True)

    with col4:
        st.markdown("### 🔴 Bottom 10 Quantity")

        low_qty = df.groupby("state")["quantity"].sum().sort_values().head(10)

        fig_low_q = px.bar(
            low_qty,
            x=low_qty.values,
            y=low_qty.index,
            orientation='h'
        )

        st.plotly_chart(fig_low_q, use_container_width=True)

    # =========================
    # INSIGHT (UPDATED 🔥)
    # =========================
    st.markdown("## 📌 Insight Wilayah")

    # SALES
    top_sales_state = df.groupby("state")["sales"].sum().idxmax()
    low_sales_state = df.groupby("state")["sales"].sum().idxmin()

    # PROFIT
    top_profit_state = df.groupby("state")["profit"].sum().idxmax()
    low_profit_state = df.groupby("state")["profit"].sum().idxmin()

    # QUANTITY
    top_qty_state = df.groupby("state")["quantity"].sum().idxmax()

    # DISPLAY
    st.success(f"🏆 Sales tertinggi: {top_sales_state}")
    st.warning(f"⚠️ Sales terendah: {low_sales_state}")

    st.success(f"💰 Profit tertinggi: {top_profit_state}")
    st.error(f"📉 Profit terendah (rugi): {low_profit_state}")

    st.info(f"📦 Quantity tertinggi: {top_qty_state}")

    # OPTIONAL SMART INSIGHT
    if top_sales_state == low_profit_state:
        st.warning("⚠️ State dengan sales tinggi tetapi profit rendah → indikasi diskon terlalu besar")


# =========================
# TAB 4: CATEGORY
# =========================
with tab4:
    st.subheader("📦 Category & Sub-Category Analysis")

    # =========================
    # CATEGORY ANALYSIS
    # =========================
    st.markdown("## 📊 Category Performance")

    df_category = df.groupby("category")[["quantity", "sales", "profit"]].sum().reset_index()

    col1, col2, col3 = st.columns(3)

    # SALES
    with col1:
        fig_sales = px.bar(
            df_category,
            x="category",
            y="sales",
            title="Total Sales per Category"
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    # QUANTITY
    with col2:
        fig_qty = px.bar(
            df_category,
            x="category",
            y="quantity",
            title="Total Quantity per Category"
        )
        st.plotly_chart(fig_qty, use_container_width=True)

    # PROFIT
    with col3:
        fig_profit = px.bar(
            df_category,
            x="category",
            y="profit",
            title="Total Profit per Category",
            color="profit"
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    # =========================
    # SUB-CATEGORY ANALYSIS
    # =========================
    st.markdown("## 📦 Sub-Category Performance")

    df_sub = df.groupby("sub-category")[["quantity", "sales", "profit"]].sum().reset_index()

    # TOP & BOTTOM
    top_sub = df_sub.sort_values("sales", ascending=False).head(5)
    low_sub = df_sub.sort_values("sales").head(5)

    col4, col5 = st.columns(2)

    with col4:
        fig_top_sub = px.bar(
            top_sub,
            x="sales",
            y="sub-category",
            orientation='h',
            title="Top 5 Sub-Category (Sales)"
        )
        st.plotly_chart(fig_top_sub, use_container_width=True)

    with col5:
        fig_low_sub = px.bar(
            low_sub,
            x="sales",
            y="sub-category",
            orientation='h',
            title="Lowest 5 Sub-Category (Sales)"
        )
        st.plotly_chart(fig_low_sub, use_container_width=True)

    # =========================
    # DISCOUNT ANALYSIS
    # =========================
    st.markdown("## 💸 Average Discount per Sub-Category")

    df_discount = df.groupby("sub-category")["discount"].mean().reset_index()
    df_discount = df_discount.sort_values("discount", ascending=False)

    fig_discount = px.bar(
        df_discount,
        x="discount",
        y="sub-category",
        orientation='h',
        title="Average Discount per Sub-Category"
    )

    st.plotly_chart(fig_discount, use_container_width=True)

    # =========================
    # INSIGHT (DATA-DRIVEN)
    # =========================
    st.markdown("## 📌 Insight Category & Sub-Category")

    # Category insight
    best_cat_sales = df_category.sort_values("sales", ascending=False).iloc[0]["category"]
    best_cat_profit = df_category.sort_values("profit", ascending=False).iloc[0]["category"]
    worst_cat_profit = df_category.sort_values("profit").iloc[0]["category"]

    # Sub-category insight
    best_sub_sales = df_sub.sort_values("sales", ascending=False).iloc[0]["sub-category"]
    best_sub_profit = df_sub.sort_values("profit", ascending=False).iloc[0]["sub-category"]
    worst_sub_profit = df_sub.sort_values("profit").iloc[0]["sub-category"]

    # Discount insight
    high_discount = df_discount.iloc[0]["sub-category"]

    # DISPLAY
    st.success(f"🏆 Category paling laris: {best_cat_sales}")
    st.success(f"💰 Category paling menguntungkan: {best_cat_profit}")
    st.warning(f"⚠️ Category profit terendah: {worst_cat_profit}")

    st.info(f"📦 Sub-category paling laris: {best_sub_sales}")
    st.success(f"💰 Sub-category paling menguntungkan: {best_sub_profit}")
    st.error(f"📉 Sub-category paling merugi: {worst_sub_profit}")

    st.warning(f"💸 Diskon tertinggi: {high_discount}")

    # =========================
    # INSIGHT (BUSINESS / CONSULTING STYLE 🔥)
    # =========================
    st.markdown("## 💡 Business Insight")

    st.write("""
    - Produk seperti Phones dan Chairs menjadi kontributor utama revenue dan profit
    - Storage dan Binders menunjukkan performa stabil (baik dari sales dan quantity)
    - Produk seperti Supplies dan Labels memiliki profit rendah → perlu evaluasi strategi harga atau diskon
    - Diskon tinggi pada beberapa sub-category dapat menyebabkan penurunan profit
    """)

    # SMART WARNING
    if worst_sub_profit == high_discount:
        st.warning("⚠️ Sub-category dengan diskon tinggi juga mengalami kerugian → strategi diskon perlu diperbaiki")

# =========================
# TAB 5: DRILLDOWN
# =========================
with tab5:
    st.subheader("🔍 Drilldown Analysis by State")

    # =========================
    # SELECT STATE
    # =========================
    selected_state = st.selectbox("📍 Pilih State", sorted(df["state"].unique()))

    df_state = df[df["state"] == selected_state]

    st.markdown(f"## 📊 Analisis untuk {selected_state}")

    # =========================
    # KPI
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Sales", f"${df_state['sales'].sum():,.0f}")
    col2.metric("📈 Profit", f"${df_state['profit'].sum():,.0f}")
    col3.metric("📦 Quantity", f"{df_state['quantity'].sum():,}")

    # =========================
    # SUB-CATEGORY PERFORMANCE
    # =========================
    st.markdown("## 📦 Sub-Category Performance")

    subcat = df_state.groupby("sub-category")[["sales", "profit"]].sum().reset_index()
    subcat = subcat.sort_values("sales", ascending=False)

    fig_sub = px.bar(
        subcat.head(10),
        x="sales",
        y="sub-category",
        orientation='h',
        color="profit",
        title=f"Top Sub-Category in {selected_state}"
    )

    st.plotly_chart(fig_sub, use_container_width=True)

    # =========================
    # TOP PRODUCTS
    # =========================
    st.markdown("## 🛒 Top Products")

    top_products = df_state.groupby("product_name")["sales"].sum().sort_values(ascending=False).head(10)

    fig_prod = px.bar(
        top_products,
        x=top_products.values,
        y=top_products.index,
        orientation='h',
        title="Top 10 Products"
    )

    st.plotly_chart(fig_prod, use_container_width=True)

    # =========================
    # INSIGHT
    # =========================
    st.markdown("## 📌 Insight State")

    best_sub = subcat.iloc[0]["sub-category"]
    worst_sub = subcat.sort_values("profit").iloc[0]["sub-category"]

    st.success(f"🏆 Sub-category terbaik: {best_sub}")
    st.error(f"📉 Sub-category paling merugi: {worst_sub}")

    # =========================
    # BUSINESS INSIGHT
    # =========================
    st.markdown("## 💡 Business Insight")

    st.write(f"""
    - {selected_state} memiliki kontribusi signifikan terhadap performa bisnis
    - Sub-category {best_sub} menjadi kontributor utama revenue
    - Sub-category {worst_sub} perlu evaluasi strategi harga atau produk
    """)

    # =========================
    # RECOMMENDATION
    # =========================
    st.markdown("## 🎯 Recommendation")

    st.success("""
    - Fokuskan stok pada produk dengan performa tinggi
    - Evaluasi atau kurangi produk dengan profit rendah
    - Optimalkan strategi pemasaran untuk meningkatkan profit
    """)

with tab6:
    st.subheader("📈 Forecast & Business Insight")

    # =========================
    # MODEL
    # =========================
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_ts)

    # =========================
    # FORECAST
    # =========================
    period = 90
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    actual = df_ts.copy()
    forecast_only = forecast[forecast["ds"] > df_ts["ds"].max()]

    # =========================
    # ACTUAL
    # =========================
    st.markdown("## 📊 Actual Sales")
    fig_actual = px.line(actual, x="ds", y="y")
    st.plotly_chart(fig_actual, use_container_width=True)

    # =========================
    # FORECAST
    # =========================
    st.markdown("## 🔮 Forecast Sales")
    fig_forecast = px.line(forecast_only, x="ds", y="yhat")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # =========================
    # TREND
    # =========================
    last_actual = actual["y"].iloc[-1]
    future_pred = forecast_only["yhat"].iloc[-1]
    growth = ((future_pred - last_actual) / last_actual) * 100

    trend_text = "meningkat" if growth > 0 else "menurun"

    st.caption(f"""
    Prediksi menunjukkan tren {trend_text} sebesar {abs(growth):.2f}% 
    dalam 3 bulan ke depan.
    """)

    # =========================
    # AMBIL BEST MODEL
    # =========================
    best_model = st.session_state.get("best_model", None)

    if best_model is None:
        st.warning("Model belum dihitung. Silakan buka tab Modeling terlebih dahulu.")
        best_model = "Belum tersedia"

    # =========================
    # BUSINESS RECOMMENDATION
    # =========================
    st.markdown("## 🎯 Business Recommendation")

    st.write(f"""
        - Gunakan forecast untuk perencanaan inventory
        - Optimalkan strategi promosi pada periode kenaikan tren
        - Kurangi risiko overstock saat tren menurun
    """)

    # =========================
    # MODEL IMPACT
    # =========================
    st.markdown("## 💡 Model Impact")

    st.write("""
    Dengan menggunakan model forecasting:

    - Perusahaan dapat mengantisipasi perubahan demand
    - Pengambilan keputusan menjadi lebih berbasis data
    - Risiko operasional dapat diminimalkan
    """)


with tab7:
    st.subheader("🤖 Modeling & Evaluation")

    import numpy as np
    import pandas as pd
    import plotly.express as px
    from prophet import Prophet

    # =========================
    # MODEL EXPLANATION
    # =========================
    st.markdown("## 🤖 Model Overview")

    st.write("""
    Model utama yang digunakan adalah **Prophet**, yaitu model time series 
    yang mampu menangkap tren dan seasonality.

    Sebagai pembanding, digunakan **Moving Average (MA)** sebagai baseline sederhana.
    
    Evaluasi dilakukan menggunakan **Rolling Validation (Backtesting)** 
    untuk memastikan model stabil di berbagai periode waktu.
    """)

    # =========================
    # PARAMETER
    # =========================
    horizon = 30
    step = 30

    # =========================
    # METRIC FUNCTION
    # =========================
    def evaluate(y_true, y_pred):
        mae_p = np.mean(np.abs(y_true - y_pred))
        rmse_p = np.sqrt(np.mean((y_true - y_pred)**2))
        return mae_p, rmse_p

    # =========================
    # ROLLING VALIDATION
    # =========================
    st.markdown("## 🔁 Rolling Validation (Backtesting)")

    results = []

    for start in range(0, len(df_ts) - horizon, step):
        train = df_ts[:start + horizon]
        test = df_ts[start + horizon:start + horizon + step]

        if len(test) == 0:
            continue

        # =========================
        # PROPHET
        # =========================
        model = Prophet()
        model.fit(train)

        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)

        y_true = test["y"].values

        # 🔥 AMBIL HANYA FUTURE (BUKAN HISTORY)
        y_pred_prophet = forecast["yhat"].iloc[-len(test):].values

        # =========================
        # MOVING AVERAGE
        # =========================
        train_ma = train.copy()
        train_ma["ma"] = train_ma["y"].rolling(7).mean()

        # 🔥 HANDLE NAN (ini yang sering bikin issue!)
        last_ma = train_ma["ma"].dropna().iloc[-1]

        y_pred_ma = np.full(len(test), last_ma)

        # safety check
        if len(y_true) != len(y_pred_ma):
            st.error(f"Mismatch MA: {len(y_true)} vs {len(y_pred_ma)}")
            continue

        mae_ma, rmse_ma = evaluate(y_true, y_pred_ma)

        # Prophet
        if len(y_true) != len(y_pred_prophet):
            st.warning(f"Skip window {start} (Prophet mismatch)")
            continue

        mae_p, rmse_p = evaluate(y_true, y_pred_prophet)

        # Moving Average
        if len(y_true) != len(y_pred_ma):
            st.warning(f"Skip window {start} (MA mismatch)")
            continue

        # 🔥 append hanya kalau semua aman
        results.append({
            "window": start,
            "mae_prophet": mae_p,
            "rmse_prophet": rmse_p,
            "mae_ma": mae_ma,
            "rmse_ma": rmse_ma
        })

    # =========================
    # FINAL DATAFRAME
    # =========================
    df_res = pd.DataFrame(results)

    # =========================
    # METRICS
    # =========================
    st.markdown("## 📏 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Prophet")
        st.metric("MAE", f"{df_res['mae_prophet'].mean():.2f}")
        st.metric("RMSE", f"{df_res['rmse_prophet'].mean():.2f}")

    with col2:
        st.markdown("### Moving Average")
        st.metric("MAE (MA)", f"{df_res['mae_ma'].mean():.2f}")
        st.metric("RMSE (MA)", f"{df_res['rmse_ma'].mean():.2f}")

    # =========================
    # MODEL COMPARISON
    # =========================
    st.markdown("## ⚖️ Model Comparison")

    if df_res["rmse_prophet"].mean() < df_res["rmse_ma"].mean():
        st.success("🏆 Prophet memberikan performa lebih baik")
        best_model = "Prophet"
    else:
        st.warning("Moving Average lebih baik")
        best_model = "Moving Average"

    # =========================
    # VISUAL COMPARISON
    # =========================
    st.markdown("## 📊 Error Comparison per Window")

    fig = px.line(
        df_res,
        x="window",
        y=["rmse_prophet", "rmse_ma"],
        title="RMSE per Window"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # INSIGHT
    # =========================
    st.markdown("## 🧠 Model Insight")

    st.write(f"""
    Evaluasi menggunakan rolling validation menunjukkan bahwa model **{best_model}** 
    memiliki performa yang lebih baik secara konsisten.

    Prophet unggul dalam menangkap pola tren dan seasonality, 
    sedangkan Moving Average hanya berfungsi sebagai baseline sederhana.

    Hal ini menunjukkan bahwa penggunaan model time series yang lebih kompleks 
    memberikan hasil prediksi yang lebih akurat untuk data penjualan.
    """)

    # =========================
    # HYPERPARAMETER TUNING
    # =========================
    st.markdown("## ⚙️ Hyperparameter Tuning (Prophet)")

    params = [0.01, 0.1, 0.5]
    tuning = []

    for p in params:
        model = Prophet(changepoint_prior_scale=p)
        model.fit(df_ts)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        y_true = df_ts["y"].tail(30).values
        y_pred_prophet = forecast["yhat"].tail(30).values

        mae, rmse = evaluate(y_true, y_pred_prophet)

        tuning.append({"param": p, "rmse": rmse})

    df_tune = pd.DataFrame(tuning)

    best = df_tune.loc[df_tune["rmse"].idxmin()]

    st.dataframe(df_tune)
    st.success(f"Best changepoint_prior_scale: {best['param']}")