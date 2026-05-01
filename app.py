import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🗺️ Region",
    "📦 Category",
    "🔍 Drilldown",
    "📈 Forecast"   # NEW TAB
])

# =========================
# TAB 1: OVERVIEW
# =========================
with tab1:
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
# TAB 2: MAP
# =========================
with tab2:
  with tab2:
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
# TAB 3: CATEGORY
# =========================
with tab3:
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
# TAB 4: DRILLDOWN
# =========================
with tab4:
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

with tab5:
    st.subheader("📈 Sales Forecast & Seasonality Analysis")

    from prophet import Prophet

    # =========================
    # FIXED PERIOD (90 HARI)
    # =========================
    period = 90

    # =========================
    # PREPARE DATA
    # =========================
    df_ts = df.groupby("order_date")["sales"].sum().reset_index()
    df_ts = df_ts.rename(columns={"order_date": "ds", "sales": "y"})

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

    st.caption(f"""
    Data historis menunjukkan tren penjualan. 
    Nilai terakhir sebesar ${actual['y'].iloc[-1]:,.0f}.
    """)

    # =========================
    # FORECAST
    # =========================
    st.markdown("## 🔮 Forecast Sales")

    fig_forecast = px.line(forecast_only, x="ds", y="yhat")
    st.plotly_chart(fig_forecast, use_container_width=True)

    last_actual = actual["y"].iloc[-1]
    future_pred = forecast_only["yhat"].iloc[-1]
    growth = ((future_pred - last_actual) / last_actual) * 100

    trend_text = "meningkat" if growth > 0 else "menurun"

    st.caption(f"""
    Prediksi menunjukkan tren {trend_text} sebesar {abs(growth):.2f}% 
    dalam 3 bulan ke depan (estimasi ${future_pred:,.0f}).
    """)

    # =========================
    # COMBINED
    # =========================
    st.markdown("## 📈 Actual vs Forecast")

    fig_combined = px.line()

    fig_combined.add_scatter(
        x=actual["ds"], y=actual["y"],
        mode="lines", name="Actual",
        line=dict(color="blue")
    )

    fig_combined.add_scatter(
        x=forecast_only["ds"], y=forecast_only["yhat"],
        mode="lines", name="Forecast",
        line=dict(color="red", dash="dash")
    )

    fig_combined.add_scatter(
        x=forecast_only["ds"],
        y=forecast_only["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    )

    fig_combined.add_scatter(
        x=forecast_only["ds"],
        y=forecast_only["yhat_lower"],
        mode="lines",
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(width=0),
        name="Confidence Interval"
    )

    st.plotly_chart(fig_combined, use_container_width=True)

    st.caption("Area bayangan menunjukkan ketidakpastian prediksi.")

    # =========================
    # COMPONENT
    # =========================
    st.markdown("## 🔍 Trend & Seasonality")

    fig_comp = model.plot_components(forecast)
    st.pyplot(fig_comp)

    # =========================
    # WEEKLY ANALYSIS
    # =========================
    st.markdown("## 📅 Weekly Pattern")

    df_ts["day_name"] = df_ts["ds"].dt.day_name()

    weekly = df_ts.groupby("day_name")["y"].mean().reindex([
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
    ])

    fig_weekly = px.bar(
        weekly,
        x=weekly.index,
        y=weekly.values,
        color=weekly.values,
        color_continuous_scale=["red", "yellow", "green"]
    )

    st.plotly_chart(fig_weekly, use_container_width=True)

    peak_day = weekly.idxmax()
    low_day = weekly.idxmin()

    st.caption(f"Puncak penjualan: {peak_day} | Terendah: {low_day}")

    # =========================
    # MONTHLY ANALYSIS
    # =========================
    st.markdown("## 📆 Monthly Pattern")

    df_ts["month"] = df_ts["ds"].dt.month
    monthly = df_ts.groupby("month")["y"].mean()

    fig_month = px.line(monthly, x=monthly.index, y=monthly.values)
    st.plotly_chart(fig_month, use_container_width=True)

    peak_month = monthly.idxmax()

    st.caption(f"Bulan dengan performa tertinggi: {peak_month}")

    # =========================
    # 🔥 AUTO HIGHLIGHT
    # =========================
    st.markdown("## 🔥 Key Highlight")

    if growth > 10:
        st.success("🚀 Pertumbuhan signifikan terdeteksi (>10%)")
    elif growth < -10:
        st.error("📉 Penurunan signifikan terdeteksi")
    else:
        st.info("📊 Tren relatif stabil")

    if peak_day in ["Saturday", "Sunday"]:
        st.success("🛍️ Penjualan tinggi terjadi di akhir pekan")
    else:
        st.info("📅 Penjualan dominan di hari kerja")

    # =========================
    # 🧠 EXECUTIVE INSIGHT
    # =========================
    st.markdown("## 🧠 Executive Insight")

    st.write(f"""
    Model menunjukkan tren penjualan yang {trend_text} dengan perubahan sebesar {abs(growth):.2f}%.

    Pola musiman terdeteksi:
    - Hari terbaik: {peak_day}
    - Bulan terbaik: {peak_month}

    Insight ini menunjukkan adanya pola perilaku pelanggan yang dapat dimanfaatkan
    untuk strategi pemasaran dan pengelolaan stok.
    """)

    # =========================
    # RECOMMENDATION
    # =========================
    st.markdown("## 🎯 Recommendation")

    st.write("""
    - Fokuskan promosi pada hari dengan performa tinggi
    - Optimalkan inventory berdasarkan tren permintaan
    - Gunakan forecast sebagai dasar perencanaan bisnis
    """)

    st.warning("⚠️ Forecast digunakan untuk melihat tren, bukan angka pasti")