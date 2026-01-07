import pandas as pd
import streamlit as st
from constantes import *
import plotly.express as px
import numpy as np

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = px.colors.qualitative.Set2



def config_page():
    """Configura los parámetros básicos de la página de Streamlit.

    Define el título, el icono y el layout general de la app.
    """
    st.set_page_config(
        page_title="Dashboard Ventas",
        page_icon="📊",
        layout="wide"
    )


# load_data: carga + limpieza (cache)
@st.cache_data
def load_data(path1: str, path2: str) -> pd.DataFrame:
    """Carga, concatena y realiza una limpieza mínima de los datos.

    Lee dos ficheros CSV con tipos de datos definidos (DTYPES) y parsea la columna
    `date` como fecha. Después concatena ambos DataFrames y elimina columnas
    residuales típicas de exportación (por ejemplo, `Unnamed: 0`).

    Args:
        path1: Ruta al primer CSV.
        path2: Ruta al segundo CSV.

    Returns:
        Un DataFrame con ambos CSV concatenados y una limpieza mínima aplicada.
    """
    # Lectura de CSV con tipado consistente y parseo de fechas para facilitar
    # agrupaciones temporales (year, month, week, etc.).
    df1 = pd.read_parquet(path1)
    df2 = pd.read_parquet(path2)
    # Concatena preservando el índice consecutivo.
    df = pd.concat([df1, df2], ignore_index=True)
    # Aplicar tipos columna por columna
    for col, dtype in DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    df["date"] = pd.to_datetime(df["date"])
    # Limpieza mínima: elimina columna residual si existe.
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

    return df


def header(df: pd.DataFrame | None):
    """Pinta la cabecera ejecutiva del dashboard.

    Muestra el título, una breve descripción y, si hay datos, el rango de fechas
    disponible en el dataset.

    Args:
        df: DataFrame principal de ventas (o None si no se ha cargado).
    """
    # “Portada” ejecutiva
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown("## 📊 Dashboard de Ventas")
        st.caption("Visión ejecutiva • KPIs y tendencias • Streamlit")

    with right:
        # Manejo defensivo: dataset ausente o vacío.
        if df is None or df.empty:
            st.error("Sin datos")
        else:
            # Rango de fechas como badge informativo para contexto temporal.
            fmin = df["date"].min().date()
            fmax = df["date"].max().date()
            st.success("Datos cargados")
            st.caption(f"Rango: {fmin} → {fmax}")

    st.divider()


def sidebar_controls(df: pd.DataFrame | None):
    """Construye la barra lateral con navegación y resumen del dataset.

    Incluye:
    - Radio de navegación por secciones.
    - Resumen del estado de carga del dataset.
    - Créditos/footer.

    Args:
        df: DataFrame principal (o None).

    Returns:
        La sección seleccionada en el control de radio.
    """
    with st.sidebar:
        st.title("⚙️ Panel de control")
        st.divider()

        # Navegación: radio como selector de sección principal.
        pagina = st.radio(
            "Sección",
            ["🌍 Global", "🏬 Tienda", "🗺️ Estado", "✨ Extra"],
            label_visibility="collapsed"
        )

        st.divider()

        # Resumen rápido del dataset para UX (estado de carga + tamaño).
        with st.expander("🧾 Resumen del dataset", expanded=True):
            if df is None or df.empty:
                st.write("- Dataset: **no cargado**")
            else:
                st.write("- Dataset: **cargado** ✅")
                st.write(f"- Filas: **{len(df):,}**")

        st.divider()
        st.caption("© 2025 • Dashboard Ventas")

    return pagina


# ---------- Cálculos ----------
def calc_kpis(df: pd.DataFrame) -> dict:
    """Calcula KPIs generales del dataset.

    Args:
        df: DataFrame de ventas con columnas de tienda, familia, estado y fecha.

    Returns:
        Diccionario con KPIs: número de tiendas, productos, estados y meses únicos.
    """
    # KPIs basados en cardinalidad para visión ejecutiva.
    return {
        "tiendas": df["store_nbr"].nunique(),
        "productos": df["family"].nunique(),
        "estados": df["state"].nunique(),
        "meses": df["date"].dt.to_period("M").nunique(),
    }


def top_10_products(df: pd.DataFrame) -> pd.DataFrame:
    """Obtiene el top 10 de familias de productos por volumen de ventas.

    Args:
        df: DataFrame principal.

    Returns:
        DataFrame con columnas `family` y `sales` para las 10 familias con más ventas.
    """
    # Groupby por familia; se suma ventas y se ordena desc para quedarse con el top.
    return (
        df.groupby("family", observed=True)["sales"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )


def sales_by_store(df: pd.DataFrame) -> pd.Series:
    """Agrega el volumen total de ventas por tienda.

    Args:
        df: DataFrame principal.

    Returns:
        Serie indexada por `store_nbr` con las ventas totales ordenadas de mayor a menor.
    """
    # Serie ordenada desc: útil para rankings y análisis de concentración.
    return (
        df.groupby("store_nbr", observed=True)["sales"]
          .sum()
          .sort_values(ascending=False)
    )


def top_10_promo_stores(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las 10 tiendas con más ventas realizadas en promoción.

    Considera únicamente filas con `onpromotion` > 0.

    Args:
        df: DataFrame principal.

    Returns:
        DataFrame con `store_nbr` (como string para Plotly) y `sales`.
    """
    # Filtra a filas con promoción activa y agrega ventas por tienda.
    d = (
        df.loc[df["onpromotion"] > 0]
          .groupby("store_nbr", observed=True)["sales"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )
    # Plotly trata categorías mejor si store_nbr es string (evita escala numérica).
    d["store_nbr"] = d["store_nbr"].astype(str)  # importante para Plotly
    return d


def avg_sales_by_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ventas medias por día de la semana, respetando un orden fijo.

    Args:
        df: DataFrame principal con columna `day_of_week`.

    Returns:
        DataFrame con `day_of_week` y `sales` (media), reindexado según orden semanal.
    """
    # Orden explícito para no depender del orden alfabético.
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return (
        df.groupby("day_of_week", observed=True)["sales"]
          .mean()
          .reindex(order)
          .reset_index()
    )


def avg_sales_by_week(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ventas medias por semana del año.

    Args:
        df: DataFrame principal con columna `week`.

    Returns:
        DataFrame con `week` y `sales` (media), ordenado por semana.
    """
    # Media por semana; se ordena para que la serie temporal sea coherente.
    return (
        df.groupby("week", observed=True)["sales"]
          .mean()
          .reset_index()
          .sort_values("week")
    )


def avg_sales_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ventas medias por mes del año.

    Args:
        df: DataFrame principal con columna `month`.

    Returns:
        DataFrame con `month` y `sales` (media), ordenado por mes.
    """
    # Media por mes; se ordena para lectura temporal correcta.
    return (
        df.groupby("month", observed=True)["sales"]
          .mean()
          .reset_index()
          .sort_values("month")
    )


# ---------- UI: subsecciones ----------
def section_kpis(df: pd.DataFrame):
    """Renderiza la sección de KPIs globales.

    Args:
        df: DataFrame principal.
    """
    # Contenedor con borde para separar visualmente secciones.
    with st.container(border=True):
        st.subheader("📌 Indicadores generales")
        k = calc_kpis(df)

        # Distribución en cuatro columnas para KPIs principales.
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏬 Tiendas", k["tiendas"])
        c2.metric("📦 Productos", k["productos"])
        c3.metric("🗺️ Estados", k["estados"])
        c4.metric("📅 Meses con datos", k["meses"])


def section_rankings(df: pd.DataFrame):
    """Renderiza la sección de rankings (productos y tiendas) y distribución por tienda.

    Args:
        df: DataFrame principal.
    """
    # Dos columnas: producto top y tiendas promo top.
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🏆 Top 10 productos (volumen de ventas)")
            tp = top_10_products(df)
            fig = px.bar(tp, x="sales", y="family", orientation="h")
            # Invertir el eje y para que el mayor aparezca arriba.
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")

    with col2:
        with st.container(border=True):
            st.subheader("🏷️ Top 10 tiendas (ventas en promoción)")
            promo = top_10_promo_stores(df)
            fig = px.bar(promo, x="sales", y="store_nbr", orientation="h")
            # Forzar categorías y ordenar para ranking visual correcto.
            fig.update_yaxes(type="category", autorange="reversed")
            st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.subheader("🏬 Distribución de ventas por tienda")
        ventas_tienda = sales_by_store(df)

        # Columna izquierda: gráfico; derecha: métricas de concentración.
        c1, c2 = st.columns([2, 1])
        with c1:
            # Control interactivo para ajustar número de tiendas en el gráfico.
            top_n = st.slider("Top tiendas a mostrar", 10, 54, 30, step=5)

            # Preparación del DF para Plotly.
            hist_df = ventas_tienda.head(top_n).reset_index()
            hist_df.columns = ["store_nbr", "sales"]
            # Categoría (string) para que Plotly no haga escala numérica continua.
            hist_df["store_nbr"] = hist_df["store_nbr"].astype(str)

            fig = px.bar(
                hist_df,
                x="store_nbr",
                y="sales",
                title=f"Ventas por tienda (Top {top_n})",
                labels={"store_nbr": "Tienda #", "sales": "Ventas totales"},
                height=420
            )

            # Asegura el orden numérico correcto aunque sea categoría.
            orden_correcto = sorted(hist_df["store_nbr"].astype(int).unique())
            orden_correcto_str = [str(x) for x in orden_correcto]

            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=orden_correcto_str
            )

            # Plantilla de hover más legible para negocio.
            fig.update_traces(hovertemplate="Tienda %{x}<br>Ventas: %{y:,.2f}")
            st.plotly_chart(fig, width="stretch")

        with c2:
            # Métricas de concentración para entender desigualdad entre tiendas.
            total = float(ventas_tienda.sum())
            top3 = float(ventas_tienda.head(3).sum()) / total * 100
            top10 = float(ventas_tienda.head(10).sum()) / total * 100
            st.metric("Concentración Top 3", f"{top3:.1f}%")
            st.metric("Concentración Top 10", f"{top10:.1f}%")
            st.caption("Porcentaje del volumen total de ventas.")


def section_seasonality(df: pd.DataFrame):
    """Renderiza gráficos de estacionalidad: día de semana, semana del año y mes.

    Args:
        df: DataFrame principal.
    """
    with st.container(border=True):
        st.subheader("📅 Ventas medias por día de la semana")
        d = avg_sales_by_day_of_week(df)
        fig = px.bar(
            d,
            x="day_of_week",
            y="sales",
            labels={"day_of_week": "Día", "sales": "Ventas medias"}
        )
        st.plotly_chart(fig, width="stretch")
        # Insight rápido: día con mayor media.
        st.caption(f"📌 Día con mayor volumen medio: **{d.loc[d['sales'].idxmax(), 'day_of_week']}**")

    with st.container(border=True):
        st.subheader("📈 Ventas medias por semana del año")
        w = avg_sales_by_week(df)
        fig = px.line(
            w,
            x="week",
            y="sales",
            markers=True,
            labels={"week": "Semana", "sales": "Ventas medias"}
        )
        st.plotly_chart(fig, width="stretch")
        # Insight: semana con máxima media.
        st.caption(f"📌 Semana con mayor volumen medio: **{int(w.loc[w['sales'].idxmax(), 'week'])}**")

    with st.container(border=True):
        st.subheader("📈 Ventas medias por mes del año")
        m = avg_sales_by_month(df)
        fig = px.line(
            m,
            x="month",
            y="sales",
            markers=True,
            labels={"month": "Mes", "sales": "Ventas medias"}
        )
        st.plotly_chart(fig, width="stretch")
        # Insight: mes con máxima media.
        st.caption(f"📌 Mes con mayor volumen medio: **{int(m.loc[m['sales'].idxmax(), 'month'])}**")


# ---------- TAB GLOBAL con tabs internos ----------
def tab_global(df: pd.DataFrame):
    """Pestaña principal: visión global con tabs internos (KPIs, rankings y estacionalidad).

    Args:
        df: DataFrame principal.
    """
    st.header("🌍 Visión global")

    # Tabs internos para evitar scroll excesivo y ordenar la narrativa.
    t_kpi, t_rank, t_season = st.tabs(["📌 KPIs", "🏆 Rankings", "🧭 Estacionalidad"])

    with t_kpi:
        section_kpis(df)

    with t_rank:
        section_rankings(df)

    with t_season:
        section_seasonality(df)


def tab_store(df: pd.DataFrame):
    """Pestaña de análisis por tienda (detalle y métricas por store_nbr).

    Nota:
        Dentro del código original hay imports/def duplicados y una indentación
        inconsistente. No se corrige aquí porque se pidió no cambiar el código.

    Args:
        df: DataFrame principal.
    """
    st.header("🏬 Pestaña 2 · Análisis por tienda")

    # 1) Selector de tienda
    tiendas = sorted(df["store_nbr"].unique())
    store = st.selectbox("Selecciona la tienda (store_nbr)", tiendas)

    # Filtrar datos
    d = df.loc[df["store_nbr"] == store].copy()

    # ---------------------------------------------------------
    # 2) KPIs (Ahora integran la lógica avanzada de ventas > 0)
    # ---------------------------------------------------------

    # KPI 1: Ventas totales
    total_ventas = float(d["sales"].sum())

    # KPI 2: Familias con ventas reales (> 0)
    # (Antes estaba en el container 'b' de abajo)
    vendidos = d.loc[d["sales"] > 0, "family"].nunique()

    # KPI 3: Familias en promoción con ventas reales
    # (Antes estaba en el container 'c' de abajo)
    vendidos_promo = d.loc[(d["sales"] > 0) & (d["onpromotion"] > 0), "family"].nunique()

    # Visualización de métricas
    k1, k2, k3 = st.columns(3)
    k1.metric("💰 Volumen total de ventas", f"{total_ventas:,.2f}")
    k2.metric("📦 Familias con ventas", int(vendidos), help="Familias de producto con ventas > 0")
    k3.metric("🏷️ Familias promo vendidas", int(vendidos_promo), help="Familias vendidas que estaban en promoción")

    st.divider()

    # ---------------------------------------------------------
    # 3) Gráfico de barras (Ventas por año)
    # ---------------------------------------------------------
    ventas_por_anyo = (
        d.groupby("year", observed=True)["sales"]
          .sum()
          .reset_index()
          .sort_values("year")
    )

    with st.container(border=True):
        st.subheader("📈 Ventas totales por año")

        fig = px.bar(
            ventas_por_anyo,
            x="year",
            y="sales",
            text_auto=".2s",
            labels={"year": "Año", "sales": "Ventas totales"},
            title=f"Tienda {store} · Ventas totales por año"
        )

        # Quitamos las líneas de la rejilla (grid) del eje Y
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig, width="stretch")

        # Insight rápido: mejor año en ventas para la tienda seleccionada.
        best_year = int(ventas_por_anyo.loc[ventas_por_anyo["sales"].idxmax(), "year"])
        st.caption(f"📌 Año con mayor volumen de ventas: **{best_year}**")


def tab_state(df: pd.DataFrame):
    """Pestaña de análisis por estado: transacciones y ranking de tiendas.

    Args:
        df: DataFrame principal.
    """
    st.header("📍 Pestaña 3 · Análisis por Estado")

    # ---------------------------------------------------------
    # 1) Selector de Estado
    # ---------------------------------------------------------
    estados = sorted(df["state"].unique())
    selected_state = st.selectbox("Selecciona el estado a analizar", estados)

    # Filtramos el dataframe por el estado seleccionado
    d_state = df.loc[df["state"] == selected_state].copy()

    st.divider()

    # ---------------------------------------------------------
    # c) Producto más vendido en el estado (Lo ponemos primero como KPI)
    # ---------------------------------------------------------
    # Agrupamos por familia y sumamos ventas
    top_prods = d_state.groupby("family", observed=True)["sales"].sum().reset_index()

    # Encontramos el máximo con control de vacío
    if not top_prods.empty:
        best_row = top_prods.loc[top_prods["sales"].idxmax()]
        best_family = best_row["family"]
        best_sales = best_row["sales"]
    else:
        best_family = "N/A"
        best_sales = 0

    col_kpi, col_dummy = st.columns([1, 2])
    with col_kpi:
        st.subheader("🏆 Producto estrella")
        st.metric(
            label="Producto más vendido",
            value=best_family,
            delta=f"{best_sales:,.0f} ventas totales",
            delta_color="off"  # Delta gris, sin semántica verde/rojo.
        )

    st.divider()

    # Layout de dos columnas para los gráficos
    col1, col2 = st.columns(2)

    # ---------------------------------------------------------
    # a) Número total de transacciones por año
    # ---------------------------------------------------------
    with col1:
        st.subheader("💳 Transacciones por año")

        # Agregación anual de transacciones para el estado.
        trans_por_anyo = (
            d_state.groupby("year", observed=True)["transactions"]
            .sum()
            .reset_index()
            .sort_values("year")
        )

        fig_trans = px.bar(
            trans_por_anyo,
            x="year",
            y="transactions",
            text_auto=".2s",
            title=f"Transacciones en {selected_state}",
            labels={"year": "Año", "transactions": "Total Transacciones"}
        )

        # Estilo limpio (sin rejilla Y)
        fig_trans.update_yaxes(showgrid=False)
        st.plotly_chart(fig_trans, width="stretch")

    # ---------------------------------------------------------
    # b) Ranking de tiendas con más ventas
    # ---------------------------------------------------------
    with col2:
        st.subheader("🏪 Ranking de tiendas")

        # Ranking por tienda (store_nbr) sumando ventas.
        ranking_tiendas = (
            d_state.groupby("store_nbr", observed=True)["sales"]
            .sum()
            .reset_index()
            # Ascendente para que el mayor quede visualmente arriba en horizontal.
            .sort_values("sales", ascending=True)
        )

        # Convertimos store_nbr a string para categoría.
        ranking_tiendas["store_nbr"] = ranking_tiendas["store_nbr"].astype(str)

        fig_rank = px.bar(
            ranking_tiendas,
            x="sales",
            y="store_nbr",
            text_auto=".2s",
            orientation='h',  # Gráfico horizontal para rankings.
            title=f"Ventas por Tienda en {selected_state}",
            labels={"sales": "Ventas Totales", "store_nbr": "Tienda #"}
        )

        # Estilo limpio: quitar rejillas.
        fig_rank.update_xaxes(showgrid=False)
        fig_rank.update_yaxes(showgrid=False)
        fig_rank.update_yaxes(type="category", autorange="reversed")

        st.plotly_chart(fig_rank, width="stretch")


def generar_coordenadas_por_ciudad_con_jitter(df_input: pd.DataFrame) -> pd.DataFrame:
    """Asigna coordenadas (lat/lon) por tienda usando centro por ciudad + jitter.

    La intención es evitar superposición de puntos cuando varias tiendas comparten
    la misma ciudad: se añade un pequeño ruido pseudo-aleatorio por tienda.

    Args:
        df_input: DataFrame con, al menos, `store_nbr` y `city`.

    Returns:
        Copia del DataFrame con columnas `latitude` y `longitude` añadidas.
    """
    # Copia defensiva para no mutar el DF original.
    df_coords = df_input.copy()

    # Se calcula una coordenada por tienda.
    unique_stores = df_coords['store_nbr'].unique()
    coords_map = {}

    # Magnitud del desplazamiento (grados aprox). Ajustar con cuidado: demasiado
    # alto "descoloca" puntos; demasiado bajo sigue solapando.
    jitter_amount = 0.025

    for store_nbr in unique_stores:
        # Obtener ciudad asociada a la tienda (asume coherencia 1 tienda -> 1 ciudad).
        city_name = df_coords.loc[df_coords['store_nbr'] == store_nbr, 'city'].iloc[0]

        # Centro base por ciudad (fallback a DEFAULT si no está en el diccionario).
        center_lat, center_lon = COORDENADAS_CIUDADES.get(
            city_name,
            COORDENADAS_CIUDADES["DEFAULT"]
        )

        # Semilla derivada de store_nbr para jitter determinista (reproducible).
        seed_val = int(store_nbr) * 99
        np.random.seed(seed_val)
        lat_offset = np.random.uniform(-jitter_amount, jitter_amount)

        # Variar semilla para no correlacionar offsets de lat y lon.
        np.random.seed(seed_val + 5)
        lon_offset = np.random.uniform(-jitter_amount, jitter_amount)

        coords_map[store_nbr] = (center_lat + lat_offset, center_lon + lon_offset)

    # Map de coordenadas calculadas a cada fila del DF.
    df_coords['latitude'] = df_coords['store_nbr'].map(lambda x: coords_map[x][0])
    df_coords['longitude'] = df_coords['store_nbr'].map(lambda x: coords_map[x][1])
    return df_coords


def tab_bonus(df: pd.DataFrame):
    """Pestaña bonus con análisis avanzado: mapa y eficiencia de promociones.

    Args:
        df: DataFrame principal.
    """
    st.header("💡 Pestaña Bonus · Análisis Avanzado")

    tab1, tab2 = st.tabs(["🗺️ Mapa Geográfico", "🏷️ Eficiencia de Promociones"])

    # ==============================================================================
    # TAB 1: EL MAPA
    # ==============================================================================
    with tab1:
        st.subheader("📍 Mapa de Calor de Ventas")

        # Estadísticos por ciudad: tiendas únicas y ventas totales.
        city_stats = (
            df.groupby("city", observed=False)
            .agg({"store_nbr": "nunique", "sales": "sum"})
            .reset_index()
            .sort_values("sales", ascending=False)
        )
        top_city = city_stats.iloc[0]

        # KPIs rápidos sobre ciudad líder en ventas.
        k1, k2, k3 = st.columns(3)
        k1.metric("🏆 Ciudad Top Ventas", top_city['city'])
        k2.metric("💰 Ventas Totales", f"{top_city['sales']:,.0f}")
        k3.metric("🏪 Nº Tiendas", int(top_city['store_nbr']))

        # Tabla desplegable con detalle completo por ciudad.
        with st.expander("🔽 Ver detalle de todas las ciudades"):
            st.dataframe(
                city_stats,
                column_config={
                    "city": "Ciudad",
                    "sales": st.column_config.ProgressColumn(
                        "Ventas",
                        format="%.2f",
                        max_value=float(city_stats["sales"].max())
                    ),
                    "store_nbr": "Nº Tiendas"
                },
                hide_index=True,
                width="stretch"
            )

        # Enriquecer el DF con coordenadas para poder pintar el mapa.
        df_with_coords = generar_coordenadas_por_ciudad_con_jitter(df)

        # Data a nivel tienda (1 fila por store) para el mapa.
        map_data = (
            df_with_coords.groupby("store_nbr", observed=False)
            .agg({
                "sales": "sum",
                "state": "first",
                "city": "first",
                "latitude": "first",
                "longitude": "first"
            })
            .reset_index()
        )

        # Mapa scatter con tamaño proporcional a ventas y color por estado.
        fig_map = px.scatter_map(
            map_data,
            lat="latitude",
            lon="longitude",
            size="sales",
            color="state",
            hover_name="store_nbr",
            hover_data={
                "latitude": False,
                "longitude": False,
                "city": True,
                "state": True,
                "sales": ":,.0f"
            },
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            size_max=40,
            zoom=6,
            center={"lat": -1.5, "lon": -78.5},
            map_style="carto-darkmatter",
        )

        # Ajustes de layout (márgenes + leyenda) para encajar en wide layout.
        fig_map.update_layout(
            margin={"r": 150, "t": 0, "l": 0, "b": 0},
            height=650,
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0.5)",
                title_text=""
            )
        )
        st.plotly_chart(fig_map, width="stretch")

    # ==============================================================================
    # TAB 2: TIENDAS CON VENTAS PROMO > 0 (FILTRADO Y ORDENADO)
    # ==============================================================================
    with tab2:
        st.subheader("📈 Ranking de Dependencia de Ofertas")
        st.caption(
            "Muestra solo las tiendas que han registrado ventas en promoción, "
            "ordenadas de mayor a menor impacto."
        )

        # 1) Dimensión tienda: 1 fila por store_nbr (para no duplicar tiendas)
        store_dim = df[['store_nbr', 'city', 'state']].copy()
        store_dim['store_nbr'] = store_dim['store_nbr'].astype(int)

        # Nos quedamos con una city/state por tienda (la más frecuente)
        store_dim = (
            store_dim.groupby('store_nbr', observed=False)
            .agg(
                city=('city', lambda s: s.dropna().mode().iloc[0]
                      if not s.dropna().mode().empty else None),
                state=('state', lambda s: s.dropna().mode().iloc[0]
                       if not s.dropna().mode().empty else None),
            )
            .reset_index()
        )

        # 2) Ventas totales: 1 fila por tienda (IMPORTANTE)
        total_sales = (
            df.groupby('store_nbr', observed=False)['sales']
            .sum()
            .reset_index()
            .rename(columns={'sales': 'total_vol'})
        )
        total_sales['store_nbr'] = total_sales['store_nbr'].astype(int)

        # 3) Ventas en promoción: 1 fila por tienda
        promo_sales = (
            df[df['onpromotion'] > 0]
            .groupby('store_nbr', observed=False)['sales']
            .sum()
            .reset_index()
            .rename(columns={'sales': 'promo_vol'})
        )
        promo_sales['store_nbr'] = promo_sales['store_nbr'].astype(int)

        # 4) Merge (sin multiplicar tiendas)
        merged = (
            total_sales
            .merge(promo_sales, on='store_nbr', how='left')
            .merge(store_dim, on='store_nbr', how='left')
        )
        merged['promo_vol'] = merged['promo_vol'].fillna(0)

        # 5) % promo (evitar inf / div0)
        merged = merged[merged['total_vol'] > 0].copy()
        merged['promo_pct'] = (merged['promo_vol'] / merged['total_vol']) * 100
        merged['promo_pct'] = merged['promo_pct'].replace([np.inf, -np.inf], 0).fillna(0)

        # Etiqueta (sin meter categorías nuevas)
        city_str = merged['city'].astype("object").fillna("").astype(str)
        merged['store_label'] = "Tienda " + merged['store_nbr'].astype(str) + " (" + city_str + ")"

        # --- FILTRO: NO mostrar tiendas con 0 promo ---
        tiendas_total = merged['store_nbr'].nunique()
        merged_filtered = merged[merged['promo_vol'] > 0].copy()  # (más fiable que promo_pct>0)
        tiendas_con_ventas = merged_filtered['store_nbr'].nunique()
        tiendas_cero = tiendas_total - tiendas_con_ventas

        # Orden ascendente para que en bar horizontal la mayor quede arriba (última fila)
        merged_filtered = merged_filtered.sort_values('promo_pct', ascending=True)

        # 6) Gráfico
        if not merged_filtered.empty:
            # Altura dinámica para legibilidad en rankings largos.
            altura_grafico = 200 + (len(merged_filtered) * 25)

            fig_promo = px.bar(
                merged_filtered,
                x="promo_pct",
                y="store_label",
                orientation='h',
                text_auto='.1f',
                title=f"Tiendas con ventas en promoción ({tiendas_con_ventas} tiendas)",
                labels={"promo_pct": "% Ventas Promo", "store_label": ""},
                color="promo_pct",
                color_continuous_scale="Viridis"
            )

            fig_promo.update_layout(
                xaxis_title="% del Total de Ventas que fue Promo",
                showlegend=False,
                height=altura_grafico,
                bargap=0.1,
                yaxis={'type': 'category'}
            )
            fig_promo.update_traces(texttemplate='%{x:.1f}%', textposition='outside')

            st.plotly_chart(fig_promo, width="stretch")

            # Insights Top/Bottom de las mostradas
            best_store = merged_filtered.iloc[-1]
            worst_store = merged_filtered.iloc[0]

            c1, c2 = st.columns(2)
            with c1:
                st.success(
                    f"🥇 **Mayor Impacto:** Tienda {best_store['store_nbr']} "
                    f"({best_store['city']}) con **{best_store['promo_pct']:.1f}%**"
                )
            with c2:
                st.info(
                    f"🥉 **Menor Impacto (de las activas):** Tienda {worst_store['store_nbr']} "
                    f"({worst_store['city']}) con **{worst_store['promo_pct']:.1f}%**"
                )

            st.divider()

            if tiendas_cero > 0:
                st.warning(
                    f"⚠️ **Nota:** Existen otras **{tiendas_cero} tiendas** que no aparecen en el gráfico "
                    f"porque **no registraron ventas promocionales** (0.0%)."
                )
        else:
            st.warning("Ninguna tienda tiene ventas registradas en promoción.")


def main():
    """Punto de entrada principal de la aplicación Streamlit.

    Orquesta:
    - Configuración de página.
    - Carga de datos.
    - Header + sidebar.
    - Navegación entre pestañas.
    - Footer.
    """
    config_page()

    # Carga del dataset (cacheada) desde las dos partes.
    df = load_data("parte_1.parquet", "parte_2.parquet")

    # Header ejecutivo + sidebar “pro”
    header(df)
    pagina = sidebar_controls(df)

    # Navegación: decide qué pestaña renderizar en función del radio.
    if pagina == "🌍 Global":
        tab_global(df)
    elif pagina == "🏬 Tienda":
        tab_store(df)
    elif pagina == "🗺️ Estado":
        tab_state(df)
    else:
        tab_bonus(df)

    st.divider()
    st.caption("Desarrollado con Streamlit • Dashboard Ventas • © 2025")


if __name__ == "__main__":
    # Ejecución directa del script.
    main()
