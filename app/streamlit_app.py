"""橋梁リスク可視化ダッシュボード（シンプル版 + 迂回路表示 + EDA閲覧）

地図上で、橋梁データを属性（クラスター・道路種別）で絞り込んだり、
特定の橋を選んで表示したりできる、Streamlitアプリです。
「迂回路ルートの表示」トグルと「EDA」タブを追加しています。
"""
import os
import tempfile

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from streamlit_folium import st_folium

st.set_page_config(page_title="橋梁リスク可視化ダッシュボード", layout="wide")

# 【備考】以前は matplotlib/seaborn でグラフを描画しており、日本語フォントの設定
# （japanize_matplotlibやフォントファイルの同梱）が必要だった。Plotlyはブラウザ側で
# 描画されるため、サーバー側のフォント環境に左右されず日本語もそのまま表示できる。

# ---------------------------------------------------------------------------
# クラスター・道路種別を分かりやすい日本語ラベルに変換するための対応表
# （03_eda_clustering.ipynb でのクラスター解釈まとめに基づく。データ更新に伴い
# クラスター番号ごとの意味が変わることがあるため、notebook側の解釈まとめと
# 合わせてこの対応表も更新してください）
# ---------------------------------------------------------------------------
CLUSTER_LABELS = {
    0: "0: 高交通需要",
    1: "1: 交通集中×長距離迂回",
    2: "2: 標準橋群（橋長20〜100m程度・幹線）",
    3: "3: 標準橋群（橋長20m程度）",
    4: "4: 老朽化進行",
    5: "5: 超長大橋",
}
CLUSTER_COLORS = {
    0: "#4C72B0", 1: "#DD8452", 2: "#55A868",
    3: "#C44E52", 4: "#8172B2", 5: "#937860",
}
HIGHWAY_LABELS = {
    1: "1: 主要道",
    2: "2: 二次主要道",
    3: "3: 一般道路",
    4: "4: 補助道路",
    5: "5: 生活道路",
}
QUADRANT_LABELS = {
    1: "左下：交通量少 × 迂回路短",
    2: "左上：交通量少 × 迂回路長",
    3: "右下：交通量多 × 迂回路短",
    4: "右上：交通量多 × 迂回路長（優先度高）",
}

# 地図上に迂回路ルートを描画する際、一度に表示する上限件数
# （多すぎると地図が見づらく、描画も重くなるための安全弁）
MAX_ROUTES_TO_DRAW = 60

DEFAULT_DATA_PATHS = [
    "../data/processed/final_analysis_gdf.gpkg",
    "data/processed/final_analysis_gdf.gpkg",
    "final_analysis_gdf.gpkg",
]


@st.cache_data(show_spinner="データを読み込んでいます...")
def load_data(file_source) -> pd.DataFrame:
    """GeoPackageを読み込み、緯度経度付きの通常のDataFrameとして返す。

    Args:
        file_source: ファイルパス（str）、またはst.file_uploaderから受け取ったファイルオブジェクト

    Returns:
        lat, lon列を持つDataFrame（クラスター等のラベル列も付与済み）
    """
    # GeoPackageはSQLiteベースの形式で、アップロードされたファイルオブジェクト(メモリ上のバイト列)を
    # 直接は読み込めないことがあるため、一時ファイルに保存してから読み込む
    if isinstance(file_source, str):
        gdf = gpd.read_file(file_source)
    else:
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
            tmp_file.write(file_source.getbuffer())
            tmp_path = tmp_file.name
        gdf = gpd.read_file(tmp_path)
        os.remove(tmp_path)

    gdf = gdf.to_crs(epsg=4326)  # 地図表示用に緯度経度(WGS84)へ変換
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x

    df = pd.DataFrame(gdf.drop(columns="geometry"))
    # GeoPackageによっては架設年度が文字列型で保存されていることがあるため、数値型に変換しておく
    # （グラフ描画・中央値計算でのエラーを防ぐため）
    df["dpf_kasetsu_nendo"] = pd.to_numeric(df["dpf_kasetsu_nendo"], errors="coerce")
    # 【注意】GeoPackage上でosm_onewayがBOOLEAN型として保存されていると、
    # 読み込み時にPythonのbool型(True/False)になることがある。bool型のまま引き算をすると
    # 「numpy boolean subtract」というTypeErrorになる（レーダーチャートの正規化で発生）ため、
    # ここで明示的にint型(0/1)へ変換しておく。
    df["osm_oneway"] = df["osm_oneway"].astype(int)
    # 【注意】迂回路が見つからなかった橋はdetour_length_mがinf(無限大)のまま
    # 保存されていることがある(notebook側で中央値補完がbridges_gdfに反映されていないケース)。
    # inf のままだとグラフの正規化や表示が壊れるため、ここで欠損(NaN)として扱う。
    df["detour_length_m"] = df["detour_length_m"].where(
        ~df["detour_length_m"].isin([float("inf"), float("-inf")])
    )
    df["cluster_label_ja"] = df["cluster_6_label"].map(CLUSTER_LABELS)
    df["highway_label_ja"] = df["osm_highway_aggregated"].map(HIGHWAY_LABELS)
    df["quadrant_label_ja"] = df["traffic_detour_quadrant"].map(QUADRANT_LABELS)
    df["oneway_label_ja"] = df["osm_oneway"].map({0: "一方通行ではない", 1: "一方通行"})
    return df


def find_default_data_path():
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            return path
    return None


def route_wkt_to_latlon_lines(route_wkt: str):
    """迂回路の経路WKT文字列を、folium.PolyLineに渡せる[(lat, lon), ...]のリストに変換する。

    LineString・MultiLineStringの両方に対応する（経路がedge単位で分割されたまま
    マージされずに保存されているケースを考慮）。

    Args:
        route_wkt: `02_detour_calculation.ipynb`で保存した経路のWKT文字列

    Returns:
        [(lat, lon), ...] のリストのリスト（MultiLineStringの場合は複数本になる）
    """
    if not route_wkt or not isinstance(route_wkt, str):
        return []
    try:
        geom = wkt.loads(route_wkt)
    except Exception:
        return []

    lines = []
    if isinstance(geom, LineString):
        # 保存時の座標順は(lon, lat)のため、foliumが期待する(lat, lon)に入れ替える
        lines.append([(lat, lon) for lon, lat in geom.coords])
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            lines.append([(lat, lon) for lon, lat in part.coords])
    return lines


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------
st.title("🌉 橋梁リスク可視化ダッシュボード")
st.markdown(
    "データサイエンスの個人学習記録です。オープンデータの東京23区の橋梁データをもとに、"
    "交通量・迂回路長などの観点から、補修・補強の優先度設定を試行した結果を可視化するダッシュボードです。\n\n"
    "GitHubはこちら： https://github.com/nobishun/bridge-risk-analysis\n\n"
    "※現在も作業中のプロジェクトです。"
)
st.caption("東京23区の橋梁データを、地図上で属性ごとに絞り込んで確認できます。")

default_path = find_default_data_path()

if default_path is not None:
    df = load_data(default_path)
else:
    st.warning(
        "既定の場所（`data/processed/final_analysis_gdf.gpkg`）にデータが見つかりませんでした。"
        " 下のボタンから `final_analysis_gdf.gpkg` をアップロードしてください。"
    )
    uploaded_file = st.file_uploader("final_analysis_gdf.gpkg をアップロード", type=["gpkg"])
    if uploaded_file is None:
        st.stop()
    df = load_data(uploaded_file)

has_route_data = "detour_route_wkt" in df.columns

# ---------------------------------------------------------------------------
# サイドバー：絞り込み条件
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 絞り込み条件")

selected_clusters = st.sidebar.multiselect(
    "クラスター（橋の特徴グループ）",
    options=sorted(df["cluster_6_label"].unique()),
    default=sorted(df["cluster_6_label"].unique()),
    format_func=lambda x: CLUSTER_LABELS.get(x, str(x)),
)

selected_highways = st.sidebar.multiselect(
    "道路種別",
    options=sorted(df["osm_highway_aggregated"].unique()),
    default=sorted(df["osm_highway_aggregated"].unique()),
    format_func=lambda x: HIGHWAY_LABELS.get(x, str(x)),
)

only_high_priority = st.sidebar.checkbox(
    "優先度が高い橋（総合優先度スコア 上位30件）のみ表示", value=False
)

st.sidebar.markdown("---")
bridge_names = ["（指定しない）"] + sorted(df["bridge_name"].unique().tolist())
selected_bridge = st.sidebar.selectbox("特定の橋を検索して表示", bridge_names)

st.sidebar.markdown("---")
st.sidebar.subheader("🛣️ 迂回路ルート")
if has_route_data:
    show_detour_routes = st.sidebar.checkbox("迂回路ルートを地図に表示する", value=False)
else:
    show_detour_routes = False
    st.sidebar.caption("このデータには迂回路ルート情報が含まれていません。")

# ---------------------------------------------------------------------------
# サイドバー下部：使用データの出典・注意事項
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    "**使用データ**\n\n"
    "- [国土交通データプラットフォーム](https://www.mlit-data.jp/)："
    "橋梁データ（rsdb_bridge）、道路交通センサス（rtc_2021）\n"
    "- [OpenStreetMap](https://www.openstreetmap.org/)："
    "道路ネットワークデータ（`osmnx`経由で取得）\n\n"
    "※国土交通データプラットフォーム API機能利用規約より、"
    "「このサービスは、国土交通データプラットフォームのAPI機能を使用していますが、"
    "最新のデータを保証するものではありません。」"
)

# ---------------------------------------------------------------------------
# 絞り込み処理
# ---------------------------------------------------------------------------
filtered_df = df[
    df["cluster_6_label"].isin(selected_clusters)
    & df["osm_highway_aggregated"].isin(selected_highways)
].copy()

if only_high_priority:
    filtered_df = filtered_df.sort_values("overall_rank").head(30)

if selected_bridge != "（指定しない）":
    filtered_df = filtered_df[filtered_df["bridge_name"] == selected_bridge]

st.write(f"**表示件数：{len(filtered_df)} 件** / 全{len(df)}件")

# ---------------------------------------------------------------------------
# タブ構成：地図 / EDA
# ---------------------------------------------------------------------------
tab_map, tab_eda = st.tabs(["🗺️ 地図", "📊 EDA（探索的データ分析）"])

# ============================= 地図タブ =====================================
with tab_map:
    if len(filtered_df) == 0:
        st.info("条件に一致する橋がありません。絞り込み条件を変更してください。")
    else:
        if selected_bridge != "（指定しない）":
            map_center = [filtered_df["lat"].iloc[0], filtered_df["lon"].iloc[0]]
            zoom_start = 16
        else:
            map_center = [filtered_df["lat"].mean(), filtered_df["lon"].mean()]
            zoom_start = 11

        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="OpenStreetMap")
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in filtered_df.iterrows():
            detour_display = f"{row['detour_length_m']:.0f}m" if pd.notna(row["detour_length_m"]) else "算出不可"
            popup_html = f"""
            <b>{row['bridge_name']}</b><br>
            架設年度: {row['dpf_kasetsu_nendo']}年<br>
            幅員: {row['dpf_fukuin']:.1f}m / 橋長: {row['dpf_kyouchou']:.1f}m<br>
            交通量: {row['traffic_count_24h_auto']:.0f}台/日<br>
            迂回路長: {detour_display}<br>
            分類: {row['cluster_label_ja']}<br>
            交通量×迂回路: {row['quadrant_label_ja']}<br>
            優先度順位: {int(row['overall_rank'])}位 / {len(df)}件中
            """
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color=CLUSTER_COLORS.get(row["cluster_6_label"], "#333333"),
                fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row["bridge_name"],
            ).add_to(marker_cluster)

        # ------------------------------------------------------------------
        # 迂回路ルートの描画（トグルON時のみ）
        # ------------------------------------------------------------------
        if show_detour_routes and has_route_data:
            routable_df = filtered_df[filtered_df["detour_route_wkt"].notna()]
            if len(routable_df) > MAX_ROUTES_TO_DRAW:
                st.warning(
                    f"迂回路ルートを表示できる橋が{len(routable_df)}件あり、上限（{MAX_ROUTES_TO_DRAW}件）を超えています。"
                    " 絞り込み条件を追加するか、特定の橋を1件選択してから表示してください。"
                )
            else:
                for _, row in routable_df.iterrows():
                    detour_display = f"{row['detour_length_m']:.0f}m" if pd.notna(row["detour_length_m"]) else "算出不可"
                    for line_coords in route_wkt_to_latlon_lines(row["detour_route_wkt"]):
                        folium.PolyLine(
                            line_coords,
                            color="#2E8B57",
                            weight=4,
                            opacity=0.8,
                            tooltip=f"{row['bridge_name']} の迂回路（長さ約{detour_display}）",
                        ).add_to(m)

        st.subheader("地図")
        st.caption("マーカーの色はクラスター（橋の特徴グループ）を表します。クリックすると詳細が表示されます。")
        if show_detour_routes:
            st.caption("緑色の線は、その橋が通行止めになった場合の迂回路ルートです。")
        st_folium(m, width=1100, height=600, returned_objects=[])

        with st.expander("凡例：マーカーの色とクラスターの対応"):
            for cid, label in CLUSTER_LABELS.items():
                st.markdown(
                    f"<span style='color:{CLUSTER_COLORS[cid]}; font-size:20px;'>●</span> {label}",
                    unsafe_allow_html=True,
                )
        # ------------------------------------------------------------------
        # 一覧表
        # ------------------------------------------------------------------
        st.subheader("一覧表")
        display_cols = {
            "bridge_name": "橋梁名",
            "dpf_kasetsu_nendo": "架設年度",
            "dpf_fukuin": "幅員(m)",
            "dpf_kyouchou": "橋長(m)",
            "traffic_count_24h_auto": "交通量(台/日)",
            "detour_length_m": "迂回路長(m)",
            "cluster_label_ja": "クラスター",
            "overall_rank": "優先度順位",
        }
        table_df = (
            filtered_df[list(display_cols.keys())]
            .rename(columns=display_cols)
            .sort_values("優先度順位")
            .reset_index(drop=True)
        )
        st.dataframe(table_df, use_container_width=True)

# ============================= EDAタブ =======================================
with tab_eda:
    st.caption(f"全{len(df):,}件のデータ全体をもとにした分布・相関の確認です（左側の絞り込み条件とは独立して表示しています）。")

    numeric_cols = {
        "dpf_kasetsu_nendo": "架設年度",
        "dpf_fukuin": "幅員(m)",
        "dpf_kyouchou": "橋長(m)",
        "traffic_count_24h_auto": "交通量(台/日)",
        "detour_length_m": "迂回路長(m)",
    }
    order = sorted(df["cluster_6_label"].unique())
    cluster_order_labels = [CLUSTER_LABELS.get(c, str(c)) for c in order]

    # --- クラスターごとの件数 ---
    st.subheader("クラスター別の件数")
    cluster_counts = (
        df["cluster_label_ja"].value_counts().reindex(cluster_order_labels).reset_index()
    )
    cluster_counts.columns = ["クラスター", "件数"]
    fig = px.bar(
        cluster_counts, x="件数", y="クラスター", orientation="h",
        color="クラスター",
        color_discrete_map={CLUSTER_LABELS[c]: CLUSTER_COLORS[c] for c in order},
    )
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "array", "categoryarray": cluster_order_labels[::-1]})
    st.plotly_chart(fig, use_container_width=True)

    # --- 数値変数の分布 ---
    st.subheader("数値変数の分布")
    st.caption("グラフ上でドラッグすると拡大表示できます（ダブルクリックで元に戻ります）。")
    cols = st.columns(3)
    for i, (col, label) in enumerate(numeric_cols.items()):
        with cols[i % 3]:
            fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=["#4C72B0"])
            fig.update_layout(
                title=label, xaxis_title="", yaxis_title="件数",
                margin=dict(l=10, r=10, t=40, b=10), height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- カテゴリ変数の分布（全体） ---
    st.subheader("カテゴリ変数の分布（全体）")
    cat_overall_specs = [
        ("osm_highway_aggregated", "道路種別", HIGHWAY_LABELS),
        ("osm_oneway", "一方通行の区分", {0: "一方通行ではない", 1: "一方通行"}),
    ]
    cat_overall_cols = st.columns(len(cat_overall_specs))
    for col_widget, (col, title, label_map) in zip(cat_overall_cols, cat_overall_specs):
        with col_widget:
            counts = df[col].map(label_map).value_counts().reset_index()
            counts.columns = [title, "件数"]
            fig = px.pie(counts, names=title, values="件数", hole=0.4)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(title=f"{title}の構成比（全体）", margin=dict(l=10, r=10, t=40, b=10), height=350)
            st.plotly_chart(fig, use_container_width=True)

    # --- クラスター別の箱ひげ図（数値変数：すべて一覧表示） ---
    st.subheader("クラスター別の分布（箱ひげ図）")
    st.caption("赤い破線は、その変数の全体（クラスター全体）における中央値です。凡例のクリックで表示・非表示を切り替えられます。")
    box_cols_per_row = 3
    numeric_items = list(numeric_cols.items())
    for row_start in range(0, len(numeric_items), box_cols_per_row):
        row_items = numeric_items[row_start:row_start + box_cols_per_row]
        cols = st.columns(len(row_items))
        for col_widget, (col, label) in zip(cols, row_items):
            with col_widget:
                fig = px.box(
                    df, x="cluster_6_label", y=col, color="cluster_6_label",
                    color_discrete_map=CLUSTER_COLORS,
                    category_orders={"cluster_6_label": order},
                )
                fig.add_hline(y=df[col].median(), line_dash="dash", line_color="red")
                fig.update_layout(
                    title=label, xaxis_title="クラスター", yaxis_title="",
                    showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- クラスター別のカテゴリ変数分布（構成比の積み上げ棒グラフ） ---
    st.subheader("クラスター別の分布（カテゴリ変数）")
    category_specs = [
        ("osm_highway_aggregated", "道路種別", HIGHWAY_LABELS),
        ("osm_oneway", "一方通行の区分", {0: "一方通行ではない", 1: "一方通行"}),
    ]
    cat_cols = st.columns(len(category_specs))
    for col_widget, (col, title, label_map) in zip(cat_cols, category_specs):
        with col_widget:
            plot_data = (
                df.groupby("cluster_6_label")[col]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
                .reindex(order)
                .rename(columns=label_map)
                .reset_index()
            )
            plot_data["cluster_6_label"] = plot_data["cluster_6_label"].astype(str)
            fig = px.bar(
                plot_data, x="cluster_6_label", y=list(label_map.values()),
                title=f"クラスター別 {title} 構成比",
            )
            fig.update_layout(
                xaxis_title="クラスター番号", yaxis_title="構成比", legend_title=title,
                margin=dict(l=10, r=10, t=40, b=10), height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- クラスターの特徴レーダーチャート ---
    st.subheader("クラスター特徴のレーダーチャート")
    st.caption(
        "各変数を全体の範囲で0〜1に正規化した上での、クラスターごとの平均値を表しています。"
        "凡例のクリックで特定のクラスターだけを表示・比較できます。"
    )
    radar_axes = dict(numeric_cols)
    radar_axes["osm_oneway"] = "一方通行率"

    radar_source = df[list(radar_axes.keys())].astype(float).copy()
    normalized = (radar_source - radar_source.min()) / (radar_source.max() - radar_source.min())
    normalized["cluster_6_label"] = df["cluster_6_label"]
    cluster_means = normalized.groupby("cluster_6_label").mean().reindex(order)

    theta_labels = list(radar_axes.values())
    fig = go.Figure()
    for c in order:
        values = cluster_means.loc[c, list(radar_axes.keys())].tolist()
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1],
            theta=theta_labels + theta_labels[:1],
            fill="toself",
            name=CLUSTER_LABELS.get(c, str(c)),
            line_color=CLUSTER_COLORS.get(c, "#333333"),
            opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=550,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 相関ヒートマップ ---
    st.subheader("数値変数の相関")
    corr = df[list(numeric_cols.keys())].rename(columns=numeric_cols).corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "各クラスターの特徴の言語的な解釈は、notebook（`03_eda_clustering.ipynb`）内の"
        "「クラスター分析（6分類）解釈まとめ」、または`cluster_interpretation.md`を参照してください。"
    )

# ---------------------------------------------------------------------------
# ページ下部フッター：使用データの出典・注意事項
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "使用データ：[国土交通データプラットフォーム](https://www.mlit-data.jp/)"
    "（橋梁データ・道路交通センサス）、[OpenStreetMap](https://www.openstreetmap.org/)（道路ネットワーク）　"
    "※国土交通データプラットフォーム API機能利用規約より、"
    "「このサービスは、国土交通データプラットフォームのAPI機能を使用していますが、"
    "最新のデータを保証するものではありません。」"
)
