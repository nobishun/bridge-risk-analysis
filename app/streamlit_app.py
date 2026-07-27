"""橋梁リスク可視化ダッシュボード（シンプル版 + 迂回路表示 + EDA閲覧）

地図上で、橋梁データを属性（クラスター・道路種別）で絞り込んだり、
特定の橋を選んで表示したりできる、Streamlitアプリです。
「迂回路ルートの表示」トグルと「EDA」タブを追加しています。
"""
import os
import tempfile

import folium
import geopandas as gpd
#import japanize_matplotlib  # matplotlib/seabornグラフの日本語表示に必要←Python(3.14)に非対応
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from folium.plugins import MarkerCluster
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from streamlit_folium import st_folium

st.set_page_config(page_title="橋梁リスク可視化ダッシュボード", layout="wide")

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
    df["cluster_label_ja"] = df["cluster_6_label"].map(CLUSTER_LABELS)
    df["highway_label_ja"] = df["osm_highway_aggregated"].map(HIGHWAY_LABELS)
    df["quadrant_label_ja"] = df["traffic_detour_quadrant"].map(QUADRANT_LABELS)
    df["oneway_label_ja"] = df["osm_oneway"].map({0: "一方通行ではない", 1: "一方通行", False: "一方通行ではない", True: "一方通行"})
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
            popup_html = f"""
            <b>{row['bridge_name']}</b><br>
            架設年度: {row['dpf_kasetsu_nendo']}年<br>
            幅員: {row['dpf_fukuin']:.1f}m / 橋長: {row['dpf_kyouchou']:.1f}m<br>
            交通量: {row['traffic_count_24h_auto']:.0f}台/日<br>
            迂回路長: {row['detour_length_m']:.0f}m<br>
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
                    for line_coords in route_wkt_to_latlon_lines(row["detour_route_wkt"]):
                        folium.PolyLine(
                            line_coords,
                            color="#2E8B57",
                            weight=4,
                            opacity=0.8,
                            tooltip=f"{row['bridge_name']} の迂回路（長さ約{row['detour_length_m']:.0f}m）",
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
    st.caption("全1,226件のデータ全体をもとにした分布・相関の確認です（左側の絞り込み条件とは独立して表示しています）。")

    # --- クラスターごとの件数 ---
    st.subheader("クラスター別の件数")
    cluster_counts = df["cluster_label_ja"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    cluster_counts.plot(kind="barh", ax=ax, color=[CLUSTER_COLORS.get(i, "#333333") for i in range(len(cluster_counts))])
    ax.set_xlabel("件数")
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close(fig)

    numeric_cols = {
        "dpf_kasetsu_nendo": "架設年度",
        "dpf_fukuin": "幅員(m)",
        "dpf_kyouchou": "橋長(m)",
        "traffic_count_24h_auto": "交通量(台/日)",
        "detour_length_m": "迂回路長(m)",
    }

    # --- 数値変数の分布 ---
    st.subheader("数値変数の分布")
    cols = st.columns(3)
    for i, (col, label) in enumerate(numeric_cols.items()):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df[col].dropna(), ax=ax, color="#4C72B0")
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("")
            st.pyplot(fig)
            plt.close(fig)

    # --- クラスター別の箱ひげ図（数値変数：すべて小さめに一覧表示） ---
    st.subheader("クラスター別の分布（箱ひげ図）")
    st.caption("赤い破線は、その変数の全体（クラスター全体）における中央値です。")
    order = sorted(df["cluster_6_label"].unique())
    box_cols_per_row = 3
    numeric_items = list(numeric_cols.items())
    for row_start in range(0, len(numeric_items), box_cols_per_row):
        row_items = numeric_items[row_start:row_start + box_cols_per_row]
        cols = st.columns(len(row_items))
        for col_widget, (col, label) in zip(cols, row_items):
            with col_widget:
                fig, ax = plt.subplots(figsize=(3.5, 2.8))
                # 【注意】sns.boxplotは、hue指定時に legend=False を渡すと
                # seabornのバージョンによってUnboundLocalErrorになることがあるため、
                # legend=Falseは使わず、描画後にlegendオブジェクトを明示的に削除する。
                sns.boxplot(
                    data=df, x="cluster_6_label", y=col, order=order,
                    hue="cluster_6_label", palette=CLUSTER_COLORS, ax=ax,
                )
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                ax.axhline(df[col].median(), color="red", linestyle="--", linewidth=1)
                ax.set_title(label, fontsize=10)
                ax.set_xlabel("クラスター", fontsize=8)
                ax.set_ylabel("")
                ax.tick_params(labelsize=8)
                st.pyplot(fig)
                plt.close(fig)

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
            )
            plot_data = plot_data.rename(columns=label_map)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            plot_data.plot(kind="bar", stacked=True, ax=ax, cmap="viridis")
            ax.set_title(f"クラスター別 {title} 構成比", fontsize=10)
            ax.set_xlabel("クラスター番号")
            ax.set_ylabel("構成比")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title=title, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

    # --- 相関ヒートマップ ---
    st.subheader("数値変数の相関")
    corr = df[list(numeric_cols.keys())].rename(columns=numeric_cols).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

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
