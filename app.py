import streamlit as st
import librosa
import numpy as np
import io
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="サウンドマーカー生成ツール", page_icon="🎬", layout="wide")

st.title("🎬 サウンドマーカー生成ツール")
st.write("音声を解析し、特定の音をタイムラインマーカーとして書き出します。")

# ─────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────
st.sidebar.header("🎵 検出対象")
detect_clap        = st.sidebar.checkbox("拍手 (Applause)",          value=True)
detect_laughter    = st.sidebar.checkbox("笑い声 (Laughter)",         value=False)
detect_cough_sneeze= st.sidebar.checkbox("咳・くしゃみ (Cough/Sneeze)",value=False)
detect_throat      = st.sidebar.checkbox("咳払い (Throat clearing)",  value=False)

st.sidebar.header("⚙️ 解析設定")
fps         = st.sidebar.selectbox("フレームレート (FPS)", [23.976, 24, 25, 29.97, 30, 59.94, 60], index=1)
sensitivity = st.sidebar.slider("検出感度（大きいほど多く検出）", 1.0, 5.0, 2.5, 0.5)

st.sidebar.header("📤 書き出し形式")
export_fcpxml   = st.sidebar.checkbox("FCPXML (.fcpxml)  — Final Cut Pro",    value=True)
export_edl      = st.sidebar.checkbox("EDL (.edl)  — 汎用",                   value=False)
export_resolve  = st.sidebar.checkbox("DaVinci Resolve CSV (.csv)",           value=False)
export_premiere = st.sidebar.checkbox("Premiere Pro マーカー (.csv)",          value=False)
export_youtube  = st.sidebar.checkbox("YouTube チャプター (.txt)",              value=False)

# ─────────────────────────────────────────
# ヘルパー：タイムコード変換
# ─────────────────────────────────────────
def sec_to_tc(t, fps):
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = int(t % 60)
    f  = int(round((t % 1) * fps))
    if f >= int(fps):
        f = int(fps) - 1
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

def sec_to_yt(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

# ─────────────────────────────────────────
# 書き出し関数
# ─────────────────────────────────────────
LABEL_COLOR_FCPXML = {
    "Applause":    "blue",
    "Laughter":    "yellow",
    "Cough_Sneeze":"red",
    "Throat":      "pink",
}

LABEL_COLOR_RESOLVE = {
    "Applause":    "Blue",
    "Laughter":    "Yellow",
    "Cough_Sneeze":"Red",
    "Throat":      "Pink",
}

LABEL_COLOR_PREMIERE = {
    "Applause":    3,   # 青
    "Laughter":    2,   # 黄
    "Cough_Sneeze":1,   # 赤
    "Throat":      4,   # ピンク
}

def build_fcpxml(markers, fps):
    fps_int = int(round(fps))
    root = ET.Element("fcpxml", version="1.10")
    lib  = ET.SubElement(root, "library")
    evt  = ET.SubElement(lib, "event", name="Sound Markers")
    proj = ET.SubElement(evt, "project", name="Sound Markers")
    seq  = ET.SubElement(proj, "sequence",
                         format=f"r1",
                         tcStart="0s",
                         tcFormat="NDF",
                         audioLayout="stereo",
                         audioRate="48k")
    ET.SubElement(seq, "spine")
    for t, label, conf in markers:
        frames = int(round(t * fps))
        m = ET.SubElement(seq, "marker",
                          start=f"{frames}/{fps_int}s",
                          duration="1/24s",
                          value=label,
                          completed="0")
        color = LABEL_COLOR_FCPXML.get(label, "blue")
        m.set("color", color)
        m.set("note", f"confidence: {conf:.2f}")
    rough = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ")

def build_edl(markers, fps):
    lines = ["TITLE: SOUND MARKERS", "FCM: NON-DROP FRAME", ""]
    for idx, (t, label, conf) in enumerate(markers):
        tc = sec_to_tc(t, fps)
        num = f"{idx+1:03d}"
        lines.append(f"{num}  AX       V     C        {tc} {tc} {tc} {tc}")
        lines.append(f" |M:{label}_{num} |C:{conf:.2f}")
        lines.append("")
    return "\n".join(lines)

def build_resolve_csv(markers, fps):
    rows = [["#", "Record In", "Record Out", "Name", "Color", "Confidence"]]
    for idx, (t, label, conf) in enumerate(markers):
        tc = sec_to_tc(t, fps)
        color = LABEL_COLOR_RESOLVE.get(label, "Blue")
        rows.append([idx+1, tc, tc, label, color, f"{conf:.2f}"])
    return "\n".join([",".join(map(str, r)) for r in rows])

def build_premiere_csv(markers, fps):
    # Premiere Pro マーカー CSV 仕様
    rows = [["Marker Name", "Description", "In", "Out", "Duration", "Marker Type"]]
    for t, label, conf in markers:
        tc = sec_to_tc(t, fps)
        rows.append([label, f"confidence:{conf:.2f}", tc, tc, "00:00:00:00", "Comment"])
    return "\n".join([",".join(map(str, r)) for r in rows])

def build_youtube(markers):
    lines = []
    for t, label, conf in markers:
        tc = sec_to_yt(t)
        lines.append(f"{tc} {label}")
    return "\n".join(lines)

# ─────────────────────────────────────────
# ファイルアップロード & 解析
# ─────────────────────────────────────────
uploaded_file = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "m4a", "aac"])

if uploaded_file is not None:
    any_selected = detect_clap or detect_laughter or detect_cough_sneeze or detect_throat
    if not any_selected:
        st.warning("サイドバーで検出したい音を1つ以上選択してください。")
        st.stop()

    any_export = export_fcpxml or export_edl or export_resolve or export_premiere or export_youtube
    if not any_export:
        st.warning("サイドバーで書き出し形式を1つ以上選択してください。")
        st.stop()

    with st.spinner("音声を読み込み中..."):
        audio_bytes = uploaded_file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)

    with st.spinner("音響特徴量を計算中..."):
        hop = 512
        rmse      = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        db        = librosa.amplitude_to_db(rmse, ref=np.max)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3, post_max=3, pre_avg=3, post_avg=5,
        delta=3.0 / sensitivity, wait=10
    )
    detected_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)

    raw_markers = []
    for t in detected_times:
        fi = librosa.time_to_frames(t, sr=sr, hop_length=hop)
        if fi >= len(db) or fi >= mfcc.shape[1]:
            continue
        if db[fi] < -32:
            continue
        m1, m2, m3 = mfcc[1, fi], mfcc[2, fi], mfcc[3, fi]
        fc = centroid[fi]
        oe = onset_env[fi] if fi < len(onset_env) else 0

        if detect_clap and fc > 3500 and m1 < 40 and m2 > -10:
            conf = float(np.clip(oe / (onset_env.max() + 1e-6), 0, 1))
            raw_markers.append((t, "Applause", conf))
        elif detect_laughter and 1200 < fc < 2800 and m1 > 60:
            if fi + 5 < len(db) and db[fi + 5] > -25:
                conf = float(np.clip(oe / (onset_env.max() + 1e-6), 0, 1))
                raw_markers.append((t, "Laughter", conf))
        elif detect_cough_sneeze and fc > 3000 and m1 < 20 and m2 < -30:
            conf = float(np.clip(oe / (onset_env.max() + 1e-6), 0, 1))
            raw_markers.append((t, "Cough_Sneeze", conf))
        elif detect_throat and 800 < fc < 1800 and 20 <= m1 <= 60:
            conf = float(np.clip(oe / (onset_env.max() + 1e-6), 0, 1))
            raw_markers.append((t, "Throat", conf))

    # 重複排除（1秒以内）
    raw_markers.sort(key=lambda x: x[0])
    markers = []
    last_t = -1.0
    for m in raw_markers:
        if m[0] - last_t > 1.0:
            markers.append(m)
            last_t = m[0]

    # ─────────────────────────────────────────
    # 結果表示
    # ─────────────────────────────────────────
    if not markers:
        st.warning("検出されませんでした。感度スライダーを右に動かしてみてください。")
        st.stop()

    st.success(f"✅ {len(markers)} 箇所のマーカーを検出しました")

    # 波形 + マーカー プロット
    duration = len(y) / sr
    times_wave = np.linspace(0, duration, num=len(db))

    COLOR_MAP = {
        "Applause":    "#4A9EFF",
        "Laughter":    "#FFD700",
        "Cough_Sneeze":"#FF4C4C",
        "Throat":      "#FF69B4",
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_wave, y=db,
        mode="lines", line=dict(color="#888888", width=1),
        name="音量 (dB)", hovertemplate="%{x:.2f}s / %{y:.1f}dB"
    ))

    for label in set(m[1] for m in markers):
        pts = [(t, conf) for t, l, conf in markers if l == label]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts],
            y=[db[min(librosa.time_to_frames(p[0], sr=sr, hop_length=hop), len(db)-1)] for p in pts],
            mode="markers",
            marker=dict(size=12, color=COLOR_MAP.get(label, "#ffffff"), symbol="triangle-up"),
            name=label,
            hovertemplate=f"<b>{label}</b><br>%{{x:.2f}}s<br>conf: %{{customdata:.2f}}",
            customdata=[p[1] for p in pts]
        ))

    fig.update_layout(
        title="波形 & 検出マーカー",
        xaxis_title="時間 (秒)",
        yaxis_title="音量 (dB)",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#eeeeee"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 検出結果テーブル
    df = pd.DataFrame(
        [(sec_to_tc(t, fps), label, f"{conf:.0%}") for t, label, conf in markers],
        columns=["タイムコード", "ラベル", "信頼度"]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────
    # 書き出しボタン
    # ─────────────────────────────────────────
    st.subheader("📤 書き出し")
    cols = st.columns(5)
    col_idx = 0

    if export_fcpxml:
        data = build_fcpxml(markers, fps)
        cols[col_idx].download_button(
            "⬇️ FCPXML\n(Final Cut Pro)",
            data=data, file_name="markers.fcpxml", mime="text/xml"
        )
        col_idx += 1

    if export_edl:
        data = build_edl(markers, fps)
        cols[col_idx].download_button(
            "⬇️ EDL\n(汎用)",
            data=data, file_name="markers.edl", mime="text/plain"
        )
        col_idx += 1

    if export_resolve:
        data = build_resolve_csv(markers, fps)
        cols[col_idx].download_button(
            "⬇️ Resolve CSV\n(DaVinci Resolve)",
            data=data, file_name="markers_resolve.csv", mime="text/csv"
        )
        col_idx += 1

    if export_premiere:
        data = build_premiere_csv(markers, fps)
        cols[col_idx].download_button(
            "⬇️ Premiere CSV\n(Premiere Pro)",
            data=data, file_name="markers_premiere.csv", mime="text/csv"
        )
        col_idx += 1

    if export_youtube:
        data = build_youtube(markers)
        cols[col_idx].download_button(
            "⬇️ YouTube チャプター\n(.txt)",
            data=data, file_name="chapters.txt", mime="text/plain"
        )
