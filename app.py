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
# YAMNet ロード（初回のみ）
# ─────────────────────────────────────────
@st.cache_resource
def load_yamnet():
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        # クラス名マップ
        class_map_path = model.class_map_path().numpy().decode("utf-8")
        import csv, urllib.request
        class_names = []
        with urllib.request.urlopen(class_map_path) as f:
            reader = csv.DictReader(io.TextIOWrapper(f))
            for row in reader:
                class_names.append(row["display_name"])
        return model, class_names
    except Exception as e:
        return None, None

# ─────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────
st.sidebar.header("🎵 検出対象")
detect_clap         = st.sidebar.checkbox("拍手 (Applause)",           value=True)
detect_laughter     = st.sidebar.checkbox("笑い声 (Laughter)",          value=False)
detect_cough_sneeze = st.sidebar.checkbox("咳・くしゃみ (Cough/Sneeze)", value=False)
detect_throat       = st.sidebar.checkbox("咳払い (Throat clearing)",   value=False)

st.sidebar.header("⚙️ 解析設定")
fps          = st.sidebar.selectbox("フレームレート (FPS)", [23.976, 24, 25, 29.97, 30, 59.94, 60], index=1)
sensitivity  = st.sidebar.slider("検出感度（大きいほど多く検出）", 0.05, 0.95, 0.35, 0.05,
                                  help="YAMNet信頼度スコアの下限閾値。小さいほど多く検出。")
min_interval = st.sidebar.slider("最小マーカー間隔（秒）", 0.5, 5.0, 1.5, 0.5)

st.sidebar.header("📤 書き出し形式")
export_fcpxml   = st.sidebar.checkbox("FCPXML (.fcpxml)  — Final Cut Pro",  value=True)
export_edl      = st.sidebar.checkbox("EDL (.edl)  — 汎用",                 value=False)
export_resolve  = st.sidebar.checkbox("DaVinci Resolve CSV (.csv)",         value=False)
export_premiere = st.sidebar.checkbox("Premiere Pro マーカー (.csv)",        value=False)
export_youtube  = st.sidebar.checkbox("YouTube チャプター (.txt)",            value=False)

# ─────────────────────────────────────────
# YAMNet ラベルマッピング
# ─────────────────────────────────────────
YAMNET_LABEL_MAP = {
    "Applause":        "Applause",
    "Clapping":        "Applause",
    "Cheering":        "Applause",
    "Laughter":        "Laughter",
    "Giggling":        "Laughter",
    "Chuckle, chortle":"Laughter",
    "Cough":           "Cough_Sneeze",
    "Sneeze":          "Cough_Sneeze",
    "Throat clearing": "Throat",
}

DETECT_FLAGS = {
    "Applause":    lambda: detect_clap,
    "Laughter":    lambda: detect_laughter,
    "Cough_Sneeze":lambda: detect_cough_sneeze,
    "Throat":      lambda: detect_throat,
}

# ─────────────────────────────────────────
# ヘルパー：タイムコード変換
# ─────────────────────────────────────────
def sec_to_tc(t, fps):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    f = int(round((t % 1) * fps))
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

def build_fcpxml(markers, fps):
    fps_int = int(round(fps))
    root = ET.Element("fcpxml", version="1.10")
    lib  = ET.SubElement(root, "library")
    evt  = ET.SubElement(lib, "event", name="Sound Markers")
    proj = ET.SubElement(evt, "project", name="Sound Markers")
    seq  = ET.SubElement(proj, "sequence",
                         format="r1", tcStart="0s",
                         tcFormat="NDF", audioLayout="stereo", audioRate="48k")
    ET.SubElement(seq, "spine")
    for t, label, conf in markers:
        frames = int(round(t * fps))
        m = ET.SubElement(seq, "marker",
                          start=f"{frames}/{fps_int}s",
                          duration="1/24s",
                          value=label,
                          completed="0")
        m.set("color", LABEL_COLOR_FCPXML.get(label, "blue"))
        m.set("note", f"confidence: {conf:.2f}")
    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")

def build_edl(markers, fps):
    lines = ["TITLE: SOUND MARKERS", "FCM: NON-DROP FRAME", ""]
    for idx, (t, label, conf) in enumerate(markers):
        tc  = sec_to_tc(t, fps)
        num = f"{idx+1:03d}"
        lines.append(f"{num}  AX       V     C        {tc} {tc} {tc} {tc}")
        lines.append(f" |M:{label}_{num} |C:{conf:.2f}")
        lines.append("")
    return "\n".join(lines)

def build_resolve_csv(markers, fps):
    rows = [["#", "Record In", "Record Out", "Name", "Color", "Confidence"]]
    for idx, (t, label, conf) in enumerate(markers):
        tc    = sec_to_tc(t, fps)
        color = LABEL_COLOR_RESOLVE.get(label, "Blue")
        rows.append([idx+1, tc, tc, label, color, f"{conf:.2f}"])
    return "\n".join([",".join(map(str, r)) for r in rows])

def build_premiere_csv(markers, fps):
    rows = [["Marker Name", "Description", "In", "Out", "Duration", "Marker Type"]]
    for t, label, conf in markers:
        tc = sec_to_tc(t, fps)
        rows.append([label, f"confidence:{conf:.2f}", tc, tc, "00:00:00:00", "Comment"])
    return "\n".join([",".join(map(str, r)) for r in rows])

def build_youtube(markers):
    return "\n".join([f"{sec_to_yt(t)} {label}" for t, label, conf in markers])

# ─────────────────────────────────────────
# YAMNet 解析
# ─────────────────────────────────────────
def analyze_with_yamnet(y, sr, model, class_names, sensitivity, min_interval):
    import tensorflow as tf

    # YAMNet は sr=16000 を期待
    if sr != 16000:
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    else:
        y16 = y

    waveform = tf.constant(y16, dtype=tf.float32)
    scores, embeddings, spectrogram = model(waveform)
    # scores shape: (frames, 521) — 約0.48秒/フレーム
    scores_np = scores.numpy()
    n_frames  = scores_np.shape[0]
    frame_dur = len(y16) / 16000 / n_frames  # 秒/フレーム

    raw_markers = []
    for fi in range(n_frames):
        t = fi * frame_dur + frame_dur / 2  # フレーム中央時刻
        for yamnet_name, internal_label in YAMNET_LABEL_MAP.items():
            if not DETECT_FLAGS.get(internal_label, lambda: False)():
                continue
            matches = [j for j, name in enumerate(class_names)
                       if yamnet_name.lower() in name.lower()]
            if not matches:
                continue
            conf = float(max(scores_np[fi, j] for j in matches))
            if conf >= sensitivity:
                raw_markers.append((t, internal_label, conf))

    # 重複排除：同ラベルで min_interval 以内は信頼度最大を残す
    raw_markers.sort(key=lambda x: x[0])
    markers = []
    last_per_label = {}
    for t, label, conf in raw_markers:
        last_t = last_per_label.get(label, -999)
        if t - last_t >= min_interval:
            markers.append((t, label, conf))
            last_per_label[label] = t
        elif markers and markers[-1][1] == label and conf > markers[-1][2]:
            markers[-1] = (t, label, conf)

    return markers

# ─────────────────────────────────────────
# フォールバック：librosa解析
# ─────────────────────────────────────────
def analyze_with_librosa(y, sr, sensitivity, min_interval,
                          detect_clap, detect_laughter, detect_cough_sneeze, detect_throat):
    hop       = 512
    db        = librosa.amplitude_to_db(
        librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0], ref=np.max)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    delta = 5.0 - (sensitivity / 0.95) * 4.0
    peaks = librosa.util.peak_pick(onset_env,
                                   pre_max=3, post_max=3, pre_avg=3, post_avg=5,
                                   delta=delta, wait=10)
    detected_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)

    raw = []
    for t in detected_times:
        fi = librosa.time_to_frames(t, sr=sr, hop_length=hop)
        if fi >= len(db) or fi >= mfcc.shape[1]: continue
        if db[fi] < -32: continue
        m1, m2 = mfcc[1, fi], mfcc[2, fi]
        fc = centroid[fi]
        oe = onset_env[fi] if fi < len(onset_env) else 0
        conf = float(np.clip(oe / (onset_env.max() + 1e-6), 0, 1))

        if detect_clap and fc > 3500 and m1 < 40 and m2 > -10:
            raw.append((t, "Applause", conf))
        elif detect_laughter and 1200 < fc < 2800 and m1 > 60:
            if fi + 5 < len(db) and db[fi + 5] > -25:
                raw.append((t, "Laughter", conf))
        elif detect_cough_sneeze and fc > 3000 and m1 < 20 and m2 < -30:
            raw.append((t, "Cough_Sneeze", conf))
        elif detect_throat and 800 < fc < 1800 and 20 <= m1 <= 60:
            raw.append((t, "Throat", conf))

    raw.sort(key=lambda x: x[0])
    markers = []
    last_t = -999.0
    for m in raw:
        if m[0] - last_t >= min_interval:
            markers.append(m)
            last_t = m[0]
    return markers, db, hop

# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
uploaded_file = st.file_uploader("音声ファイルをアップロード",
                                  type=["wav", "mp3", "m4a", "aac", "flac"])

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

    with st.spinner("AIモデルを準備中...（初回は少し時間がかかります）"):
        model, class_names = load_yamnet()

    if model is not None:
        st.info("🤖 YAMNet（Googleの音声認識AI）で解析中...")
        with st.spinner("解析中..."):
            markers = analyze_with_yamnet(
                y, sr, model, class_names, sensitivity, min_interval)
        hop = 512
        db  = librosa.amplitude_to_db(
            librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0], ref=np.max)
    else:
        st.warning("⚠️ YAMNetが利用できないため librosa モードで解析します。")
        markers, db, hop = analyze_with_librosa(
            y, sr, sensitivity, min_interval,
            detect_clap, detect_laughter, detect_cough_sneeze, detect_throat)

    # ─────────────────────────────────────────
    # 結果表示
    # ─────────────────────────────────────────
    if not markers:
        st.warning("検出されませんでした。感度スライダーを左に動かしてみてください。")
        st.stop()

    st.success(f"✅ {len(markers)} 箇所のマーカーを検出しました")

    # 波形 + マーカープロット
    duration   = len(y) / sr
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
        y_pos = [db[min(int(t / duration * len(db)), len(db)-1)] for t, _ in pts]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=y_pos,
            mode="markers",
            marker=dict(size=14, color=COLOR_MAP.get(label, "#fff"), symbol="triangle-up"),
            name=label,
            hovertemplate=f"<b>{label}</b><br>%{{x:.2f}}s<br>conf: %{{customdata:.0%}}",
            customdata=[p[1] for p in pts]
        ))

    fig.update_layout(
        title="波形 & 検出マーカー",
        xaxis_title="時間 (秒)", yaxis_title="音量 (dB)",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="#111111", paper_bgcolor="#111111",
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
    cols    = st.columns(5)
    col_idx = 0

    if export_fcpxml:
        cols[col_idx].download_button("⬇️ FCPXML\n(Final Cut Pro)",
            data=build_fcpxml(markers, fps), file_name="markers.fcpxml", mime="text/xml")
        col_idx += 1
    if export_edl:
        cols[col_idx].download_button("⬇️ EDL\n(汎用)",
            data=build_edl(markers, fps), file_name="markers.edl", mime="text/plain")
        col_idx += 1
    if export_resolve:
        cols[col_idx].download_button("⬇️ Resolve CSV\n(DaVinci Resolve)",
            data=build_resolve_csv(markers, fps), file_name="markers_resolve.csv", mime="text/csv")
        col_idx += 1
    if export_premiere:
        cols[col_idx].download_button("⬇️ Premiere CSV\n(Premiere Pro)",
            data=build_premiere_csv(markers, fps), file_name="markers_premiere.csv", mime="text/csv")
        col_idx += 1
    if export_youtube:
        cols[col_idx].download_button("⬇️ YouTube チャプター\n(.txt)",
            data=build_youtube(markers), file_name="chapters.txt", mime="text/plain")
