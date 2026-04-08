#!/usr/bin/env python3
"""Word alignment viewer — Flask dev server.

Usage:
    python alignment/serve_viewer.py [--port 7860]

오디오 + 단어 타임스탬프를 시각화. 오디오 재생 시 현재 단어 하이라이팅.
"""
import argparse, base64, io, json, pickle, threading
from pathlib import Path

import pyarrow.ipc as ipc
import torchaudio
from flask import Flask, jsonify, request

MERGED = Path("/mnt/tmp/cache/word_alignments_merged")
INDEX_DIR = MERGED / "viewer_indices"

DATASETS = {
    "librispeech/dev-clean":       MERGED / "librispeech" / "dev-clean.arrow",
    "librispeech/train-clean-100": MERGED / "librispeech" / "train-clean-100.arrow",
    "librispeech/train-other-500": MERGED / "librispeech" / "train-other-500.arrow",
    "mls/train":                   MERGED / "mls"         / "train.arrow",
    "gigaspeech/train":            MERGED / "gigaspeech"  / "train.arrow",
    "voxpopuli/train":             MERGED / "voxpopuli"   / "train.arrow",
}

# ── LibriSpeech: flac 파일 직접 로드 ─────────────────────────────────────────

LS_ROOT = Path("/mnt/tmp/cache/LibriSpeech")

def _ls_audio_bytes(row: dict) -> bytes | None:
    split = row.get("split", "")
    sp    = row.get("speaker_id", "")
    ch    = row.get("chapter_id", "")
    uid   = row.get("utterance_id", "")
    p = LS_ROOT / split / sp / ch / f"{uid}.flac"
    return p.read_bytes() if p.exists() else None


# ── HF 소스 Arrow (MLS / GigaSpeech / VoxPopuli) ────────────────────────────
#   utterance_id → (shard_path, row_within_shard) 인덱스를 백그라운드로 빌드

HF_SOURCES = {
    "mls/train": {
        "dir":    Path("/mnt/tmp/cache/parler-tts___mls_eng_10k/default/0.0.0/"
                       "22ad0ad7a3f32d8ec26c6b670ba7091200a5a1f8/"),
        "prefix": "mls_eng_10k-train",
        # included_fields=[0] → audio struct only (bytes는 읽지 않고 path만 추출)
        "id_field_idx": 0,
        "get_ids":  lambda batch: [
            p.replace(".opus", "")
            for p in batch.column("audio").field("path").to_pylist()
        ],
        # 오디오 fetch 시 audio struct (field 0) 만 읽음
        "audio_field_idx": 0,
        "get_audio": lambda batch, li: batch.column("audio")[li].as_py(),
    },
    "gigaspeech/train": {
        "dir":    Path("/mnt/tmp/cache/speechcolab___gigaspeech/xl/0.0.0/"
                       "63c0836b643dc6136a608de041e56b67c12649b3/"),
        "prefix": "gigaspeech-train",
        "id_field_idx": 0,   # segment_id
        "get_ids":  lambda batch: batch.column("segment_id").to_pylist(),
        "audio_field_idx": 3,
        "get_audio": lambda batch, li: batch.column("audio")[li].as_py(),
    },
    "voxpopuli/train": {
        "dir":    Path("/mnt/tmp/cache/facebook___voxpopuli/en/0.0.0/"
                       "42f01879c780b4a2e90ec0b4f616c2ece526e4f1/"),
        "prefix": "voxpopuli-train",
        "id_field_idx": 0,   # audio_id
        "get_ids":  lambda batch: batch.column("audio_id").to_pylist(),
        "audio_field_idx": 2,
        "get_audio": lambda batch, li: batch.column("audio")[li].as_py(),
    },
}

_hf_indices: dict = {}   # ds_key → {utterance_id: (shard_path, row_within_shard)}
_index_status: dict = {}  # ds_key → "building" | "ready" | "failed"


def _build_index(ds_key: str):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    pkl = INDEX_DIR / f"{ds_key.replace('/', '_')}.pkl"

    if pkl.exists():
        try:
            _hf_indices[ds_key] = pickle.loads(pkl.read_bytes())
            _index_status[ds_key] = "ready"
            print(f"[index] {ds_key}: loaded {len(_hf_indices[ds_key]):,} entries", flush=True)
            return
        except Exception:
            pass

    print(f"[index] {ds_key}: building...", flush=True)
    cfg = HF_SOURCES[ds_key]
    shards = sorted(f for f in cfg["dir"].iterdir()
                    if f.name.startswith(cfg["prefix"]) and f.suffix == ".arrow")
    opts = ipc.IpcReadOptions(included_fields=[cfg["id_field_idx"]])
    idx: dict = {}

    for shard_path in shards:
        row_off = 0
        try:
            with open(shard_path, "rb") as f:
                reader = ipc.open_stream(f, options=opts)
                while True:
                    try:
                        batch = reader.read_next_batch()
                        for uid in cfg["get_ids"](batch):
                            if uid:
                                idx[uid] = (str(shard_path), row_off)
                            row_off += 1
                    except StopIteration:
                        break
        except Exception as e:
            print(f"[index] {shard_path.name}: {e}", flush=True)

    _hf_indices[ds_key] = idx
    pkl.write_bytes(pickle.dumps(idx))
    _index_status[ds_key] = "ready"
    print(f"[index] {ds_key}: done ({len(idx):,} entries)", flush=True)


def _fetch_hf_audio(ds_key: str, utterance_id: str) -> dict | None:
    idx = _hf_indices.get(ds_key)
    if not idx or utterance_id not in idx:
        return None
    shard_str, target_row = idx[utterance_id]
    cfg = HF_SOURCES[ds_key]
    opts = ipc.IpcReadOptions(included_fields=[cfg["audio_field_idx"]])

    cum = 0
    try:
        with open(shard_str, "rb") as f:
            reader = ipc.open_stream(f, options=opts)
            while True:
                try:
                    batch = reader.read_next_batch()
                    if cum + batch.num_rows > target_row:
                        return cfg["get_audio"](batch, target_row - cum)
                    cum += batch.num_rows
                except StopIteration:
                    break
    except Exception as e:
        print(f"[fetch] {ds_key}/{utterance_id}: {e}", flush=True)
    return None


# ── 오디오 bytes → wav base64 ─────────────────────────────────────────────────

def _bytes_to_wav_b64(raw: bytes | None) -> str | None:
    if not raw:
        return None
    try:
        wav, sr = torchaudio.load(io.BytesIO(raw))
    except Exception:
        return None
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    out = io.BytesIO()
    torchaudio.save(out, wav, sr, format="wav")
    out.seek(0)
    return base64.b64encode(out.read()).decode()


def row_to_wav_b64(ds_key: str, row: dict) -> tuple[str | None, str]:
    """Returns (b64_wav_or_None, status_msg)."""
    if ds_key.startswith("librispeech"):
        raw = _ls_audio_bytes(row)
        if raw is None:
            return None, "오디오 파일 없음"
        return _bytes_to_wav_b64(raw), ""
    else:
        status = _index_status.get(ds_key, "unknown")
        if status == "building":
            return None, f"오디오 인덱스 빌드 중... ({ds_key})"
        if status == "failed":
            return None, "인덱스 빌드 실패"
        audio = _fetch_hf_audio(ds_key, row.get("utterance_id", ""))
        if audio is None:
            return None, "오디오 없음 (utterance_id 미발견)"
        raw = audio.get("bytes") if isinstance(audio, dict) else None
        if not raw:
            path = (audio.get("path") if isinstance(audio, dict) else None)
            if path and Path(path).exists():
                raw = open(path, "rb").read()
        return _bytes_to_wav_b64(raw), "" if raw else "오디오 bytes 없음"


# ── Alignment Arrow 랜덤 접근 ─────────────────────────────────────────────────

_readers: dict = {}

def _get_reader(path: Path):
    key = str(path)
    if key not in _readers:
        f = open(path, "rb")
        reader = ipc.open_file(f)
        offsets, cum = [], 0
        for i in range(reader.num_record_batches):
            offsets.append(cum)
            cum += reader.get_batch(i).num_rows
        _readers[key] = (reader, offsets, cum)
    return _readers[key]


def get_row(path: Path, idx: int) -> dict:
    reader, offsets, _ = _get_reader(path)
    bi = 0
    for i, off in enumerate(offsets):
        if off <= idx:
            bi = i
        else:
            break
    local = idx - offsets[bi]
    batch = reader.get_batch(bi)
    return {col: batch.column(col)[local].as_py() for col in batch.schema.names}


def total_rows(path: Path) -> int:
    _, _, total = _get_reader(path)
    return total


# ── Flask ─────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=None)

HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Word Alignment Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }
header { background: #1a1d27; padding: 16px 24px; border-bottom: 1px solid #2d3748;
         display: flex; align-items: center; gap: 16px; }
header h1 { font-size: 1.1rem; font-weight: 600; color: #90cdf4; }
.controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 16px 24px; background: #13151f; border-bottom: 1px solid #2d3748; }
select, button, input[type=number] {
  background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568;
  padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; cursor: pointer; }
button { background: #3182ce; border-color: #3182ce; }
button:hover { background: #2b6cb0; }
button:disabled { background: #4a5568; border-color: #4a5568; cursor: default; }
input[type=number] { width: 90px; }
.label { font-size: 0.8rem; color: #a0aec0; }
.total { font-size: 0.8rem; color: #68d391; }
main { padding: 24px; max-width: 960px; }
.card { background: #1a1d27; border: 1px solid #2d3748; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
.meta { font-size: 0.78rem; color: #a0aec0; display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 14px; }
.meta span b { color: #90cdf4; }
audio { width: 100%; margin-bottom: 16px; accent-color: #3182ce; }
.waveform-wrap { position: relative; margin-bottom: 14px; cursor: pointer; }
canvas#waveform { width: 100%; height: 60px; border-radius: 6px; background: #0d1117; display: block; }
.progress-line { position: absolute; top: 0; bottom: 0; width: 2px; background: #f6ad55; pointer-events: none; left: 0; }
.words { display: flex; flex-wrap: wrap; gap: 6px; line-height: 1.8; }
.word { display: inline-flex; align-items: center; padding: 3px 8px;
        border-radius: 4px; font-size: 0.95rem; cursor: pointer;
        background: #2d3748; border: 1px solid transparent; transition: background 0.1s; }
.word:hover { background: #3a4a5c; }
.word.active { background: #2b4c7e; border-color: #63b3ed; color: #bee3f8; }
.word .ts { font-size: 0.65rem; color: #718096; margin-left: 4px; }
.word.active .ts { color: #90cdf4; }
.score-bar { display: inline-block; width: 4px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
#status { font-size: 0.85rem; color: #fc8181; padding: 8px 24px; }
</style>
</head>
<body>
<header>
  <h1>Word Alignment Viewer</h1>
</header>
<div class="controls">
  <span class="label">Dataset</span>
  <select id="ds"></select>
  <span class="label">Index</span>
  <input type="number" id="idx" value="0" min="0">
  <span class="total" id="total"></span>
  <button id="btnRand">Random</button>
  <button id="btnPrev">◀ Prev</button>
  <button id="btnNext">Next ▶</button>
  <button id="btnLoad">Load</button>
</div>
<div id="status"></div>
<main>
  <div class="card" id="card" style="display:none">
    <div class="meta" id="meta"></div>
    <div class="waveform-wrap" id="waveWrap">
      <canvas id="waveform"></canvas>
      <div class="progress-line" id="prog"></div>
    </div>
    <audio id="player" controls></audio>
    <div class="words" id="words"></div>
  </div>
</main>
<script>
const dsEl=document.getElementById('ds'), idxEl=document.getElementById('idx'),
      totalEl=document.getElementById('total'), statusEl=document.getElementById('status'),
      card=document.getElementById('card'), metaEl=document.getElementById('meta'),
      wordsEl=document.getElementById('words'), player=document.getElementById('player'),
      prog=document.getElementById('prog'), waveWrap=document.getElementById('waveWrap');

let currentData=null, animFrame=null;

// ── init datasets ──
fetch('/api/datasets').then(r=>r.json()).then(list=>{
  list.forEach(d=>{ const o=document.createElement('option'); o.value=o.textContent=d; dsEl.append(o); });
  dsEl.dispatchEvent(new Event('change'));
});

dsEl.addEventListener('change', ()=>{
  fetch('/api/total?ds='+encodeURIComponent(dsEl.value))
    .then(r=>r.json()).then(d=>{ totalEl.textContent='/ '+d.total.toLocaleString(); idxEl.max=d.total-1; });
});

document.getElementById('btnRand').onclick=()=>{
  fetch('/api/total?ds='+encodeURIComponent(dsEl.value)).then(r=>r.json()).then(d=>{
    idxEl.value=Math.floor(Math.random()*d.total); load();
  });
};
document.getElementById('btnPrev').onclick=()=>{ idxEl.value=Math.max(0,+idxEl.value-1); load(); };
document.getElementById('btnNext').onclick=()=>{ idxEl.value=+idxEl.value+1; load(); };
document.getElementById('btnLoad').onclick=load;
idxEl.addEventListener('keydown', e=>{ if(e.key==='Enter') load(); });

function load(){
  statusEl.textContent='Loading...';
  card.style.display='none';
  const params=new URLSearchParams({ds:dsEl.value, idx:idxEl.value});
  fetch('/api/sample?'+params).then(r=>r.json()).then(render).catch(e=>{ statusEl.textContent='Error: '+e; });
}

function render(d){
  statusEl.textContent = d.audio_status || '';
  if(d.error){ statusEl.textContent=d.error; return; }
  currentData=d;

  metaEl.innerHTML=`
    <span><b>ID</b> ${d.utterance_id||d.segment_id||'—'}</span>
    <span><b>Duration</b> ${d.audio_duration?.toFixed(2)}s</span>
    <span><b>Words</b> ${d.words?.length}</span>
    <span><b>Model</b> ${d.alignment_model||'—'}</span>
    ${d.dataset?`<span><b>Dataset</b> ${d.dataset}</span>`:''}
  `;

  if(d.audio_b64){
    player.src='data:audio/wav;base64,'+d.audio_b64;
    const ab=base64ToArrayBuffer(d.audio_b64);
    new AudioContext().decodeAudioData(ab).then(buf=>drawWave(buf)).catch(()=>{});
  } else {
    player.removeAttribute('src');
  }

  wordsEl.innerHTML='';
  (d.words||[]).forEach((w,i)=>{
    const el=document.createElement('span');
    el.className='word'; el.dataset.i=i;
    const scoreColor=w.score!=null ? hsl(w.score) : '#718096';
    el.innerHTML=`${w.score!=null?`<span class="score-bar" style="background:${scoreColor}"></span>`:''}${w.word}<span class="ts">${w.start.toFixed(2)}–${w.end.toFixed(2)}</span>`;
    el.onclick=()=>{ player.currentTime=w.start; player.play(); };
    wordsEl.append(el);
  });

  card.style.display='';
  player.ontimeupdate=()=>{ syncHighlight(player.currentTime); syncProgress(player.currentTime); };
  cancelAnimationFrame(animFrame);
  const tick=()=>{ if(!player.paused){ syncHighlight(player.currentTime); syncProgress(player.currentTime); } animFrame=requestAnimationFrame(tick); };
  tick();
}

function syncHighlight(t){
  const words=currentData?.words||[];
  wordsEl.querySelectorAll('.word').forEach((el,i)=>{
    const w=words[i];
    el.classList.toggle('active', t>=w.start && t<w.end+0.05);
  });
}

function syncProgress(t){
  const dur=currentData?.audio_duration||1;
  prog.style.left=(t/dur*100)+'%';
}

waveWrap.addEventListener('click', e=>{
  const rect=waveWrap.getBoundingClientRect();
  const frac=(e.clientX-rect.left)/rect.width;
  const dur=currentData?.audio_duration||1;
  player.currentTime=frac*dur;
});

function drawWave(buf){
  const canvas=document.getElementById('waveform');
  const dpr=window.devicePixelRatio||1;
  canvas.width=canvas.offsetWidth*dpr; canvas.height=60*dpr;
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  const data=buf.getChannelData(0);
  const W=canvas.offsetWidth, H=60, step=Math.ceil(data.length/W);
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#1a1d27'; ctx.fillRect(0,0,W,H);
  for(let x=0;x<W;x++){
    let max=0;
    for(let j=0;j<step;j++){ const v=Math.abs(data[x*step+j]||0); if(v>max)max=v; }
    const h=max*(H-4);
    ctx.fillStyle='#3182ce';
    ctx.fillRect(x,H/2-h/2,1,h);
  }
}

function base64ToArrayBuffer(b64){
  const bin=atob(b64); const buf=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) buf[i]=bin.charCodeAt(i);
  return buf.buffer;
}

function hsl(score){ const h=Math.round(score*120); return `hsl(${h},70%,50%)`; }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/datasets")
def api_datasets():
    return jsonify([k for k, v in DATASETS.items() if v.exists()])


@app.route("/api/total")
def api_total():
    ds = request.args.get("ds", "")
    path = DATASETS.get(ds)
    if not path or not path.exists():
        return jsonify({"error": "unknown dataset"}), 400
    return jsonify({"total": total_rows(path)})


@app.route("/api/sample")
def api_sample():
    ds = request.args.get("ds", "")
    idx = int(request.args.get("idx", 0))
    path = DATASETS.get(ds)
    if not path or not path.exists():
        return jsonify({"error": f"dataset not found: {ds}"}), 400

    row = get_row(path, idx)
    audio_b64, audio_status = row_to_wav_b64(ds, row)
    row_clean = {k: v for k, v in row.items() if k != "audio"}
    row_clean["audio_b64"] = audio_b64
    row_clean["audio_status"] = audio_status
    return jsonify(row_clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # HF 데이터셋 인덱스 백그라운드 빌드
    for ds_key in HF_SOURCES:
        if DATASETS.get(ds_key, Path()).exists():
            _index_status[ds_key] = "building"
            t = threading.Thread(target=_build_index, args=(ds_key,), daemon=True)
            t.start()

    print(f"Viewer: http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
