"""组件二：复核界面（FastAPI，中英双语）。

A 模式（带误差提示）显示模型距离与该区间的历史低质量帧比例；
B 模式（仅触发提示）只显示"已触发避障"。
复核者输入姓名后，按姓名哈希决定先看哪一组（顺序平衡）。

运行：python scripts/review_app.py  → http://127.0.0.1:8000
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "review" / "review_queue.jsonl"
REVIEWS = ROOT / "review" / "reviews.jsonl"
IMG_DIR = ROOT / "review" / "imgs"

# v4_imagenet error_analysis 的低质量帧占比（带误差提示组的先验，百分数）
BIN_PRIOR = {"0.5-2 m": 18, "2-5 m": 18, "5-10 m": 18,
             "10-20 m": 0, "20-40 m": 6, "40+ m": 0}

# 理由的存储值始终用中文原文，保证与已有记录一致；显示时按语言取标签
REASONS = ["玻璃反光", "天空无纹理", "障碍占满画面", "目标小/遮挡",
           "距离目标太近", "距离目标太远", "模糊过曝阴影", "其他"]

REASON_EN = {
    "玻璃反光": "Glass reflection",
    "天空无纹理": "Textureless sky",
    "障碍占满画面": "Obstacle fills frame",
    "目标小/遮挡": "Small / occluded target",
    "距离目标太近": "Too close",
    "距离目标太远": "Too far",
    "模糊过曝阴影": "Blur, glare or shadow",
    "其他": "Other",
}

T = {
    "zh": {
        "doc": "避障复核",
        "app": "无人机避障告警复核",
        "home_p": "看画面，判断这次避障触发是否可信。",
        "samples": "%d 条样本",
        "minutes": "约 6 分钟",
        "summary": "汇总",
        "start": "开始",
        "name_ph": "姓名或代号",
        "q": "这次触发可信吗？",
        "accept": "可信",
        "hold": "存疑",
        "reject": "不可信",
        "unknown": "无法判断",
        "why": "为什么？",
        "note_ph": "补充说明",
        "back": "返回",
        "confirm": "提交",
        "quit": "退出",
        "dl": "数据包",
        "dl_full": "含图片",
        "triggered": "已触发避障",
        "front": "前方",
        "prior": "低质量帧 %d%%",
        "g1": "第一组完成",
        "g1_to": "接下来：%s",
        "cont": "继续",
        "done": "全部完成",
        "thanks": "记录已保存，谢谢。",
        "out": "已退出",
        "out_p": "本机姓名已清除，已提交的数据保留。",
        "home": "返回首页",
        "sum_h": "复核汇总",
        "queue": "队列",
        "reviewed": "已复核",
        "reviewers": "复核人",
        "avg": "平均用时",
        "c_group": "分组",
        "c_n": "复核",
        "c_ok": "明确",
        "c_und": "存疑",
        "c_acc": "与真值一致",
        "mA": "带误差提示",
        "mB": "仅触发提示",
        "other_note": "选择「其他」时请填写备注",
        "sec": "秒",
        "tog": "EN",
    },
    "en": {
        "doc": "Review",
        "app": "Drone avoidance alert review",
        "home_p": "Look at the frame, then judge whether this avoidance trigger was justified.",
        "samples": "%d frames",
        "minutes": "about 6 min",
        "summary": "Summary",
        "start": "Start",
        "name_ph": "Name or initials",
        "q": "Was this trigger justified?",
        "accept": "Justified",
        "hold": "Unsure",
        "reject": "Not justified",
        "unknown": "Cannot tell",
        "why": "Why?",
        "note_ph": "Note",
        "back": "Back",
        "confirm": "Submit",
        "quit": "Quit",
        "dl": "Data",
        "dl_full": "With images",
        "triggered": "Avoidance triggered",
        "front": "Front",
        "prior": "low-quality %d%%",
        "g1": "Set 1 done",
        "g1_to": "Next: %s",
        "cont": "Continue",
        "done": "All done",
        "thanks": "Saved. Thank you.",
        "out": "Signed out",
        "out_p": "Name cleared on this device. Submitted reviews are kept.",
        "home": "Home",
        "sum_h": "Review summary",
        "queue": "Queue",
        "reviewed": "Reviewed",
        "reviewers": "Reviewers",
        "avg": "Avg time",
        "c_group": "Set",
        "c_n": "n",
        "c_ok": "Decided",
        "c_und": "Unsure",
        "c_acc": "Matches truth",
        "mA": "With error hint",
        "mB": "Trigger only",
        "other_note": "Pick a note when choosing Other",
        "sec": "s",
        "tog": "中文",
    },
}

items = [json.loads(l) for l in QUEUE.read_text(encoding="utf-8").splitlines()]
IMG_DIR.mkdir(parents=True, exist_ok=True)
for it in items:
    stem = it["item_id"].split("/")[-1]
    src = Path(it["image_dir"]) / (stem + "_front_rgb.png")
    dst = IMG_DIR / (it["item_id"].replace("/", "_") + ".png")
    if not dst.exists():
        shutil.copy(src, dst)

app = FastAPI(title="Drone Avoidance Review")
app.mount("/imgs", StaticFiles(directory=IMG_DIR), name="imgs")


def lang_of(request: Request) -> str:
    q = request.query_params.get("lang")
    if q in ("zh", "en"):
        return q
    return "en" if request.cookies.get("lang") == "en" else "zh"


def tr(lang: str) -> dict:
    return T[lang]


def mode_label(mode: str, lang: str) -> str:
    return tr(lang)["mA"] if mode == "A" else tr(lang)["mB"]


def lang_toggle(lang: str, path: str) -> str:
    """右上角的小切换：跳到另一种语言，再回到当前页。"""
    code = "zh" if lang == "en" else "en"
    return ("<a class='export' href='/lang/%s?next=%s'>%s</a>"
            % (code, quote(path, safe="/"), tr(lang)["tog"]))


def load_reviews() -> list[dict]:
    if not REVIEWS.exists():
        return []
    return [json.loads(line) for line in REVIEWS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_summary() -> dict:
    reviews = load_reviews()
    qmap = {it["item_id"]: it for it in items}
    by_mode = {}
    for mode in ("A", "B"):
        rs = [r for r in reviews if r.get("mode") == mode]
        decided = [r for r in rs if r.get("verdict") in ("accept", "reject")]
        ok = sum(
            (r["verdict"] == "accept")
            == (qmap.get(r["item_id"], {}).get("gt_dist", 9.0) < 9.0)
            for r in decided
        )
        by_mode[mode] = {
            "n": len(rs),
            "decided": len(decided),
            "undecided": len(rs) - len(decided),
            "accuracy": round(ok / len(decided), 3) if decided else None,
        }
    reasons = Counter()
    for r in reviews:
        for reason in r.get("reasons", []):
            reasons[reason] += 1
    secs = [r.get("seconds", 0) for r in reviews if r.get("seconds")]
    return {
        "queue_size": len(items),
        "review_count": len(reviews),
        "reviewed_items": len({r["item_id"] for r in reviews}),
        "reviewers": dict(Counter(r.get("reviewer", "unknown") for r in reviews)),
        "by_verdict": dict(Counter(r.get("verdict") for r in reviews)),
        "by_reason": dict(reasons),
        "by_mode": by_mode,
        "avg_seconds": round(sum(secs) / len(secs), 1) if secs else None,
    }


def image_index() -> tuple[dict, dict]:
    """按内容哈希建立图片索引；相同图片只保留一份。"""
    mapping, unique = {}, {}
    for it in items:
        fname = it["item_id"].replace("/", "_") + ".png"
        p = IMG_DIR / fname
        if not p.exists():
            continue
        digest = hashlib.sha1(p.read_bytes()).hexdigest()
        mapping[it["item_id"]] = {"file": fname, "sha1": digest}
        unique.setdefault(digest, p)
    return mapping, unique


def make_export(include_images: bool) -> io.BytesIO:
    mapping, unique = image_index()
    if include_images:
        indexed = {
            k: {"file": v["sha1"][:12] + Path(v["file"]).suffix, "sha1": v["sha1"]}
            for k, v in mapping.items()
        }
    else:
        indexed = {
            k: {"source": v["file"], "sha1": v["sha1"]}
            for k, v in mapping.items()
        }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("reviews.jsonl", "\n".join(
            json.dumps(r, ensure_ascii=False) for r in load_reviews()))
        z.writestr("review_queue.jsonl", "\n".join(
            json.dumps(it, ensure_ascii=False) for it in items))
        z.writestr("summary.json", json.dumps(build_summary(), ensure_ascii=False, indent=2))
        z.writestr("image_index.json", json.dumps(indexed, ensure_ascii=False, indent=2))
        if include_images:
            for digest, p in unique.items():
                z.write(p, "imgs/" + digest[:12] + p.suffix)
    buf.seek(0)
    return buf


sessions: dict[str, dict] = {}   # reviewer -> 会话状态

CSS = """
*{box-sizing:border-box}
:root{--ink:#1a1d21;--sub:#5f6875;--line:#e2e6ea;
      --accent:#3a6b8c;--accent-soft:#e8f0f6;--accent-dark:#305d7c;--warm:#a8452f}
html,body{height:100%}
body{font-family:"Source Han Sans","Microsoft YaHei",-apple-system,sans-serif;background:#fff;
     color:var(--ink);height:100vh;margin:0;overflow:hidden;font-size:14px;
     -webkit-font-smoothing:antialiased}
.page{max-width:1180px;width:100%;height:100%;margin:0 auto;padding:18px 20px;
      display:flex;flex-direction:column}
h2{font-size:20px;font-weight:600;margin:0 0 6px;letter-spacing:.01em}
.dim{color:var(--sub);font-size:13px;line-height:1.7;margin:0}
button,input{font-family:inherit}
button{cursor:pointer;border:1px solid var(--line);background:#f6f8fa;color:var(--ink);
       border-radius:6px;transition:background .15s,border-color .15s,transform .05s}
button:hover{background:#eef2f6;border-color:#cfd6dd}
button:active{transform:translateY(1px)}
button:focus-visible,a:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* 首页 */
.hero{max-width:560px;margin:12vh auto 0;padding:30px 32px;border:1px solid var(--line);
      border-radius:12px;background:#fff;box-shadow:0 8px 30px rgba(28,38,48,.05)}
.eyebrow{font-size:11px;letter-spacing:.14em;color:var(--accent);font-weight:600;margin-bottom:12px}
.hero h2{font-size:25px;margin-bottom:10px;line-height:1.35}
.hero .dim{margin-bottom:20px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 20px}
.chip{font-size:12px;color:var(--sub);background:#f4f6f8;border-radius:999px;padding:5px 11px}
.name_row{display:flex;gap:10px;align-items:center}
.name_row input{flex:1;padding:12px 13px;font-size:14px;border:1px solid var(--line);
                border-radius:6px;min-width:0}
.name_row button{padding:12px 24px;font-size:14px;background:var(--accent);
                 border-color:var(--accent);color:#fff}
.name_row button:hover{background:var(--accent-dark)}
.links{display:flex;gap:8px;margin-top:18px;flex-wrap:wrap;align-items:center}
.export{display:inline-block;padding:6px 11px;font-size:12px;border:1px solid var(--line);
        border-radius:5px;background:#f7f9fa;color:var(--sub);text-decoration:none;white-space:nowrap}
.export:hover{background:#eef2f6;color:var(--ink)}

/* 复核页 */
.top{flex:0 0 auto;margin-bottom:14px}
.top_line{display:flex;justify-content:space-between;align-items:center;gap:12px}
.prog{font-size:12px;color:var(--sub);display:flex;align-items:center;gap:8px;min-width:0}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600;
       background:#eef2f6;color:var(--sub);white-space:nowrap}
.badge.a{background:var(--accent-soft);color:var(--accent)}
.top_actions{display:flex;gap:6px;flex:0 0 auto}
.top h2{margin:9px 0 0;font-size:18px}
.progress_track{height:3px;background:#edf0f3;border-radius:2px;margin-top:11px;overflow:hidden}
.progress_fill{height:100%;background:var(--accent);border-radius:2px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;flex:1;min-height:0}
.stage{min-height:0;min-width:0;display:flex;align-items:center;justify-content:center;
       background:#f7f9fa;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.stage img{width:100%;height:100%;object-fit:contain;display:block;background:#fff}
.stage > img{cursor:zoom-in}
/* 点画面放大到全屏，再点收回；手机上看不清细节时用 */
.zoom{position:fixed;inset:0;z-index:60;background:#0f1317;display:none;
      align-items:center;justify-content:center;padding:0}
.zoom.on{display:flex}
.zoom img{width:100%;height:100%;object-fit:contain;cursor:zoom-out}
.side{min-height:0;display:flex;flex-direction:column;gap:14px;overflow-y:auto;overflow-x:hidden;
      align-self:start;max-height:100%;padding:16px;background:#fbfcfd;
      border:1px solid var(--line);border-radius:10px}
.side form{display:flex;flex-direction:column;gap:12px;flex:0 0 auto}

/* 结论卡：数字说话 */
.claim{padding:16px 18px;background:#fff;border:1px solid var(--line);
       border-left:3px solid var(--accent);border-radius:8px;flex:0 0 auto}
.claim .lab{font-size:11px;letter-spacing:.08em;color:var(--sub);text-transform:uppercase}
.claim .num{font-size:34px;font-weight:600;line-height:1.05;margin:4px 0 2px;
            font-variant-numeric:tabular-nums}
.claim .num span{font-size:16px;font-weight:400;margin-left:2px;color:var(--sub)}
.claim .tag{font-size:15px;font-weight:600;color:var(--accent)}
.claim .tag.plain{font-size:19px;margin-top:6px}
.prior{display:flex;align-items:center;gap:8px;margin-top:11px}
.prior .track{flex:1;height:5px;background:#edf0f3;border-radius:3px;overflow:hidden}
.prior .track i{display:block;height:100%;background:var(--warm);border-radius:3px}
.prior span{font-size:11px;color:var(--sub);white-space:nowrap}

.actions{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;flex:0 0 auto}
.actions button{padding:15px 6px;font-size:14px;min-height:58px;line-height:1.25}
.actions kbd{display:block;font-size:10px;color:#98a1ab;margin-top:3px}
/* 已经选过一次之后，四个按钮压成一条，把高度让给理由列表 */
.actions.compact{grid-template-columns:repeat(2,1fr);gap:6px}
.actions.compact button{min-height:34px;padding:7px 4px;font-size:12.5px;background:#fff}
.actions.compact button.selected{background:var(--accent);border-color:var(--accent);color:#fff}
.actions.compact kbd{display:none}
button.selected{background:var(--accent);border-color:var(--accent);color:#fff}
button.selected kbd{color:#cfe0ec}

.reasons{padding:14px;background:#fff;border:1px solid var(--line);border-radius:8px;flex:0 0 auto;
         display:grid;grid-template-columns:1fr;column-gap:10px;align-content:start}
/* 中文标签短，两列更紧凑；英文标签长，单列才不折行 */
.reasons.two{grid-template-columns:1fr 1fr}
.reasons .why{grid-column:1 / -1;font-size:13px;font-weight:600;margin-bottom:2px}
.reasons label{display:flex;align-items:center;gap:7px;width:100%;margin:0;padding:6px 2px;
               font-size:13px;color:#303943;cursor:pointer;border-radius:4px}
.reasons label:hover{background:#f5f7f9}
.reasons input[type=radio]{accent-color:var(--accent);margin:0;flex:0 0 auto}
.note{grid-column:1 / -1;width:100%;margin-top:8px;padding:8px 10px;border:1px solid var(--line);
      border-radius:5px;font-size:13px}
.reason_actions{grid-column:1 / -1;display:flex;gap:8px;margin-top:10px}
.reason_actions button{flex:1;padding:10px;font-size:13px}
.reason_actions .primary{background:var(--accent);border-color:var(--accent);color:#fff}
.reason_actions .primary:hover{background:var(--accent-dark)}

/* 汇总与完成页 */
.panel{max-width:620px;padding:26px 28px;border:1px solid var(--line);border-radius:12px;
       background:#fff;margin-top:8vh;box-shadow:0 8px 30px rgba(28,38,48,.05)}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:16px}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--sub);background:#f7f9fa;font-size:12px}
td.n{font-variant-numeric:tabular-nums}

@media (max-width:760px){
  /* 手机上一屏装不下「画面 + 结论 + 四个按钮 + 八个理由」，
     所以不再锁屏：画面按宽度满幅显示，其余内容自然向下滚动。 */
  html,body{height:auto}
  body{height:auto;overflow-y:auto;overflow-x:hidden;min-height:100svh}
  .page{padding:10px 12px;height:auto;min-height:100svh}
  .hero,.panel{margin-top:4vh;padding:22px}
  .hero h2{font-size:21px}
  .name_row{flex-direction:column;align-items:stretch}
  .name_row button{width:100%}
  .top{flex:0 0 auto;margin-bottom:10px}
  .top h2{font-size:15px;margin-top:7px}
  .prog{font-size:11px}
  .layout{display:flex;flex-direction:column;flex:0 0 auto;gap:10px}
  .stage{flex:0 0 auto;align-self:stretch}
  .stage > img{width:100%;height:auto;max-height:46svh}
  .side{padding:12px;gap:10px;min-width:0;overflow:visible;align-self:stretch;max-height:none}
  .claim{min-width:0}
  .actions button{white-space:normal}
  .claim{padding:11px 14px}
  .claim .num{font-size:26px}
  .claim .tag.plain{font-size:17px}
  .actions{gap:7px}
  .actions button{padding:11px 4px;min-height:48px;font-size:13px}
  .reasons{padding:12px 11px;grid-template-columns:1fr;column-gap:0}
  .reasons.two{grid-template-columns:1fr}
  .reasons label{padding:9px 2px;font-size:13.5px;gap:10px}
  .reasons input[type=radio]{width:17px;height:17px}
  .note{margin-top:6px;padding:10px;font-size:16px}
  .reason_actions{margin-top:8px}
  .reason_actions button{padding:13px;font-size:14px}
  .export{padding:5px 9px;font-size:11px}
}
@media (max-height:700px){
  .page{padding:10px 12px}
  .top{margin-bottom:8px}
  .top h2{display:none}
  .side{padding:12px;gap:9px}
  .claim{padding:11px 14px}
  .claim .num{font-size:26px}
  .actions button{padding:10px 4px;min-height:46px}
}
"""

JS_REVIEW = """
function choose(v){
  document.getElementById('verdict').value=v;
  var btns=document.querySelectorAll('.actions button');
  for(var i=0;i<btns.length;i++){btns[i].classList.remove('selected')}
  document.getElementById('btn-'+v).classList.add('selected');
  if(v==='accept'){document.getElementById('f').submit();return}
  var box=document.getElementById('rbox');
  box.style.display='grid';   /* 不能用 block：会盖掉 CSS 里的两列网格 */
  document.querySelector('.actions').classList.add('compact');
  if(window.matchMedia('(max-width:760px)').matches){
    box.scrollIntoView({block:'nearest',behavior:'smooth'});
  }
}
function cancel(){
  document.getElementById('rbox').style.display='none';
  document.getElementById('verdict').value='';
  var btns=document.querySelectorAll('.actions button');
  for(var i=0;i<btns.length;i++){btns[i].classList.remove('selected')}
  document.querySelector('.actions').classList.remove('compact');
}
function zoomIn(e){
  var z=document.getElementById('zoom');
  if(!z)return;
  z.querySelector('img').src=e.target.src;
  z.classList.add('on');
}
function zoomOut(){
  var z=document.getElementById('zoom');
  if(z)z.classList.remove('on');
}
document.getElementById('f').addEventListener('submit',function(e){
  var other=document.querySelector('input[name="reasons"][value="其他"]');
  if(other&&other.checked&&!document.querySelector('.note').value.trim()){
    alert(NOTE_MSG);e.preventDefault();
  }
});
var t0=Date.now();
document.getElementById('secs').value=0;
setInterval(function(){document.getElementById('secs').value=Math.round((Date.now()-t0)/1000)},500);
document.addEventListener('keydown',function(e){
  if(e.key==='Escape'){zoomOut()}
  if(['INPUT','TEXTAREA'].indexOf(e.target.tagName)>=0)return;
  if(e.key==='1'){choose('accept')}
  if(e.key==='2'){choose('hold')}
  if(e.key==='3'){choose('reject')}
  if(e.key==='4'){choose('unknown')}
});
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        "<html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>%s</title><style>%s</style></head>"
        "<body><div class='page'>%s</div></body></html>" % (title, CSS, body),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/lang/{code}")
def set_lang(code: str, next: str = "/"):
    if not next.startswith("/") or next.startswith("//"):
        next = "/"
    r = RedirectResponse(next, 302)
    r.set_cookie("lang", "en" if code == "en" else "zh",
                 max_age=86400 * 180, samesite="lax")
    return r


@app.get("/")
def index(request: Request):
    lang = lang_of(request)
    t = tr(lang)
    body = """
    <div class="hero">
      <div class="eyebrow">HUMAN-IN-THE-LOOP · 01</div>
      <h2>%s</h2>
      <p class="dim">%s</p>
      <div class="chips">
        <span class="chip">%s</span>
        <span class="chip">%s</span>
      </div>
      <form class="name_row" action="/start" method="post"
            onsubmit="localStorage.setItem('reviewer_name',this.reviewer.value.trim())">
        <input type="text" name="reviewer" placeholder="%s" required>
        <button type="submit">%s</button>
      </form>
      <div class="links">
        <a class="export" href="/summary">%s</a>
        %s
      </div>
    </div>
    <script>
    (function(){
      var n=localStorage.getItem('reviewer_name');
      if(!n)return;
      var f=document.createElement('form');
      f.method='post';f.action='/start';
      var i=document.createElement('input');
      i.type='hidden';i.name='reviewer';i.value=n;
      f.appendChild(i);document.body.appendChild(f);f.submit();
    })();
    </script>
    """ % (t["app"], t["home_p"], t["samples"] % len(items), t["minutes"],
           t["name_ph"], t["start"], t["summary"], lang_toggle(lang, "/"))
    return page(t["app"], body)


@app.get("/summary")
def summary(request: Request):
    lang = lang_of(request)
    t = tr(lang)
    s = build_summary()
    rows = "".join(
        "<tr><td>%s</td><td class='n'>%s</td><td class='n'>%s</td>"
        "<td class='n'>%s</td><td class='n'>%s</td></tr>"
        % (mode_label(mode, lang), d["n"], d["decided"], d["undecided"],
           "—" if d["accuracy"] is None else "%.0f%%" % (100 * d["accuracy"]))
        for mode, d in s["by_mode"].items()
    )
    stats = " · ".join([
        "%s %d" % (t["queue"], s["queue_size"]),
        "%s %d" % (t["reviewed"], s["review_count"]),
        "%s %d" % (t["reviewers"], len(s["reviewers"])),
        "%s %s%s" % (t["avg"],
                     "—" if s["avg_seconds"] is None else s["avg_seconds"],
                     "" if s["avg_seconds"] is None else t["sec"]),
    ])
    body = """
    <div class="panel">
      <div class="eyebrow">SUMMARY</div>
      <h2>%s</h2>
      <p class="dim">%s</p>
      <table>
        <tr><th>%s</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th></tr>
        %s
      </table>
      <div class="links">
        <a class="export" href="/">%s</a>
        <a class="export" href="/export">%s</a>
        <a class="export" href="/export_full">%s</a>
        <a class="export" href="/logout">%s</a>
        %s
      </div>
    </div>
    """ % (t["sum_h"], stats, t["c_group"], t["c_n"], t["c_ok"], t["c_und"],
           t["c_acc"], rows, t["home"], t["dl"], t["dl_full"], t["quit"],
           lang_toggle(lang, "/summary"))
    return page(t["sum_h"], body)


def _zip_response(buf: io.BytesIO, prefix: str) -> StreamingResponse:
    name = prefix + time.strftime("%Y%m%d_%H%M") + ".zip"
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="%s"' % name},
    )


@app.get("/export")
def export():
    return _zip_response(make_export(False), "review_data_")


@app.get("/export_full")
def export_full():
    return _zip_response(make_export(True), "review_data_with_images_")


@app.post("/start")
def start(reviewer: str = Form(...)):
    name = reviewer.strip()
    if not name:
        return RedirectResponse("/", 302)
    # 姓名哈希决定模式顺序，奇偶平衡
    if name not in sessions:
        order = "AB" if sum(map(ord, name)) % 2 == 0 else "BA"
        sessions[name] = {"order": order, "phase": 0, "pos": 0, "t0": time.time()}
    return RedirectResponse("/review/" + name, 302)


@app.get("/logout")
def logout_current(request: Request):
    lang = lang_of(request)
    t = tr(lang)
    body = """
    <div class="panel">
      <div class="eyebrow">SIGNED OUT</div>
      <h2>%s</h2>
      <p class="dim">%s</p>
      <div class="links"><a class="export" href="/">%s</a>%s</div>
    </div>
    <script>localStorage.removeItem('reviewer_name');</script>
    """ % (t["out"], t["out_p"], t["home"], lang_toggle(lang, "/logout"))
    return page(t["out"], body)


@app.get("/logout/{name}")
def logout(name: str, request: Request):
    sessions.pop(name, None)
    return logout_current(request)


def cur_items(name):
    s = sessions[name]
    mode = s["order"][s["phase"]]
    return [it for it in items if it["mode"] == mode]


@app.get("/review/{name}")
def review(name: str, request: Request):
    if name not in sessions:
        return RedirectResponse("/", 302)
    lang = lang_of(request)
    t = tr(lang)
    s = sessions[name]
    pool = cur_items(name)
    path = "/review/" + name

    # 仅用于论文配图：?peek=N 直接渲染第 N 条（浅拷贝，不改会话状态），
    # ?reasons=1 让理由面板初始就展开（正常流程里它要等判定之后才出现）。
    # 两个参数都不写数据库，也不影响真实评审。
    show_reasons = request.query_params.get("reasons") == "1"
    peek = request.query_params.get("peek")
    if peek is not None and pool:
        try:
            s = dict(s)
            s["pos"] = max(0, min(int(peek), len(pool) - 1))
        except ValueError:
            pass

    if s["pos"] >= len(pool):
        if s["phase"] == 0:
            s["phase"] = 1
            s["pos"] = 0
            nxt = mode_label(s["order"][1], lang)
            body = """
            <div class="panel">
              <div class="eyebrow">1 / 2</div>
              <h2>%s</h2>
              <p class="dim">%s</p>
              <div class="links">
                <a href="/review/%s"><button style="flex:none;padding:11px 26px">%s</button></a>
                %s
              </div>
            </div>
            """ % (t["g1"], t["g1_to"] % nxt, quote(name, safe=""), t["cont"],
                   lang_toggle(lang, path))
            return page(t["g1"], body)
        body = """
        <div class="panel">
          <div class="eyebrow">2 / 2</div>
          <h2>%s</h2>
          <p class="dim">%s</p>
          <div class="links">
            <a class="export" href="/summary">%s</a>
            <a class="export" href="/export">%s</a>
            <a class="export" href="/logout/%s">%s</a>
            %s
          </div>
        </div>
        """ % (t["done"], t["thanks"], t["summary"], t["dl"],
               quote(name, safe=""), t["quit"], lang_toggle(lang, path))
        return page(t["done"], body)

    it = pool[s["pos"]]
    img = "/imgs/" + it["item_id"].replace("/", "_") + ".png"

    if it["mode"] == "A":
        prior = BIN_PRIOR.get(it["bin_label"], 0)
        claim = (
            "<div class='claim'>"
            "<div class='lab'>%s</div>"
            "<div class='num'>%.1f<span>m</span></div>"
            "<div class='tag'>%s</div>"
            "<div class='prior'><div class='track'><i style='width:%d%%'></i></div>"
            "<span>%s</span></div>"
            "</div>"
            % (t["front"], it["pred_dist"], t["triggered"],
               prior, t["prior"] % prior)
        )
    else:
        claim = ("<div class='claim'><div class='tag plain'>%s</div></div>"
                 % t["triggered"])

    reasons = "".join(
        "<label><input type='radio' name='reasons' value='%s'>%s</label>"
        % (r, REASON_EN[r] if lang == "en" else r) for r in REASONS
    )
    progress_pct = 100.0 * (s["pos"] + 1) / len(pool)
    note_msg = t["other_note"].replace("'", "\\'")

    body = f"""
    <div class="top">
      <div class="top_line">
        <div class="prog">
          <span class="badge {'a' if it['mode'] == 'A' else ''}">{mode_label(it['mode'], lang)}</span>
          {name} · {s['pos'] + 1}/{len(pool)}
        </div>
        <div class="top_actions">
          <a class="export" href="/logout/{quote(name, safe='')}">{t['quit']}</a>
          {lang_toggle(lang, path)}
        </div>
      </div>
      <h2>{t['q']}</h2>
      <div class="progress_track"><div class="progress_fill" style="width:{progress_pct:.1f}%"></div></div>
    </div>
    <div class="layout">
      <div class="stage"><img src="{img}" alt="" onclick="zoomIn(event)"></div>
      <div class="side">
        {claim}
        <form id="f" action="/submit/{name}" method="post">
          <input type="hidden" name="item_id" value="{it["item_id"]}">
          <input type="hidden" name="mode" value="{it["mode"]}">
          <input type="hidden" name="seconds" value="0" id="secs">
          <input type="hidden" name="verdict" id="verdict" value="">
          <div class="actions">
            <button type="button" id="btn-accept" onclick="choose('accept')">{t['accept']}<kbd>1</kbd></button>
            <button type="button" id="btn-hold" onclick="choose('hold')">{t['hold']}<kbd>2</kbd></button>
            <button type="button" id="btn-reject" onclick="choose('reject')">{t['reject']}<kbd>3</kbd></button>
            <button type="button" id="btn-unknown" onclick="choose('unknown')">{t['unknown']}<kbd>4</kbd></button>
          </div>
          <div class="reasons{' two' if lang == 'zh' else ''}" id="rbox" style="display:{'grid' if show_reasons else 'none'}">
            <div class="why">{t['why']}</div>
            {reasons}
            <input class="note" type="text" name="note" placeholder="{t['note_ph']}">
            <div class="reason_actions">
              <button type="button" onclick="cancel()">{t['back']}</button>
              <button type="submit" class="primary">{t['confirm']}</button>
            </div>
          </div>
        </form>
      </div>
    </div>
    <div class="zoom" id="zoom" onclick="zoomOut()"><img src="" alt=""></div>
    <script>var NOTE_MSG='{note_msg}';{JS_REVIEW}</script>
    """
    return page("%s · %s" % (t["doc"], name), body)


@app.post("/submit/{name}")
def submit(name: str, item_id: str = Form(...), mode: str = Form(...),
           verdict: str = Form(...), seconds: int = Form(0),
           reasons: list[str] = Form([]), note: str = Form("")):
    sessions.setdefault(name, {"order": "AB", "phase": 0, "pos": 0, "t0": time.time()})
    rec = {"reviewer": name, "item_id": item_id, "mode": mode, "verdict": verdict,
           "reasons": reasons, "note": note, "seconds": seconds, "ts": int(time.time())}
    with REVIEWS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sessions[name]["pos"] += 1
    return RedirectResponse("/review/" + name, 302)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
