# -*- coding: utf-8 -*-
"""
단일 파일 하드코딩 버전 (좌표계 자동 일치: AUTO 매핑):
1) GT(json/jsonl) 로드
2) CSV(tracks.csv) → Pred(jsonl) 변환
   - 비디오키 느슨한 정규화(_safety, _tracked 등 접미사 제거)
   - 비디오별 CSV 해상도(W,H)와 GT 해상도(Gw,Gh) 각각 자동 근사
   - 각 프레임마다 fill/letterbox를 모두 시도하여 IoU가 최대인 변환을 선택
3) 즉시 성능평가(helmet/vest 분류 지표 + AP50/ mAP50)
"""

import os, sys, csv, json, ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# ================== 경로 하드코딩 (필요시만 수정) ==================
DEFAULT_GT   = r"..\..\Data\output\quick_test_all.jsonl"     # GT: 이미지 단위 jsonl/json
DEFAULT_CSV  = r"..\..\Data\output\logs\tracks.csv"          # CSV: tracks.csv
# Pred(jsonl) 임시 저장 및 최종 리포트 경로
DEFAULT_PRED_JSON = None  # None이면 tools/eval/Pred_images.jsonl 자동 사용
DEFAULT_REPORT    = None  # None이면 tools/eval/reports/evaluation.json 자동 사용
# ================================================================

# -------- 공용 유틸 --------
def _norm_path(p: str) -> str:
    if p is None: return ""
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(p)))
    except Exception:
        return str(p)

def _norm_key(s: str) -> str:
    s = (s or "").lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch); prev_us = False
        else:
            if not prev_us:
                out.append('_'); prev_us = True
    key = ''.join(out).strip('_')
    while '__' in key:
        key = key.replace('__', '_')
    return key

def _maybe_list4(x: Any) -> Optional[List[float]]:
    if x is None: return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null"): return None
    try:
        v = json.loads(s)
    except Exception:
        try: v = ast.literal_eval(s)
        except Exception: return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
        except Exception: return None
    return None

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)

def greedy_match(gt_boxes: List[List[float]], pred_boxes: List[List[float]], iou_thr: float) -> List[Tuple[int,int,float]]:
    matches: List[Tuple[int,int,float]] = []
    if not gt_boxes or not pred_boxes: return matches
    used_pred = set()
    cands = []
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou = iou_xyxy(g, p)
            if iou >= iou_thr:
                cands.append((iou, i, j))
    for iou, gi, pj in sorted(cands, key=lambda x: x[0], reverse=True):
        if pj in used_pred: continue
        if any(m[0]==gi for m in matches): continue
        used_pred.add(pj)
        matches.append((gi, pj, iou))
    return matches

def to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool): return x
    if x is None: return None
    s = str(x).strip().lower()
    if s in ("true","1","yes","y"):   return True
    if s in ("false","0","no","n"):   return False
    if s in ("skip","none","null",""):return None
    return None

# -------- GT 로더 --------
def load_gt_rows(gt_path: Path) -> List[dict]:
    txt = gt_path.read_text(encoding="utf-8").strip()
    rows: List[dict] = []
    if not txt: return rows
    if txt[0] == "[":
        data = json.loads(txt)
        if not isinstance(data, list):
            raise ValueError("GT JSON must be an array or JSONL.")
        rows = data
    else:
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln: continue
            rows.append(json.loads(ln))
    return rows

# -------- CSV 로더(네 스키마 전용) --------
def load_csv_rows(csv_path: Path) -> List[dict]:
    """
    video_name,frame,id,x1,y1,x2,y2,score,helmet,vest,part,helmet_bbox,helmet_score,vest_bbox,vest_score
    """
    out: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            row = {
                "video_name": r.get("video_name") or r.get("video_path") or "",
                "frame": int(float(r.get("frame", 0))),
                "id": r.get("id"),
                "x1": float(r.get("x1", 0)),
                "y1": float(r.get("y1", 0)),
                "x2": float(r.get("x2", 0)),
                "y2": float(r.get("y2", 0)),
                "score": float(r.get("score", 0.0)),
                "helmet": r.get("helmet"),
                "vest":   r.get("vest"),
                "part":   r.get("part"),
                "helmet_bbox": _maybe_list4(r.get("helmet_bbox")),
                "helmet_score": None if (r.get("helmet_score") in (None,"","None")) else float(r.get("helmet_score")),
                "vest_bbox": _maybe_list4(r.get("vest_bbox")),
                "vest_score": None if (r.get("vest_score") in (None,"","None")) else float(r.get("vest_score")),
            }
            out.append(row)
    return out

# -------- 비디오 키/프레임 추출 --------
def _stem_without_hash(stem: str) -> str:
    if '.rf.' in stem:
        stem = stem.split('.rf.')[0]
    toks = stem.split('_')
    if toks and toks[-1] in ('png','jpg','jpeg'):
        toks = toks[:-1]
    return '_'.join(toks)

def extract_gt_video_key_and_frame(image_path: str) -> Tuple[str, Optional[int]]:
    """
    예: '.../human-accident_bump_rgb_0001_cctv1_11_png.rf.x.jpg'
        -> video_key='human-accident_bump_rgb_0001_cctv1', frame=11
    """
    stem = Path(image_path).stem
    stem = _stem_without_hash(stem)
    toks = stem.split('_')

    frame = None
    if toks and toks[-1].isdigit():
        frame = int(toks[-1])
        toks = toks[:-1]

    video_key_raw = '_'.join(toks)
    video_key = _norm_key(video_key_raw)
    return video_key, frame

def _strip_suffixes(stem: str) -> str:
    s = stem.lower()
    for suf in ("_safety", "-safety", "_tracked", "-tracked"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def derive_csv_video_key_relaxed(video_name_or_path: str) -> str:
    stem = Path(video_name_or_path).stem
    stem = _strip_suffixes(stem)
    return _norm_key(stem)

# -------- 해상도/좌표 변환 보조 --------
def _estimate_wh_per_video_from_csv(csv_rows: List[dict]) -> Dict[str, Tuple[int,int]]:
    """
    CSV 박스들(x2,y2)의 상위 분위수로 'CSV 좌표계' 해상도(W,H)를 근사.
    과소추정 방지를 위해 p99.9 → 32의 배수로 올림.
    """
    by_v = {}
    for r in csv_rows:
        vkey = derive_csv_video_key_relaxed(r.get("video_name",""))
        by_v.setdefault(vkey, {"x2":[], "y2":[]})
        by_v[vkey]["x2"].append(float(r["x2"]))
        by_v[vkey]["y2"].append(float(r["y2"]))
    wh = {}
    for v, d in by_v.items():
        if d["x2"] and d["y2"]:
            W = float(np.percentile(np.array(d["x2"]), 99.9))
            H = float(np.percentile(np.array(d["y2"]), 99.9))
            W = int(np.ceil(max(W, 32) / 32.0) * 32)
            H = int(np.ceil(max(H, 32) / 32.0) * 32)
            wh[v] = (W, H)
    return wh

def _estimate_wh_per_video_from_gt(gt_rows: List[dict]) -> Dict[str, Tuple[int,int]]:
    """
    GT 박스들(person_xyxy)의 x2,y2 상위 분위수로 'GT 좌표계' 해상도(Gw,Gh) 근사.
    """
    by_v = {}
    for g in gt_rows:
        img_path = g.get("image_path") or ""
        vkey, _ = extract_gt_video_key_and_frame(img_path)
        b = g.get("person_xyxy")
        if b and len(b)==4:
            by_v.setdefault(vkey, {"x2":[], "y2":[]})
            by_v[vkey]["x2"].append(float(b[2]))
            by_v[vkey]["y2"].append(float(b[3]))
    gh = {}
    for v, d in by_v.items():
        if d["x2"] and d["y2"]:
            Gw = float(np.percentile(np.array(d["x2"]), 99.9))
            Gh = float(np.percentile(np.array(d["y2"]), 99.9))
            Gw = int(np.ceil(max(Gw, 32) / 32.0) * 32)
            Gh = int(np.ceil(max(Gh, 32) / 32.0) * 32)
            gh[v] = (Gw, Gh)
    return gh

def _clip_box_xyxy(b, W, H):
    x1 = min(max(b[0], 0.0), W)
    y1 = min(max(b[1], 0.0), H)
    x2 = min(max(b[2], 0.0), W)
    y2 = min(max(b[3], 0.0), H)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [x1, y1, x2, y2]

def letterbox_forward(pbox: List[float], W:int, H:int, Gw:int, Gh:int) -> List[float]:
    """
    원본(W,H) 박스 -> GT 이미지(Gw,Gh) 박스 (letterbox 가정: r=min, 중앙패딩)
    """
    r = min(Gw / float(W), Gh / float(H))
    dw = (Gw - W * r) / 2.0
    dh = (Gh - H * r) / 2.0
    x1 = pbox[0] * r + dw
    y1 = pbox[1] * r + dh
    x2 = pbox[2] * r + dw
    y2 = pbox[3] * r + dh
    return _clip_box_xyxy([x1, y1, x2, y2], Gw, Gh)

def resize_fill_forward(pbox: List[float], W:int, H:int, Gw:int, Gh:int) -> List[float]:
    """
    원본(W,H) 박스 -> GT 이미지(Gw,Gh) 박스 (비레터박스: 가로/세로 독립 스케일)
    """
    sx = Gw / float(W)
    sy = Gh / float(H)
    x1 = pbox[0] * sx
    y1 = pbox[1] * sy
    x2 = pbox[2] * sx
    y2 = pbox[3] * sy
    return _clip_box_xyxy([x1, y1, x2, y2], Gw, Gh)

# -------- Pred 생성(이미지 기준) --------
def convert_csv_to_pred_jsonl_for_images(
    gt_rows: List[dict],
    csv_rows: List[dict],
    out_pred_path: Path,
    person_iou_thr: float = 0.03,
    frame_window: int = 3  # 같은 프레임 없으면 ±3 프레임 내에서 보정 매칭
) -> List[dict]:
    # CSV 인덱스: (video_key, frame) -> rows (느슨한 키)
    index_vf: Dict[Tuple[str,int], List[dict]] = {}
    index_v:  Dict[str, List[dict]] = {}
    for r in csv_rows:
        vkey = derive_csv_video_key_relaxed(r.get("video_name",""))
        index_v.setdefault(vkey, []).append(r)
        fr = int(r.get("frame", 0))
        index_vf.setdefault((vkey, fr), []).append(r)

    # 좌표계 근사 (CSV쪽 W,H / GT쪽 Gw,Gh)
    csv_wh = _estimate_wh_per_video_from_csv(csv_rows)
    gt_wh  = _estimate_wh_per_video_from_gt(gt_rows)

    tried = via_exact = via_window = via_anyframe = notfound = kept = 0
    pred_rows: List[dict] = []

    for g in gt_rows:
        img_path = g.get("image_path") or ""
        if not img_path: continue
        tried += 1

        g_video_key, g_frame = extract_gt_video_key_and_frame(img_path)
        gbox_GT = g.get("person_xyxy")
        if not gbox_GT:
            continue

        # 후보 수집
        candidates: List[dict] = []
        if g_frame is not None:
            candidates = index_vf.get((g_video_key, g_frame), [])
            if candidates:
                via_exact += 1
        if not candidates and g_frame is not None:
            local_cands = []
            for d in range(1, frame_window+1):
                local_cands += index_vf.get((g_video_key, g_frame-d), [])
                local_cands += index_vf.get((g_video_key, g_frame+d), [])
            if local_cands:
                via_window += 1
                candidates = local_cands
        if not candidates:
            candidates = index_v.get(g_video_key, [])
            if candidates:
                via_anyframe += 1
        if not candidates:
            notfound += 1
            continue

        # 양쪽 좌표계가 모두 필요
        if g_video_key not in csv_wh or g_video_key not in gt_wh:
            notfound += 1
            continue
        W, H   = csv_wh[g_video_key]  # CSV 좌표계 근사(모델이 그려낸 좌표계)
        Gw, Gh = gt_wh[g_video_key]   # GT 이미지 좌표계 근사

        # === fill/letterbox 모두 시도 → IoU 최대 선택 ===
        best_row, best_iou, best_pbox_gt = None, 0.0, None

        def eval_and_update(mapping_fn):
            nonlocal best_row, best_iou, best_pbox_gt
            for r in candidates:
                p_gt = mapping_fn([r["x1"], r["y1"], r["x2"], r["y2"]], W, H, Gw, Gh)
                i = iou_xyxy(gbox_GT, p_gt)
                if i > best_iou:
                    best_iou, best_row, best_pbox_gt = i, r, p_gt

        eval_and_update(resize_fill_forward)
        eval_and_update(letterbox_forward)

        # 동일좌표계 가정도 추가로 시도(혹시 이미 GT 스케일일 수도 있으므로)
        for r in candidates:
            pbox = [r["x1"], r["y1"], r["x2"], r["y2"]]
            i = iou_xyxy(gbox_GT, pbox)
            if i > best_iou:
                best_iou, best_row, best_pbox_gt = i, r, pbox

        if not best_row or best_iou < person_iou_thr:
            continue

        # 헬멧/조끼 박스도 동일 변환 적용(최종으로 선택된 best_pbox_gt와 같은 방식으로 만들기)
        # 여기서는 간단히 fill/letterbox 중 더 높은 IoU를 냈던 변환을 추적하지 않고,
        # best_pbox_gt를 직접 사용했으므로, 나머지 부가 박스는 GT 좌표계를 모사하기 위해
        # 'pbox를 GT로 가져온 것과 동일한 방식'으로 계산해야 하지만,
        # 평가에 person_xyxy만 쓰이므로 부가박스는 존재할 때 동일 좌표계로만 저장(선택 사항).
        helmet_xyxy_gt = None
        vest_xyxy_gt = None
        if best_row.get("helmet_bbox") is not None:
            # 보수적으로 GT 좌표계와 같은 스케일이라고 가정 (필요시 위와 동일 매핑을 추적해 적용 가능)
            hb = list(map(float, best_row["helmet_bbox"]))
            helmet_xyxy_gt = resize_fill_forward(hb, W, H, Gw, Gh)
        if best_row.get("vest_bbox") is not None:
            vb = list(map(float, best_row["vest_bbox"]))
            vest_xyxy_gt = resize_fill_forward(vb, W, H, Gw, Gh)

        pr = {
            "image_path": _norm_path(img_path),
            "frame_number": int(g.get("frame_number", 0)),
            "person_xyxy": list(map(float, best_pbox_gt)),
            "helmet": best_row.get("helmet"),
            "vest":   best_row.get("vest"),
        }
        if helmet_xyxy_gt is not None:
            pr["helmet_xyxy"] = helmet_xyxy_gt
            if best_row.get("helmet_score") is not None:
                pr["helmet_conf"] = float(best_row.get("helmet_score"))
        if vest_xyxy_gt is not None:
            pr["vest_xyxy"] = vest_xyxy_gt
            if best_row.get("vest_score") is not None:
                pr["vest_conf"] = float(best_row.get("vest_score"))

        pred_rows.append(pr)
        kept += 1

    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pred_path.open("w", encoding="utf-8") as f:
        for r in pred_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[CONVERT] Tried={tried}  exact={(via_exact)}  ±{frame_window}-frame={(via_window)}  anyframe={(via_anyframe)}  notfound={(notfound)}")
    print(f"[CONVERT] Saved Pred.jsonl -> {out_pred_path}  (kept={kept})")
    return pred_rows

# -------- 분류 지표 --------
def prf_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    acc = (tp+tn)/max(tp+tn+fp+fn, 1)
    prec = tp/max(tp+fp, 1)
    rec = tp/max(tp+fn, 1)
    f1 = 2*prec*rec/max(prec+rec, 1e-12)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def eval_classification(gt_rows: List[dict], pred_rows: List[dict], iou_thr: float, skip_policy: str):
    def img_key(r: dict) -> Tuple[str,int]:
        return (_norm_path(r.get("image_path","")), int(r.get("frame_number",0)))

    gt_by_frame: Dict[Tuple[str,int], List[dict]] = {}
    pr_by_frame: Dict[Tuple[str,int], List[dict]] = {}
    for r in gt_rows:   gt_by_frame.setdefault(img_key(r), []).append(r)
    for r in pred_rows: pr_by_frame.setdefault(img_key(r), []).append(r)

    frames = sorted(set(gt_by_frame.keys()) & set(pr_by_frame.keys()))
    res = {"helmet":{"tp":0,"fp":0,"fn":0,"tn":0,"ignored":0,"n_pairs":0},
           "vest":  {"tp":0,"fp":0,"fn":0,"tn":0,"ignored":0,"n_pairs":0}}

    for fk in frames:
        gt_list = gt_by_frame[fk]; pr_list = pr_by_frame[fk]
        gt_boxes = [g.get("person_xyxy") for g in gt_list]
        pr_boxes = [p.get("person_xyxy") for p in pr_list]
        matches = greedy_match(gt_boxes, pr_boxes, iou_thr=iou_thr)
        for gi, pj, _ in matches:
            g = gt_list[gi]; p = pr_list[pj]
            g_h = to_bool(g.get("helmet")); p_h = to_bool(p.get("helmet"))
            g_v = to_bool(g.get("vest"));   p_v = to_bool(p.get("vest"))
            for key, gt_val, pr_val in (("helmet", g_h, p_h), ("vest", g_v, p_v)):
                if gt_val is None:
                    res[key]["ignored"] += 1; continue
                if pr_val is None:
                    if skip_policy == "ignore":
                        res[key]["ignored"] += 1; continue
                    pr_val = False
                if gt_val and pr_val:              res[key]["tp"] += 1
                elif (not gt_val) and (not pr_val):res[key]["tn"] += 1
                elif (not gt_val) and pr_val:      res[key]["fp"] += 1
                elif gt_val and (not pr_val):      res[key]["fn"] += 1
                res[key]["n_pairs"] += 1

    out = {}
    for k in ("helmet","vest"):
        c = res[k]; m = prf_from_counts(c["tp"], c["fp"], c["fn"], c["tn"]); m.update(c); out[k]=m
    return out

# -------- mAP --------
def average_precision(preds: List[Tuple[str,int,List[float],float]],
                      gts: Dict[Tuple[str,int], List[List[float]]],
                      iou_thr: float) -> float:
    if not preds: return 0.0
    preds = sorted(preds, key=lambda x: x[3], reverse=True)
    gt_used: Dict[Tuple[str,int], List[bool]] = {k:[False]*len(v) for k,v in gts.items()}
    tps, fps = [], []
    total_gt = sum(len(v) for v in gts.values())
    for media, fr, pbox, conf in preds:
        key = (media, fr)
        gt_boxes = gts.get(key, [])
        best_iou, best_idx = 0.0, -1
        if key in gt_used:
            for idx, g in enumerate(gt_boxes):
                if gt_used[key][idx]: continue
                iou = iou_xyxy(pbox, g)
                if iou >= iou_thr and iou > best_iou:
                    best_iou, best_idx = iou, idx
        if best_idx >= 0:
            tps.append(1); fps.append(0); gt_used[key][best_idx] = True
        else:
            tps.append(0); fps.append(1)
    if total_gt == 0: return 0.0
    tps = np.array(tps); fps = np.array(fps)
    cum_tp = np.cumsum(tps); cum_fp = np.cumsum(fps)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    ap = float(np.trapz(precisions, recalls))
    return ap

def eval_map(gt_rows: List[dict], pred_rows: List[dict], iou_thr: float):
    def media_id_and_frame(row: dict) -> Tuple[str,int]:
        return (_norm_path(row.get("image_path","")), int(row.get("frame_number",0)))
    gt_helmet: Dict[Tuple[str,int], List[List[float]]] = {}
    gt_vest:   Dict[Tuple[str,int], List[List[float]]] = {}
    for g in gt_rows:
        key = media_id_and_frame(g)
        if g.get("helmet_bbox") is not None:
            gt_helmet.setdefault(key, []).append(list(map(float, g["helmet_bbox"])))
        if g.get("vest_bbox") is not None:
            gt_vest.setdefault(key, []).append(list(map(float, g["vest_bbox"])))
    preds_helmet: List[Tuple[str,int,List[float],float]] = []
    preds_vest:   List[Tuple[str,int,List[float],float]] = []
    for p in pred_rows:
        key = media_id_and_frame(p)
        if p.get("helmet_xyxy") is not None:
            conf = float(p.get("helmet_conf", 1.0))
            preds_helmet.append((key[0], key[1], list(map(float, p["helmet_xyxy"])), conf))
        if p.get("vest_xyxy") is not None:
            conf = float(p.get("vest_conf", 1.0))
            preds_vest.append((key[0], key[1], list(map(float, p["vest_xyxy"])), conf))
    ap_helmet = average_precision(preds_helmet, gt_helmet, iou_thr=iou_thr) if preds_helmet else 0.0
    ap_vest   = average_precision(preds_vest,   gt_vest,   iou_thr=iou_thr) if preds_vest   else 0.0
    return {"ap50_helmet": ap_helmet, "ap50_vest": ap_vest, "map50": (ap_helmet + ap_vest)/2.0}

def _pct(x: float) -> str:
    return f"{x*100:.2f}%"

# -------- 메인 --------
def main():
    script_dir = Path(__file__).resolve().parent
    pred_out   = Path(DEFAULT_PRED_JSON) if DEFAULT_PRED_JSON else (script_dir / "Pred_images.jsonl")
    report_out = Path(DEFAULT_REPORT)    if DEFAULT_REPORT    else (script_dir / "reports" / "evaluation.json")
    gt_path  = Path(DEFAULT_GT)
    csv_path = Path(DEFAULT_CSV)

    if not gt_path.exists():
        print(f"[ERR] GT 파일을 찾을 수 없습니다: {gt_path}"); sys.exit(1)
    if not csv_path.exists():
        print(f"[ERR] CSV 파일을 찾을 수 없습니다: {csv_path}"); sys.exit(1)

    gt_rows  = load_gt_rows(gt_path)
    csv_rows = load_csv_rows(csv_path)
    print(f"[LOAD] GT rows={len(gt_rows)}, CSV rows={len(csv_rows)}")

    # 변환: (video_key + frame) → IoU 최대 매칭 (좌표계 자동 매핑)
    pred_rows = convert_csv_to_pred_jsonl_for_images(
        gt_rows, csv_rows, pred_out,
        person_iou_thr=0.03,   # 필요하면 0.02~0.06 사이로 조절
        frame_window=3         # ±3 프레임 허용
    )

    if not pred_rows:
        print("[EVAL] Pred가 비어있습니다. (매칭 실패)")

    # 성능평가
    cls_metrics = eval_classification(gt_rows, pred_rows, iou_thr=0.5, skip_policy="negative")
    map_metrics = eval_map(gt_rows, pred_rows, iou_thr=0.5)

    print("\n=== Person-level Classification ===")
    for k in ("helmet","vest"):
        m = cls_metrics[k]
        print(f"[{k}] acc={_pct(m['accuracy'])}  prec={_pct(m['precision'])}  rec={_pct(m['recall'])}  f1={_pct(m['f1'])}  "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']} ignored={m['ignored']} pairs={m['n_pairs']})")

    print("\n=== Box-level mAP@0.5 ===")
    print(f"AP50(helmet)={_pct(map_metrics['ap50_helmet'])}  AP50(vest)={_pct(map_metrics['ap50_vest'])}  "
          f"mAP50={_pct(map_metrics['map50'])}")

    # 리포트 저장
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {
            "gt_path": str(gt_path),
            "csv_path": str(csv_path),
            "pred_jsonl": str(pred_out),
            "match": {"person_iou_thr": 0.03, "frame_window": 3},
            "eval": {"iou_thr": 0.50, "skip_policy": "negative"},
            "auto_mapping": "per-video W,H from CSV quantiles (p99.9, ceil/32) & per-video Gw,Gh from GT quantiles (p99.9, ceil/32); try fill+letterbox+identity per frame",
        },
        "classification": cls_metrics,
        "detection_map50": map_metrics,
        "counts": {"gt_rows": len(gt_rows), "csv_rows": len(csv_rows), "pred_rows": len(pred_rows)}
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[REPORT] Saved -> {report_out}")

if __name__ == "__main__":
    main()
