# tools/eval/evaluate_ppe.py
# -*- coding: utf-8 -*-
"""
두 개의 분리된 파일(GT.json/jsonl, Pred.json/jsonl/csv)을 비교하여
- 사람 단위 분류 지표: Accuracy / Precision / Recall / F1 (helmet, vest 각각)
- 박스 단위 검출 지표: mAP@0.5 (helmet, vest 각각)
를 계산합니다.

➡️ 인자 없이 실행하려면 파일 상단 DEFAULT_* 경로만 바꾸세요.
CLI 인자도 그대로 지원합니다.
예)
python tools/eval/evaluate_ppe.py
python tools/eval/evaluate_ppe.py --gt ..\..\Data\output\quick_test_all.jsonl --pred ..\..\Data\output\logs\tracks.csv
"""

import argparse, json, os, sys, csv, ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# ======== 여기를 네 환경에 맞게 바꾸면 인자 없이 실행 가능 ========
DEFAULT_GT          = r"..\..\Data\output\quick_test_all.jsonl"   # .json/.jsonl
DEFAULT_PRED        = r"..\..\Data\output\logs\tracks.csv"        # .csv/.json/.jsonl 모두 지원
DEFAULT_OUT         = str((Path(__file__).resolve().parent / "reports" / "evaluation.json"))
DEFAULT_IOU         = 0.5
DEFAULT_SKIP_POLICY = "negative"  # "negative" | "ignore"
# ================================================================

def _norm_path(p: str) -> str:
    if p is None:
        return ""
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(p)))
    except Exception:
        return str(p)

# ---------- 로더들 ----------
def load_json_or_jsonl(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("JSON array expected.")
    else:
        rows = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
        return rows

def _maybe_list4(x):
    """문자열 "[x1,x2,x3,x4]" 혹은 튜플/리스트를 4원소 float 리스트로 변환."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    try:
        v = json.loads(s)
    except Exception:
        try:
            v = ast.literal_eval(s)
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
        except Exception:
            return None
    return None

def load_csv_normalized(path: Path) -> List[dict]:
    """
    tracker 로그 CSV를 Pred 표준 레코드로 정규화.
    지원 헤더(네 코드 기준):
      video_name, frame, id, x1, y1, x2, y2, score, helmet, vest, part,
      helmet_bbox, helmet_score, vest_bbox, vest_score
    """
    out: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                # 필수값
                x1 = float(r.get("x1", 0)); y1 = float(r.get("y1", 0))
                x2 = float(r.get("x2", 0)); y2 = float(r.get("y2", 0))
                frame = int(float(r.get("frame", r.get("frame_number", 0))))
                video_name = r.get("video_name") or r.get("video_path") or ""

                row = {
                    "video_path":   video_name,                # 비디오 기반 키
                    "frame_number": frame,
                    "person_xyxy":  [x1, y1, x2, y2],
                    "helmet":       r.get("helmet"),
                    "vest":         r.get("vest"),
                }

                # mAP용 박스/스코어를 Pred 표준 키로 매핑
                hb = _maybe_list4(r.get("helmet_bbox"))
                if hb is not None:
                    row["helmet_xyxy"] = hb
                hs = r.get("helmet_score")
                if hs not in (None, "", "None"):
                    try: row["helmet_conf"] = float(hs)
                    except: pass

                vb = _maybe_list4(r.get("vest_bbox"))
                if vb is not None:
                    row["vest_xyxy"] = vb
                vs = r.get("vest_score")
                if vs not in (None, "", "None"):
                    try: row["vest_conf"] = float(vs)
                    except: pass

                out.append(row)
            except Exception:
                # 문제 행은 스킵
                continue
    return out

def load_rows(path: Path) -> List[dict]:
    suf = path.suffix.lower()
    if suf in (".json", ".jsonl"):
        return load_json_or_jsonl(path)
    if suf == ".csv":
        return load_csv_normalized(path)
    raise ValueError(f"지원하지 않는 포맷: {path}")

# ---------- 공용 유틸 ----------
def frame_key(row: dict) -> Tuple[str, int]:
    """
    같은 프레임을 식별하기 위한 키.
    - 이미지 기반: (image_path, frame_number)
    - 비디오 기반: (video_path|video_name, frame_number|frame)
    """
    if row.get("image_path"):
        return (_norm_path(row["image_path"]), int(row.get("frame_number", row.get("frame", 0))))
    vp = row.get("video_path") or row.get("video_name") or ""
    fn = int(row.get("frame_number", row.get("frame", 0)))
    return (_norm_path(vp), fn)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)

def greedy_match(gt_boxes: List[List[float]], pred_boxes: List[List[float]], iou_thr: float) -> List[Tuple[int,int,float]]:
    matches: List[Tuple[int,int,float]] = []
    if not gt_boxes or not pred_boxes:
        return matches
    used_pred = set()
    cands = []
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou = iou_xyxy(g, p)
            if iou >= iou_thr:
                cands.append((iou, i, j))
    for iou, gi, pj in sorted(cands, key=lambda x: x[0], reverse=True):
        if pj in used_pred:
            continue
        if any(m[0] == gi for m in matches):
            continue
        used_pred.add(pj)
        matches.append((gi, pj, iou))
    return matches

def to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("true","1","yes","y"):   return True
    if s in ("false","0","no","n"):   return False
    if s in ("skip","none","null",""):return None
    return None

def prf_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    acc = (tp+tn)/max(tp+tn+fp+fn, 1)
    prec = tp/max(tp+fp, 1)
    rec = tp/max(tp+fn, 1)
    f1 = 2*prec*rec/max(prec+rec, 1e-12)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------- 분류 지표 ----------
def eval_classification(gt_rows: List[dict], pred_rows: List[dict], iou_thr: float, skip_policy: str, debug: bool=False):
    gt_by_frame: Dict[Tuple[str,int], List[dict]] = {}
    pr_by_frame: Dict[Tuple[str,int], List[dict]] = {}
    for r in gt_rows:   gt_by_frame.setdefault(frame_key(r), []).append(r)
    for r in pred_rows: pr_by_frame.setdefault(frame_key(r), []).append(r)

    gt_keys = set(gt_by_frame.keys())
    pr_keys = set(pr_by_frame.keys())
    frames = sorted(gt_keys & pr_keys)

    if debug:
        print(f"[DEBUG] GT frames={len(gt_keys)}  Pred frames={len(pr_keys)}  Intersect={len(frames)}")
        if len(frames) == 0:
            print("  - 샘플 GT key 5개:", list(gt_keys)[:5])
            print("  - 샘플 Pred key 5개:", list(pr_keys)[:5])
            print("  ※ GT가 image_path 기반, Pred가 video_path 기반이면 교집합이 0이 됩니다.")

    res = {"helmet":{"tp":0,"fp":0,"fn":0,"tn":0,"ignored":0,"n_pairs":0},
           "vest":  {"tp":0,"fp":0,"fn":0,"tn":0,"ignored":0,"n_pairs":0}}
    pair_cnt = 0

    for fk in frames:
        gt_list = gt_by_frame[fk]; pr_list = pr_by_frame[fk]
        gt_boxes = [g.get("person_xyxy") for g in gt_list]
        pr_boxes = [p.get("person_xyxy") for p in pr_list]
        matches = greedy_match(gt_boxes, pr_boxes, iou_thr=iou_thr)
        pair_cnt += len(matches)
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

    if debug:
        print(f"[DEBUG] matched pairs={pair_cnt}")

    out = {}
    for k in ("helmet","vest"):
        c = res[k]; m = prf_from_counts(c["tp"], c["fp"], c["fn"], c["tn"]); m.update(c); out[k]=m
    return out

# ---------- mAP ----------
def average_precision(preds: List[Tuple[str,int,List[float],float]],
                      gts: Dict[Tuple[str,int], List[List[float]]],
                      iou_thr: float) -> float:
    if not preds:
        return 0.0
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
    if total_gt == 0:
        return 0.0
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
        if "image_path" in row and row["image_path"]:
            return (_norm_path(row["image_path"]), int(row.get("frame_number", row.get("frame", 0))))
        return (_norm_path(row.get("video_path") or row.get("video_name") or ""), int(row.get("frame_number", row.get("frame", 0))))
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   type=Path, help="GT json/jsonl 경로")
    parser.add_argument("--pred", type=Path, help="Pred json/jsonl/csv 경로")
    parser.add_argument("--out",  type=Path, help="리포트 저장 경로(json)")
    parser.add_argument("--iou",  type=float, default=DEFAULT_IOU)
    parser.add_argument("--skip_policy", choices=["negative","ignore"], default=DEFAULT_SKIP_POLICY)
    parser.add_argument("--debug", action="store_true", help="매칭 디버그 출력")
    args = parser.parse_args()

    # 인자 없으면 DEFAULT_* 사용
    gt_path   = args.gt   or (Path(DEFAULT_GT)   if DEFAULT_GT   else None)
    pred_path = args.pred or (Path(DEFAULT_PRED) if DEFAULT_PRED else None)
    out_path  = args.out  or (Path(DEFAULT_OUT)  if DEFAULT_OUT  else None)

    if not gt_path or not pred_path:
        print("[ERR] GT/Pred 경로가 설정되지 않았습니다. --gt/--pred 인자를 주거나, 파일 상단 DEFAULT_* 값을 채워주세요.")
        sys.exit(1)
    if not gt_path.exists():
        print(f"[ERR] GT 파일을 찾을 수 없습니다: {gt_path}"); sys.exit(1)
    if not pred_path.exists():
        print(f"[ERR] Pred 파일을 찾을 수 없습니다: {pred_path}"); sys.exit(1)

    gt_rows   = load_rows(gt_path)
    pred_rows = load_rows(pred_path)

    print(f"\nLoaded: GT={len(gt_rows)} rows, Pred={len(pred_rows)} rows")

    cls_metrics = eval_classification(gt_rows, pred_rows, iou_thr=args.iou, skip_policy=args.skip_policy, debug=args.debug)
    map_metrics = eval_map(gt_rows, pred_rows, iou_thr=args.iou)

    report = {
        "settings": {
            "iou_thr": args.iou,
            "skip_policy": args.skip_policy,
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
        },
        "classification": {
            "helmet": cls_metrics["helmet"],
            "vest":   cls_metrics["vest"],
        },
        "detection_map50": map_metrics,
    }

    print("\n=== Person-level Classification ===")
    for k in ("helmet","vest"):
        m = cls_metrics[k]
        print(f"[{k}] acc={_pct(m['accuracy'])}  prec={_pct(m['precision'])}  rec={_pct(m['recall'])}  f1={_pct(m['f1'])}  "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']} ignored={m['ignored']} pairs={m['n_pairs']})")

    print("\n=== Box-level mAP@0.5 ===")
    print(f"AP50(helmet)={_pct(map_metrics['ap50_helmet'])}  AP50(vest)={_pct(map_metrics['ap50_vest'])}  "
          f"mAP50={_pct(map_metrics['map50'])}")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report -> {out_path}")

if __name__ == "__main__":
    main()
