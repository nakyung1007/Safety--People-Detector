# compare_tracks_metrics.py
# 사용: python compare_tracks_metrics.py
# (아래 경로/옵션만 수정)

import pandas as pd
import numpy as np
import re

# ====== 경로 설정 ======
PRED_CSV = "/Users/chonakyung/Library/CloudStorage/GoogleDrive-whskrud1007@gmail.com/내 드라이브/학부 연구생_2025/Safety--People-Detector/Data/output/logs/tracks.csv"
GT_CSV   = "/Users/chonakyung/Library/CloudStorage/GoogleDrive-whskrud1007@gmail.com/내 드라이브/학부 연구생_2025/Safety--People-Detector/Data/output/logs/tracks_GT.csv"
OUT_ROWS = "/Users/chonakyung/Library/CloudStorage/GoogleDrive-whskrud1007@gmail.com/내 드라이브/학부 연구생_2025/Safety--People-Detector/Data/output/logs/compare_rows.csv"
OUT_SUM  = "/Users/chonakyung/Library/CloudStorage/GoogleDrive-whskrud1007@gmail.com/내 드라이브/학부 연구생_2025/Safety--People-Detector/Data/output/logs/metrics_summary.csv"

# ====== 필터/옵션 ======
EXCLUDE_SUBSTRINGS = {"hardhet", "hardhat"}  # video_name에 포함되면 제외 (대소문자 무시)
LIMIT_VIDEO_NAMES = 15  # 앞에서 등장하는 15개 비디오만 사용; 전체 사용하려면 None
DROP_IF_GT_NONE = True  # GT가 None인 라벨은 제거
DROP_IF_PRED_NONE = True  # 예측이 None인 라벨도 제거(원하면 False로)

USECOLS = ["video_name", "frame", "id", "helmet", "vest", "part"]

def norm_bool(x):
    if isinstance(x, bool): return x
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip().lower()
    if s in {"true","t","1","yes","y"}:  return True
    if s in {"false","f","0","no","n"}:  return False
    if s in {"none","nan",""}:           return None
    try:
        return bool(int(float(s)))
    except Exception:
        return None

def norm_part(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip().lower()
    if s in {"", "none", "nan"}: return None
    return s

def filter_video_names(df: pd.DataFrame) -> pd.DataFrame:
    # 제외어 필터
    pat = re.compile("|".join(map(re.escape, EXCLUDE_SUBSTRINGS)), flags=re.IGNORECASE)
    mask_keep = ~df["video_name"].fillna("").str.contains(pat, regex=True)
    df = df[mask_keep].copy()
    # 상위 N개 video_name 제한
    if LIMIT_VIDEO_NAMES is not None:
        order = df["video_name"].dropna().tolist()
        # 최초 등장 순서대로 unique 추출
        seen = []
        for vn in order:
            if vn not in seen:
                seen.append(vn)
            if len(seen) >= LIMIT_VIDEO_NAMES:
                break
        df = df[df["video_name"].isin(seen)].copy()
    return df

def align_by_keys(dfp: pd.DataFrame, dfg: pd.DataFrame) -> pd.DataFrame:
    # 가능한 경우 (video_name, frame, id)로 정합
    keys = [c for c in ["video_name","frame","id"] if c in dfp.columns and c in dfg.columns]
    if set(keys) == {"video_name","frame","id"}:
        merged = dfp.merge(dfg, on=keys, how="inner", suffixes=("_pred","_gt"))
        return merged

    # 그 외: video_name만 공통이면 video_name 그룹 내 순번으로 정렬 매칭
    if "video_name" in dfp.columns and "video_name" in dfg.columns:
        dfp = dfp.sort_values(["video_name"]).copy()
        dfg = dfg.sort_values(["video_name"]).copy()
        dfp["__row_no__"] = dfp.groupby("video_name").cumcount()
        dfg["__row_no__"] = dfg.groupby("video_name").cumcount()
        merged = dfp.merge(dfg, on=["video_name","__row_no__"], how="inner", suffixes=("_pred","_gt"))
        return merged

    raise ValueError("공통 정렬 키가 없습니다. 최소한 'video_name'은 두 CSV 모두에 있어야 합니다.")

def metrics_binary(y_true: pd.Series, y_pred: pd.Series):
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    n = tp + tn + fp + fn
    acc  = (tp + tn) / n if n else np.nan
    rec  = tp / (tp + fn) if (tp + fn) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1   = (2*prec*rec)/(prec+rec) if (np.isfinite(prec) and np.isfinite(rec) and (prec+rec)>0) else np.nan
    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "recall": rec, "f1": f1}

def metrics_multiclass(y_true: pd.Series, y_pred: pd.Series):
    # 정확도
    acc = (y_true == y_pred).mean() if len(y_true) else np.nan
    # 클래스별 지표
    classes = sorted(set(y_true.dropna().unique()))
    recalls = []
    f1s = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        prec= tp / (tp + fp) if (tp + fp) else np.nan
        f1  = (2*prec*rec)/(prec+rec) if (np.isfinite(prec) and np.isfinite(rec) and (prec+rec)>0) else np.nan
        if np.isfinite(rec): recalls.append(rec)
        if np.isfinite(f1):  f1s.append(f1)
    # macro average (지원이 있는 클래스만 평균)
    rec_macro = float(np.mean(recalls)) if recalls else np.nan
    f1_macro  = float(np.mean(f1s)) if f1s else np.nan
    return {"n": int(len(y_true)), "accuracy": float(acc), "recall": rec_macro, "f1": f1_macro}

def pct(x): 
    return f"{x*100:.2f}%" if (x is not None and np.isfinite(x)) else "NaN"

def main():
    # 1) 로드
    dfp = pd.read_csv(PRED_CSV, usecols=[c for c in USECOLS if c is not None and c != ""], dtype=str)
    dfg = pd.read_csv(GT_CSV,   usecols=[c for c in USECOLS if c is not None and c != ""], dtype=str)

    # 2) 정규화
    for col in ["helmet","vest"]:
        dfp[col] = dfp[col].map(norm_bool)
        dfg[col] = dfg[col].map(norm_bool)
    dfp["part"] = dfp["part"].map(norm_part)
    dfg["part"] = dfg["part"].map(norm_part)

    # 숫자형 키 복원
    for col in ["frame","id"]:
        if col in dfp.columns: dfp[col] = pd.to_numeric(dfp[col], errors="coerce").astype("Int64")
        if col in dfg.columns: dfg[col] = pd.to_numeric(dfg[col], errors="coerce").astype("Int64")

    # 3) 필터 (hardhet/hardhat 제외 + 상위 15개 비디오만)
    dfp = filter_video_names(dfp)
    dfg = filter_video_names(dfg)

    # 4) 정렬/매칭
    merged = align_by_keys(dfp, dfg)

    # 5) 라벨 컬럼만 남기고 행별 비교 저장
    keep_cols = ["video_name"]
    for k in ["frame","id"]:
        if f"{k}_pred" in merged.columns and f"{k}_gt" in merged.columns:
            keep_cols += [f"{k}_pred", f"{k}_gt"]
    keep_cols += ["helmet_pred","helmet_gt","vest_pred","vest_gt","part_pred","part_gt"]
    merged_out = merged[keep_cols].copy()

    # 6) 지표 계산

    # Helmet
    hel = merged_out[["helmet_gt","helmet_pred"]].copy()
    if DROP_IF_GT_NONE:
        hel = hel[hel["helmet_gt"].notna()]
    if DROP_IF_PRED_NONE:
        hel = hel[hel["helmet_pred"].notna()]
    m_helmet = metrics_binary(hel["helmet_gt"].astype(bool), hel["helmet_pred"].astype(bool)) if len(hel) else {"n":0,"accuracy":np.nan,"recall":np.nan,"f1":np.nan}

    # Vest
    ves = merged_out[["vest_gt","vest_pred"]].copy()
    if DROP_IF_GT_NONE:
        ves = ves[ves["vest_gt"].notna()]
    if DROP_IF_PRED_NONE:
        ves = ves[ves["vest_pred"].notna()]
    m_vest = metrics_binary(ves["vest_gt"].astype(bool), ves["vest_pred"].astype(bool)) if len(ves) else {"n":0,"accuracy":np.nan,"recall":np.nan,"f1":np.nan}

    # Part (멀티클래스) - GT/PRED가 None인 건 제거
    par = merged_out[["part_gt","part_pred"]].copy()
    par = par[par["part_gt"].notna()]
    if DROP_IF_PRED_NONE:
        par = par[par["part_pred"].notna()]
    m_part = metrics_multiclass(par["part_gt"], par["part_pred"]) if len(par) else {"n":0,"accuracy":np.nan,"recall":np.nan,"f1":np.nan}

    # 7) 출력
    print("=== FILTER ===")
    print(f"- excluded video_name contains: {EXCLUDE_SUBSTRINGS}")
    print(f"- limit video names: {LIMIT_VIDEO_NAMES}")

    print("\n=== HELMET ===")
    print(f"N={m_helmet['n']}, acc={pct(m_helmet['accuracy'])}, recall={pct(m_helmet['recall'])}, f1={pct(m_helmet['f1'])}")

    print("\n=== VEST ===")
    print(f"N={m_vest['n']}, acc={pct(m_vest['accuracy'])}, recall={pct(m_vest['recall'])}, f1={pct(m_vest['f1'])}")

    print("\n=== PART (multiclass, macro) ===")
    print(f"N={m_part['n']}, acc={pct(m_part['accuracy'])}, recall={pct(m_part['recall'])}, f1={pct(m_part['f1'])}")

    # 8) 저장(선택): 행별 비교와 요약 메트릭
    merged_out.to_csv(OUT_ROWS, index=False)
    pd.DataFrame([
        {"target":"helmet", **{k: m_helmet.get(k) for k in ["n","accuracy","recall","f1"]}},
        {"target":"vest",   **{k: m_vest.get(k)   for k in ["n","accuracy","recall","f1"]}},
        {"target":"part",   **{k: m_part.get(k)   for k in ["n","accuracy","recall","f1"]}},
    ]).to_csv(OUT_SUM, index=False)
    print(f"\nSaved per-row comparison -> {OUT_ROWS}")
    print(f"Saved metrics summary    -> {OUT_SUM}")

if __name__ == "__main__":
    main()
