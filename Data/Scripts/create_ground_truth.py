# make_gt_simple.py
# 사용법: python make_gt_simple.py
import re
import pandas as pd

INPUT  = "/Users/chonakyung/Library/CloudStorage/GoogleDrive-whskrud1007@gmail.com/내 드라이브/학부 연구생_2025/Safety--People-Detector/Data/output/logs/tracks.csv"
OUTPUT = "/Users/chonakyung/Library/CloudStorage/GoogleDrive-whskrud1007@gmail.com/내 드라이브/학부 연구생_2025/Safety--People-Detector/Data/output/logs/tracks_GT.csv"

# ── 규칙: 아래에만 추가/수정 ──────────────────────────────────────────────
CONDITIONS = [
    {"type": "contains", "pattern": "human-accident_bump_rgb_0001_cctv1", "helmet": True, "vest": False},
    {"type": "contains", "pattern": "human-accident_hit_rgb_0662_cctv2", "helmet": True, "vest": True},
    {"type": "contains", "pattern": "human-accident_jam_rgb_0644_cctv3", "helmet": True, "vest": False},
    {"type": "contains", "pattern": "human-accident_jam_rgb_1205_cctv2", "helmet": False, "vest": False},
    {"type": "contains", "pattern": "intrusion_attempt-to-set-fire_rgb_0715_cctv3_1", "helmet": False, "vest": False},
    {"type": "contains", "pattern": "intrusion_attempt-to-set-fire_rgb_0994_cctv1", "helmet": False, "vest": False},
    {"type": "contains", "pattern": "intrusion_attempt-to-set-fire_rgb_1520_cctv1_1", "helmet": False, "vest": False},
    {"type": "contains", "pattern": "intrusion_climb-over-fence_rgb_1231_cctv3_1", "helmet": False, "vest": False},
    {"type": "contains", "pattern": "intrusion_damage-to-facilities_rgb_0627_cctv2_1", "helmet": False, "vest": False},
    {"type": "contains", "pattern": "intrusion_damage-to-facilities_rgb_1241_cctv3_1", "helmet": False, "vest": False},
    {"type": "contains", "pattern": "intrusion_normal_rgb_0477_cctv1_2", "helmet": True, "vest": True},
    {"type": "contains", "pattern": "intrusion_normal_rgb_0511_cctv1", "helmet": True, "vest": True},
    {"type": "contains", "pattern": "intrusion_normal_rgb_0556_cctv4", "helmet": True, "vest": False},
    {"type": "contains", "pattern": "intrusion_normal_rgb_0938_cctv2_2", "helmet": True, "vest": True},
    {"type": "contains", "pattern": "intrusion_theft_rgb_0549_cctv2", "helmet": False, "vest": False},
]
# ──────────────────────────────────────────────────────────────────────

# (출력 포맷 옵션) CSV에 boolean을 문자열("True"/"False"/"None")로 쓸지
CAST_BOOL_TO_STR = True

def norm_bool(x):
    if isinstance(x, bool): return x
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip().lower()
    if s in {"true","t","1","yes","y"}:  return True
    if s in {"false","f","0","no","n"}:  return False
    if s in {"none","nan",""}:           return None
    try:
        return bool(int(float(s)))
    except:
        return None

def mask_by_rule(series: pd.Series, rule: dict) -> pd.Series:
    """대소문자 무시 매칭. NaN 안전."""
    s = series.fillna("").astype(str).str.lower()
    typ = rule.get("type", "equals")
    pat = str(rule.get("pattern", "")).lower()

    if typ == "equals":
        return s == pat
    if typ == "contains":
        return s.str.contains(re.escape(pat), regex=True, na=False)
    if typ == "startswith":
        return s.str.startswith(pat, na=False)
    if typ == "endswith":
        return s.str.endswith(pat, na=False)
    if typ == "regex":
        return series.fillna("").astype(str).str.contains(pat, regex=True, na=False, case=False)
    raise ValueError(f"Unknown rule type: {typ}")

# === 실행 ===
df = pd.read_csv(INPUT)

# video key (video_name 우선)
if "video_name" in df.columns:
    KEY = "video_name"
elif "video_url" in df.columns:
    KEY = "video_url"
else:
    raise ValueError("CSV에 'video_name' 또는 'video_url' 컬럼이 있어야 합니다.")

# 대상 컬럼 준비 (없으면 생성 - 기본 None)
if "helmet" not in df.columns: df["helmet"] = None
if "vest"   not in df.columns: df["vest"]   = None

# [NEW] part/bodypart 컬럼 자동 탐지 (없으면 생성)
PART_COL = None
for cand in ["part", "bodypart", "Bodypart", "BodyPart"]:
    if cand in df.columns:
        PART_COL = cand
        break
if PART_COL is None:
    PART_COL = "part"
    df[PART_COL] = None

# 현재 값 정규화(문자 'None' → None, 'true/false' → bool)
df["helmet"] = df["helmet"].map(norm_bool)
df["vest"]   = df["vest"].map(norm_bool)

# 1) 규칙 적용: 매칭된 행은 현재 값이 None이든 아니든 **무조건** 규칙값으로 덮어쓰기
applied_any = pd.Series(False, index=df.index)
for i, rule in enumerate(CONDITIONS, start=1):
    m = mask_by_rule(df[KEY], rule)
    if "helmet" in rule:
        df.loc[m, "helmet"] = bool(rule["helmet"])
    if "vest" in rule:
        df.loc[m, "vest"] = bool(rule["vest"])
    applied_any |= m
    print(f"[RULE {i}] matched rows: {int(m.sum())}")

# 2) bodypart 통일: 지정된 값들은 모두 'full_body'로
def _norm_text(s: str) -> str:
    return re.sub(r'[\s\-_]+', '', s.strip().lower())

TO_FULL = {
    "upperbodywithface", "upperbody", "lowerbody", "face", "partial"
}
mask_part = df[PART_COL].notna() & df[PART_COL].astype(str).str.len().gt(0)
df.loc[mask_part & df[PART_COL].astype(str).map(_norm_text).isin(TO_FULL), PART_COL] = "full_body"

# 3) (선택) 출력 포맷 통일
if CAST_BOOL_TO_STR:
    def as_str(v):
        if v is True:  return "True"
        if v is False: return "False"
        return "None"
    df["helmet"] = df["helmet"].map(as_str)
    df["vest"]   = df["vest"].map(as_str)

# 저장
df.to_csv(OUTPUT, index=False)
print(f"Done. Wrote -> {OUTPUT} ({len(df)} rows). "
      f"Matched(any rule): {int(applied_any.sum())}, Unmatched: {len(df) - int(applied_any.sum())}, "
      f"part-column: {PART_COL}")
