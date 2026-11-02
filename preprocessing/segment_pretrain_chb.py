"""Tạo dữ liệu pretrain pre/inter từ các bệnh nhân ngoài TEST_START.

Đọc EDF gốc có trong summary, dùng rules.py và cùng bộ lọc với train/test v4.
Không chia ngẫu nhiên các cửa sổ; có thể chia validation theo bệnh nhân sau.
"""

import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

import rules
from get_summary_chb import getSummary
from split_chb_manual import DATA_ROOT, TEST_START
from segment_test_chb import WINDOW_SECONDS, export_edf
from segment_train_chb import LABELS, save_labeled

OUTPUT_ROOT = Path(__file__).resolve().parent / "segment_pretrain"
DRY_RUN = False  # True: chỉ kiểm tra/đếm; False: xuất NPZ.

# PATIENTS = None: tự chọn tất cả bệnh nhân ngoài danh sách đích.
PATIENTS = (
    "chb02", "chb03", "chb10", "chb11", "chb12",
    "chb13", "chb16", "chb17", "chb18", "chb19",
    "chb20", "chb23", "chb24",  # Loại CHB21 vì cùng người với CHB01.
)

# CHB01 và CHB21 là cùng người: không dùng CHB21 để pretrain cho CHB01.
SUBJECT_GROUPS = ({"chb01", "chb21"},)


def select_patients():
    excluded = set(TEST_START)
    for group in SUBJECT_GROUPS:
        if group & excluded:
            excluded.update(group)
    available = {p.name for p in DATA_ROOT.glob("chb*") if p.is_dir()}
    patients = sorted(available - excluded) if PATIENTS is None else list(PATIENTS)
    if not patients or len(set(patients)) != len(patients):
        raise ValueError("No patients selected or duplicate patient IDs")
    overlap = sorted(set(patients) & excluded)
    missing = sorted(set(patients) - available)
    if overlap or missing:
        raise ValueError(f"Invalid PATIENTS: excluded target/related cases={overlap}; "
                         f"missing directories={missing}; DATA_ROOT={DATA_ROOT}")
    return patients, excluded


def plan_patient(patient):
    folder = DATA_ROOT / patient
    records, _ = rules.build_timeline(getSummary(folder / f"{patient}-summary-dated.txt"))
    names = {r["file_name"] for r in records}
    actual = {p.name for p in folder.glob("*.edf")}
    if names - actual:
        raise ValueError(f"{patient}: missing EDFs: {sorted(names - actual)}")
    # Không gán nhãn cho EDF thiếu annotation trong summary (đặc biệt CHB24).
    # Các khoảng thiếu này vẫn được xét là gap khi kiểm tra preictal/interictal.
    unlisted = sorted(actual - names)
    # Không áp dụng điều kiện >=3 cụm cơn để loại bệnh nhân khỏi dữ liệu pretrain.
    # Vẫn giữ nguyên điều kiện hợp lệ cho từng đoạn preictal/interictal.
    _, audits, segments, _ = rules.analyze_patient(records, patient)
    spans, counts = defaultdict(list), Counter()
    for segment in segments:
        if segment["label"] not in LABELS:
            continue
        spans[segment["source_edf"]].append(segment)
        first = segment["start_seconds_in_file"]
        last = segment["end_seconds_in_file"]
        count = max(0, math.floor(last / WINDOW_SECONDS) - math.ceil(first / WINDOW_SECONDS))
        kind = "pre" if LABELS[segment["label"]] == 1 else "inter"
        counts[kind] += count
    return records, spans, {"pre": counts["pre"], "inter": counts["inter"]}, audits, unlisted


def export_patient(patient, records, spans, expected):
    destination = OUTPUT_ROOT / patient
    if destination.exists():
        raise FileExistsError(destination)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Chỉ đưa kết quả ra thư mục chính khi bệnh nhân đã xuất hoàn chỉnh.
    with tempfile.TemporaryDirectory(prefix=f".{patient}_", dir=OUTPUT_ROOT) as tmp:
        stage = Path(tmp) / patient
        for kind in ("pre", "inter"):
            (stage / kind).mkdir(parents=True)
        state, previous_end, counts = None, None, Counter()
        for record in tqdm(records, desc=f"Pretrain {patient}"):
            gap = None if previous_end is None else (record["file_start_time"] - previous_end).total_seconds()
            archive = Path(tmp) / "continuous.npz"
            # Lọc cả EDF không có nhãn để giữ đúng trạng thái giữa các EDF liền nhau.
            state, previous_end, _ = export_edf(
                DATA_ROOT / patient / record["file_name"], archive, record,
                state, gap, patient_id=patient,
            )
            counts.update(save_labeled(archive, record, spans[record["file_name"]], stage, "pretrain"))
            archive.unlink()
        totals = {kind: counts[kind] for kind in ("pre", "inter")}
        if totals != expected:
            raise ValueError(f"{patient}: planned {expected}, exported {totals}")
        (stage / "counts.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
        stage.rename(destination)
    return totals


def main():
    patients, excluded = select_patients()
    print(f"Pretrain patients: {', '.join(patients)}", flush=True)
    plans, report = {}, {}
    for patient in patients:
        # Dừng khi dữ liệu sai/thiếu thay vì âm thầm bỏ bệnh nhân.
        records, spans, counts, audits, unlisted = plan_patient(patient)
        plans[patient] = records, spans, counts
        report[patient] = {"windows": counts, "edf_count": len(records),
                           "eligible_clusters": sum(a["eligible_preictal"] for a in audits),
                           "unlisted_edfs_excluded": unlisted}
        print(f"{patient}: {counts}", flush=True)
        if not counts["pre"] or not counts["inter"]:
            print(f"[NOTE] {patient}: only one class or no eligible windows", flush=True)
        if unlisted:
            print(f"[NOTE] {patient}: excluded EDFs absent from summary: {unlisted}", flush=True)
        if not DRY_RUN and (OUTPUT_ROOT / patient).exists():
            raise FileExistsError(OUTPUT_ROOT / patient)
    if DRY_RUN:
        print("DRY RUN: no EEG exported")
        return
    for patient, (records, spans, counts) in plans.items():
        if not sum(counts.values()):
            print(f"[SKIP] {patient}: no eligible windows", flush=True)
            continue
        export_patient(patient, records, spans, counts)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "patients.json").write_text(json.dumps(
        {"excluded": sorted(excluded), "patients": report,
         "source_root": str(DATA_ROOT), "window_seconds": WINDOW_SECONDS}, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
