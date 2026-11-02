"""Lọc bệnh nhân, không yêu cầu 24 giờ; CSV sắp theo mã bệnh nhân tăng dần."""

import csv
import math
from collections import Counter
from pathlib import Path

import rules
from get_summary_chb import getSummary

BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE.parents[3] / "chb-mit-scalp-eeg-database-1.0.0"
OUTPUT_DIR = BASE / "patient_selection"
MIN_TEST_HOURS = 3
MAX_GAP_SECONDS = 300
MIN_TEST_COVERAGE = 0.99
# Cùng lưới cửa sổ không chồng lấn, căn theo đầu EDF như segment_test_chb.py.
WINDOW_SECONDS = 4
FIELDS = ["patient_id", "recorded_hours", "recorded_seizures",
          "eligible_clusters", "eligible_seizure_files", "balanced_clusters",
          "balanced_seizure_files",
          "interictal_minutes", "preictal_minutes", "selected"]


def hours(records):
    return sum((r["file_end_time"] - r["file_start_time"]).total_seconds()
               for r in records) / 3600


def format_cluster_files(clusters):
    """Mỗi cặp ngoặc là một cụm, các cụm xếp theo thời điểm onset."""
    return " | ".join(
        "(" + ";".join(sorted(set(a["seizure_files"]))) + ")"
        for a in sorted(clusters, key=lambda a: a["onset"])
    )


def get_balanced_clusters(audits, segments):
    """Lấy các cụm có preictal hợp lệ và đủ cửa sổ interictal riêng để cân bằng.

    Gán mỗi đoạn interictal cho cụm kế tiếp theo onset, giống kfold_by_seizure.
    Xét cả cụm bị rules loại khi gán; không chuyển phần của cụm đó sang cụm khác.
    Interictal sau cụm cuối không thuộc cụm nào. Thống kê trên toàn bộ bệnh nhân,
    trước khi chia train/test, dựa vào thời gian summary, chưa đọc tín hiệu EDF.
    """
    ordered = sorted(audits, key=lambda a: a["onset"])
    pre_counts, inter_counts = Counter(), Counter()
    for segment in segments:
        label = segment["label"]
        if label not in {"preictal", "interictal"}:
            continue
        # Chỉ lấy cửa sổ trọn vẹn trong đoạn, không nối qua EDF hoặc gap.
        count = max(0, math.floor(segment["end_seconds_in_file"] / WINDOW_SECONDS)
                    - math.ceil(segment["start_seconds_in_file"] / WINDOW_SECONDS))
        if label == "preictal":
            pre_counts[segment["cluster_id"]] += count
        else:
            following = next((a for a in ordered
                              if segment["end"] <= a["onset"]), None)
            if following is not None:
                inter_counts[following["cluster_id"]] += count
    return [a for a in ordered if a["eligible_preictal"]
               and pre_counts[a["cluster_id"]] > 0
               and inter_counts[a["cluster_id"]] >= pre_counts[a["cluster_id"]]]


def count_balanced_clusters(audits, segments):
    """Đếm các cụm đạt điều kiện cân bằng."""
    return len(get_balanced_clusters(audits, segments))


def valid_part(records, patient, full_audits, full_segments, minimum_clusters):
    """Kiểm tra số cụm hợp lệ và có dữ liệu interictal trong một phần."""
    _, audits, _, _ = rules.analyze_patient(records, patient)
    local_valid = {a["onset"] for a in audits if a["eligible_preictal"]}
    names = {r["file_name"] for r in records}
    # Giữ ngữ cảnh toàn bộ dữ liệu, không đếm cụm bị cắt ở biên chia.
    count = sum(a["eligible_preictal"] and a["onset"] in local_valid
                and set(a["seizure_files"]).issubset(names) for a in full_audits)
    has_interictal = any(s["label"] == "interictal" and s["source_edf"] in names
                        for s in full_segments)
    return count >= minimum_clusters and has_interictal


def is_selected(records, folder, summary, audits, segments):
    listed = {r["file_name"] for r in records}
    actual = {p.name for p in folder.glob("*.edf")}
    if listed != actual or summary["eligible_clusters"] < 3:
        return False

    def cuts_cluster(boundary):
        return any(a["onset"] < boundary < a["offset"] for a in audits)

    for start in range(1, len(records)):
        boundary = records[start]["file_start_time"]
        if cuts_cluster(boundary):
            continue

        # Test gồm các EDF liên tiếp từ mốc chia đến cuối phiên ghi.
        end = start + 1
        while end < len(records):
            gap = (records[end]["file_start_time"]
                   - records[end - 1]["file_end_time"]).total_seconds()
            if gap > MAX_GAP_SECONDS:
                break
            end += 1

        test = records[start:end]
        if cuts_cluster(test[-1]["file_end_time"]):
            continue
        recorded = hours(test)
        span = (test[-1]["file_end_time"] - boundary).total_seconds() / 3600
        if recorded < MIN_TEST_HOURS or recorded / span < MIN_TEST_COVERAGE:
            continue
        # Test >=1 cụm; phần trước test còn >=2 cụm để chia train/val.
        if (valid_part(test, folder.name, audits, segments, 1)
                and valid_part(records[:start], folder.name, audits, segments, 2)):
            return True
    return False


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    folders = sorted(p for p in DATA_ROOT.glob("chb[0-9][0-9]") if p.is_dir())
    if not folders:
        raise FileNotFoundError(f"Không tìm thấy bệnh nhân trong {DATA_ROOT}")
    rows, selected_patients = [], []
    for folder in folders:
        records = sorted(getSummary(folder / f"{folder.name}-summary-dated.txt"),
                         key=lambda r: r["file_start_time"])
        summary, audits, segments, _ = rules.analyze_patient(records, folder.name)
        balanced = get_balanced_clusters(audits, segments)
        selected = is_selected(records, folder, summary, audits, segments)
        rows.append({"patient_id": folder.name,
                     "recorded_hours": round(hours(records), 6),
                     "recorded_seizures": summary["recorded_seizures"],
                     # Số cụm còn lại có preictal hợp lệ sau khi áp dụng rules.
                     "eligible_clusters": summary["eligible_clusters"],
                     # Nhóm file theo từng cụm; chỉ bỏ tên trùng trong cùng cụm.
                     "eligible_seizure_files": format_cluster_files(
                         [a for a in audits if a["eligible_preictal"]]),
                     "balanced_clusters": len(balanced),
                     "balanced_seizure_files": format_cluster_files(balanced),
                     # Tổng thời lượng có EDF sau rules, trước khi chia train/test.
                     "interictal_minutes": round(summary["interictal_minutes"], 6),
                     "preictal_minutes": round(summary["preictal_minutes"], 6),
                     "selected": selected})
        if selected:
            selected_patients.append(folder.name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "patient_selection.csv", rows)
    write_csv(OUTPUT_DIR / "selected_patients.csv", [r for r in rows if r["selected"]])
    print("Selected:", ", ".join(selected_patients))
    print("CSV:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
