"""Chia EDF thành train/test theo mốc cố định; giữ nguyên dữ liệu gốc."""

import csv
import shutil
import sys
from pathlib import Path

from get_summary_chb import getSummary

BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE.parents[3] / "chb-mit-scalp-eeg-database-1.0.0"
OUTPUT_ROOT = BASE / "edf_split_manual"
DRY_RUN = False  # True: chỉ xem số lượng; False: sao chép dữ liệu.

# File tại mốc này thuộc test, các file trước đó thuộc train (không chia val).
# TEST_START = {
#     "chb01": "25", "chb04": "34",
#     "chb05": "31", "chb06": "18", 
#     "chb07": "17", "chb08": "29", 
#     "chb09": "15", "chb14": "37", 
#     "chb15": "61", "chb22": "51",
# }

"""
chb01: train=24, test=18; test bắt đầu từ chb01_25.edf
chb05: train=20, test=19; test bắt đầu từ chb05_21.edf
chb06: train=12, test=6; test bắt đầu từ chb06_14.edf
chb07: train=14, test=5; test bắt đầu từ chb07_15.edf
chb09: train=8, test=11; test bắt đầu từ chb09_09.edf
"""

TEST_START = {
    "chb01": "25", 
    "chb05": "21",
    "chb06": "14", 
    "chb07": "15", 
    "chb09": "09", 
}


def plan_patient(data_root, patient, marker):
    folder = data_root / patient
    summary = folder / f"{patient}-summary-dated.txt"
    records = sorted(getSummary(summary), key=lambda r: r["file_start_time"])
    # CHB17 dùng tên chb17c_04.edf; các bệnh nhân khác dùng chbXX_NN.edf.
    boundary = f"{patient}{marker}.edf" if marker == "c_04" else f"{patient}_{marker}.edf"
    names = [r["file_name"] for r in records]
    if len(set(names)) != len(names) or boundary not in names:
        raise ValueError(f"{patient}: summary trùng tên hoặc thiếu mốc {boundary}")
    for name in names:
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    # Không âm thầm bỏ qua EDF chưa được khai báo trong summary.
    unlisted = {p.name for p in folder.glob("*.edf")} - set(names)
    if unlisted:
        raise ValueError(f"{patient}: EDF thiếu trong summary: {sorted(unlisted)}")
    index = names.index(boundary)
    if index == 0:
        raise ValueError(f"{patient}: tập train rỗng")
    return summary, {"train": records[:index], "test": records[index:]}


def main():
    # Hiển thị tiếng Việt trên terminal Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    source, output = DATA_ROOT.resolve(), OUTPUT_ROOT.resolve()
    if output == source or source in output.parents:
        raise ValueError("Thư mục đầu ra phải nằm ngoài thư mục dữ liệu gốc")
    if output.exists() and not DRY_RUN:
        raise FileExistsError(f"Đầu ra đã tồn tại; chọn OUTPUT_ROOT khác: {output}")

    # Kiểm tra đủ 18 bệnh nhân trước khi bắt đầu sao chép.
    plans = {p: plan_patient(source, p, m) for p, m in TEST_START.items()}
    for patient, (summary, splits) in plans.items():
        print(f"{patient}: train={len(splits['train'])}, test={len(splits['test'])}; "
              f"test bắt đầu từ {splits['test'][0]['file_name']}")
        if DRY_RUN:
            continue
        rows = []
        for split, records in splits.items():
            destination = output / patient / split
            destination.mkdir(parents=True, exist_ok=False)
            for record in records:
                shutil.copy2(source / patient / record["file_name"], destination)
                rows.append({"split": split, **record})
        # Giữ summary và manifest để các bước xử lý sau đọc lại cách chia.
        shutil.copy2(summary, output / patient)
        with (output / patient / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Chạy thử: chưa sao chép dữ liệu." if DRY_RUN else f"Đã lưu: {output}")


if __name__ == "__main__":
    main()
