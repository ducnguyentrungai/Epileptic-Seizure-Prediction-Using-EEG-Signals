"""Giữ pre theo cụm cơn, chia inter theo EDF và cân bằng số cửa sổ.

Không gán inter cho cơn tiếp theo. Pre/inter cùng EDF luôn ở cùng subset.
Chỉ xuất kfold.csv tương thích load_kfold_csv.py, không sửa các file NPZ.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rules
from get_summary_chb import getSummary

BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE / "segment_train_4h_30m"
SPLIT_ROOT = BASE / "edf_split_manual"
CSV_PATH = BASE / "segment_train_folds" / "kfold_4h_30m.csv"
PATIENTS = ("chb01", "chb05", "chb06", "chb09",)
K = None  # None: tối đa số fold khả thi; hoặc số nguyên >= 2 làm giới hạn.
MAX_MISSING_PRE_SECONDS = 300  # Cho phép thiếu tối đa 5 phút, vẫn phải qua rules.py.
FIELDS = ["patient", "n_folds", "fold", "subset", "relative_path", "label",
          "windows", "source_edf", "cluster_ids"]


def patient_rows(patient):
    for kind in ("pre", "inter"):
        if not (DATA_ROOT / patient / kind).is_dir():
            raise FileNotFoundError(DATA_ROOT / patient / kind)
    folder = SPLIT_ROOT / patient
    records, _ = rules.build_timeline(getSummary(folder / f"{patient}-summary-dated.txt"))
    with (folder / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    assignments = {r["file_name"]: r["split"] for r in manifest}
    if (len(assignments) != len(manifest)
            or set(assignments) != {r["file_name"] for r in records}
            or set(assignments.values()) - {"train", "test"}):
        raise ValueError(f"{patient}: manifest không khớp summary")
    _, audits, segments, _ = rules.analyze_patient(records, patient)
    train_clusters = {c["cluster_id"] for c in rules.build_clusters(records)
                      if all(assignments[s["source_edf"]] == "train" for s in c["seizures"])}
    candidates = [a for a in audits if a["cluster_id"] in train_clusters]
    spans = defaultdict(list)
    coverage, expected_pre, actual_pre = defaultdict(float), defaultdict(int), defaultdict(int)
    for s in segments:
        if assignments[s["source_edf"]] != "train" or s["label"] not in {"preictal", "interictal"}:
            continue
        if s["label"] == "preictal":
            cid = s["cluster_id"]
            coverage[cid] += s["duration_seconds"]
            expected_pre[cid] += max(0, math.floor(s["end_seconds_in_file"] / 4)
                                    - math.ceil(s["start_seconds_in_file"] / 4))
        else:
            cid = None  # Inter độc lập với cụm cơn; chỉ kiểm tra đoạn hợp lệ.
        spans[(s["source_edf"], int(s["label"] == "preictal"))].append((s, cid))

    files, owners = [], defaultdict(set)
    for kind, label in (("inter", 0), ("pre", 1)):
        for path in sorted((DATA_ROOT / patient / kind).glob("*.npz")):
            name = f"{path.stem}.edf"
            if assignments.get(name) != "train":
                raise ValueError(f"{path}: không thuộc EDF train")
            with np.load(path, allow_pickle=False) as data:
                starts, stops = data["start_sample"], data["available_sample"]
                sfreq = float(data["sfreq"])
                if (not len(starts) or len(starts) != len(stops)
                        or len(data["labels"]) != len(starts)
                        or not np.all(data["labels"] == label)
                        or not np.all(stops - starts == round(4 * sfreq))
                        or not np.allclose(starts % (4 * sfreq), 0)
                        or len(np.unique(starts)) != len(starts)):
                    raise ValueError(f"{path}: nhãn hoặc cửa sổ 4 giây không hợp lệ")
                matched = np.zeros(len(starts), dtype=bool)
                cids = set()
                for s, cid in spans[(name, label)]:
                    keep = ((starts >= s["start_seconds_in_file"] * sfreq - 1e-6)
                            & (stops <= s["end_seconds_in_file"] * sfreq + 1e-6))
                    if np.any(matched & keep):
                        raise ValueError(f"{path}: cửa sổ thuộc nhiều nhóm")
                    matched |= keep
                    count = int(keep.sum())
                    if count and label == 1:
                        cids.add(cid)
                        actual_pre[cid] += count
                if not matched.all():
                    raise ValueError(f"{path}: cửa sổ không khớp rules.py/summary")
                if label == 1:
                    owners[name].update(cids)
                files.append(dict(patient=patient, relative_path=path.relative_to(DATA_ROOT).as_posix(),
                                  label=label, windows=len(starts), source_edf=name,
                                  cluster_ids=";".join(sorted(cids))))

    duration = (rules.PRE_START - rules.PRE_END).total_seconds()
    eligible = {a["cluster_id"] for a in candidates if a["eligible_preictal"]
                and duration - coverage[a["cluster_id"]] <= MAX_MISSING_PRE_SECONDS + 1e-6
                and actual_pre[a["cluster_id"]] > 0
                and actual_pre[a["cluster_id"]] == expected_pre[a["cluster_id"]]}
    # Chỉ pre tạo liên kết giữa các cụm. Gộp bắc cầu nếu dùng chung EDF/cụm.
    groups = []
    for name, cids in owners.items():
        names, linked, remaining = {name}, set(cids), []
        pending = groups
        while pending:
            for group_names, group_cids in pending:
                if linked & group_cids:
                    names |= group_names
                    linked |= group_cids
                else:
                    remaining.append((group_names, group_cids))
            if len(remaining) == len(pending):
                break
            pending, remaining = remaining, []
        groups = remaining + [(names, linked)]
    order = {a["cluster_id"]: i for i, a in enumerate(audits)}
    val_groups = sorted((names for names, cids in groups if cids <= eligible),
                        key=lambda names: min(order[c] for name in names for c in owners[name]))
    if len(val_groups) < 2:
        print(f"{patient}: bỏ qua, chỉ có {len(val_groups)} nhóm đủ điều kiện val")
        return []

    pre_counts = {f["source_edf"]: f["windows"] for f in files if f["label"] == 1}
    inter_counts = {f["source_edf"]: f["windows"] for f in files if f["label"] == 0}
    # Inter cùng EDF với pre đi theo nhóm pre; inter còn lại chia độc lập.
    free_inter = sorted(set(inter_counts) - set(owners),
                        key=lambda name: (-inter_counts[name], name))
    group_inter = [sum(inter_counts.get(name, 0) for name in names) for names in val_groups]
    # Mỗi val cần cả hai lớp. Một EDF inter nguyên vẹn chỉ cấp cho một fold.
    inter_capacity = sum(count > 0 for count in group_inter) + len(free_inter)
    n_folds = min(len(val_groups), inter_capacity)
    if K is not None:
        n_folds = min(K, n_folds)
    if n_folds < 2:
        print(f"{patient}: bỏ qua, không đủ EDF inter độc lập để tạo ít nhất 2 fold")
        return []

    fold_edfs = [set() for _ in range(n_folds)]
    fold_pre, fold_inter = [0] * n_folds, [0] * n_folds
    # Rải nhóm có inter trước để tận dụng số EDF inter sẵn có.
    group_order = sorted(range(len(val_groups)), key=lambda i: (
        group_inter[i] == 0, -sum(pre_counts[name] for name in val_groups[i]), i))
    for i in group_order:
        target = min(range(n_folds), key=lambda j: (bool(fold_edfs[j]), fold_pre[j], j))
        fold_edfs[target].update(val_groups[i])
        fold_pre[target] += sum(pre_counts[name] for name in val_groups[i])
        fold_inter[target] += group_inter[i]
    # EDF lớn trước, vào fold có ít cửa sổ inter nhất; kết quả cố định mỗi lần chạy.
    for name in free_inter:
        target = min(range(n_folds), key=lambda j: (fold_inter[j], j))
        fold_edfs[target].add(name)
        fold_inter[target] += inter_counts[name]

    rows = []
    for index, val_edfs in enumerate(fold_edfs):
        for file in files:
            rows.append(dict(file, n_folds=n_folds, fold=index + 1,
                             subset="val" if file["source_edf"] in val_edfs else "train"))
    print(f"{patient}: {len(files)} NPZ, {len(val_groups)} nhóm pre hợp lệ, {n_folds} fold")
    for index in range(n_folds):
        print(f"  val fold {index + 1}: pre={fold_pre[index]}, inter={fold_inter[index]} cửa sổ")
    return rows


def main():
    if K is not None and (isinstance(K, bool) or not isinstance(K, int) or K < 2):
        raise ValueError("K phải là None hoặc số nguyên >= 2")
    if not math.isfinite(MAX_MISSING_PRE_SECONDS) or not 0 <= MAX_MISSING_PRE_SECONDS < 3000:
        raise ValueError("MAX_MISSING_PRE_SECONDS phải nằm trong [0, 3000)")
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)
    patients = PATIENTS if PATIENTS is not None else sorted(p.name for p in DATA_ROOT.glob("chb*") if p.is_dir())
    if not patients or len(set(patients)) != len(patients):
        raise ValueError("Không có bệnh nhân hoặc danh sách trùng lặp")
    rows = [row for patient in patients for row in patient_rows(patient)]
    if not rows:
        raise ValueError("Không có fold hợp lệ; kiểm tra ngưỡng thiếu pre. Giữ nguyên CSV hiện có.")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã lưu {len(rows)} dòng: {CSV_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
