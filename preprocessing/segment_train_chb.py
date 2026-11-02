"""Xuất cửa sổ train có nhãn vào chbXX/{pre,inter}/tên_EDF.npz.

Label rules come from rules.py and the full dated summary; read only train EDFs.
Use a configurable EDF-aligned window grid, keeping only windows wholly
inside an eligible preictal/interictal interval. No augmentation or z-score here.
"""

import csv
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
import rules
from get_summary_chb import getSummary
from segment_test_chb import export_edf

BASE = Path(__file__).resolve().parent
SPLIT_ROOT = BASE / "edf_split_manual"
OUTPUT_ROOT = BASE / "segment_train_4h_120m"
WINDOW_SECONDS = 30  # Do dai cua so train (giay): vi du 4, 10, 20, 30.
# PATIENTS = ("chb01", ) # None: tất cả bệnh nhân; hoặc ("chb01", "chb17").
PATIENTS = None # None: tất cả bệnh nhân; hoặc ("chb01", "chb17")
LABELS = {"interictal": 0, "preictal": 1}


def print_processing_config(patient):
    """Hiển thị quy tắc thời gian đang dùng trước khi xử lý bệnh nhân."""
    pre_start = rules.PRE_START.total_seconds() / 60
    pre_end = rules.PRE_END.total_seconds() / 60
    exclusion = rules.EXCLUSION.total_seconds() / 3600
    gap_exclusion = rules.GAP_EXCLUSION.total_seconds() / 3600
    session_gap = rules.SESSION_GAP.total_seconds() / 60
    print(
        f"\n[{patient} / train]\n"
        f"  Window: {WINDOW_SECONDS:g} s, non-overlapping.\n"
        f"  Pre   : {pre_start:g} -> {pre_end:g} phút trước cơn đầu cụm "
        f"(dự kiến {pre_start - pre_end:g} phút).\n"
        f"  Inter : loại {exclusion:g} giờ trước cơn đầu "
        f"và {exclusion:g} giờ sau kết thúc cơn cuối cụm.\n"
        f"  Gap > {session_gap:g} phút: loại khỏi inter, "
        f"kèm {gap_exclusion:g} giờ đệm mỗi phía.\n",
        flush=True,
    )


def load_patient(patient):
    folder = SPLIT_ROOT / patient
    with (folder / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assignments = {r["file_name"]: r["split"] for r in rows}
    if len(assignments) != len(rows):
        raise ValueError(f"{patient}: duplicate EDF in manifest")
    records, _ = rules.build_timeline(getSummary(folder / f"{patient}-summary-dated.txt"))
    if set(assignments) - {r["file_name"] for r in records}:
        raise ValueError(f"{patient}: manifest EDF missing from summary")
    if "train" not in assignments.values():
        raise ValueError(f"{patient}: manifest contains no train EDF")
    if set(assignments.values()) - {"train", "test"}:
        raise ValueError(f"{patient}: expected only train/test in manifest")
    names = {name for name, split in assignments.items() if split == "train"}
    actual = {p.name for p in (folder / "train").glob("*.edf")}
    if actual != names:
        raise ValueError(f"{patient}: train directory differs from manifest; "
                         f"missing={sorted(names - actual)}, extra={sorted(actual - names)}")
    # Xét cơn/gap trên summary đầy đủ, sau đó chỉ giữ các đoạn thuộc train.
    _, _, segments, _ = rules.analyze_patient(records, patient)
    spans = defaultdict(list)
    for segment in segments:
        if segment["label"] in LABELS and assignments.get(segment["source_edf"]) == "train":
            spans[segment["source_edf"]].append(segment)
    for record in records:
        split = assignments.get(record["file_name"])
        if split == "train":
            source = folder / split / record["file_name"]
            if not source.is_file():
                raise FileNotFoundError(source)
    return records, assignments, spans


def save_labeled(archive, record, segments, destination, split):
    """Select from a filtered stream; no filter reset at label boundaries."""
    counts = Counter()
    with np.load(archive, allow_pickle=False) as data:
        eeg = data["eeg"]
        starts, stops = data["start_sample"], data["available_sample"]
        sfreq = float(data["sfreq"])
        metadata = {key: data[key] for key in data.files
                    if key not in {"eeg", "start_sample", "available_sample", "preprocessing_json"}}
        config = json.loads(str(data["preprocessing_json"]))
        config["selection"] = "full windows inside eligible labeled spans on EDF-aligned grid"
        config["label_rule_seconds"] = {
            key: getattr(rules, key).total_seconds()
            for key in ("PRE_START", "PRE_END", "POST", "EXCLUSION", "GAP_EXCLUSION", "SESSION_GAP")
        }
        for label, value in LABELS.items():
            keep = np.zeros(len(starts), dtype=bool)
            for segment in segments:
                if segment["label"] == label:
                    first = (segment["start"] - record["file_start_time"]).total_seconds() * sfreq
                    last = (segment["end"] - record["file_start_time"]).total_seconds() * sfreq
                    keep |= (starts >= first - 1e-6) & (stops <= last + 1e-6)
            kind = "pre" if value == 1 else "inter"
            counts[kind] = int(keep.sum())
            if counts[kind]:
                np.savez_compressed(
                    destination / kind / f"{Path(record['file_name']).stem}.npz",
                    eeg=eeg[keep], labels=np.full(counts[kind], value, dtype=np.int8),
                    start_sample=starts[keep], available_sample=stops[keep],
                    split=split, source_split=split,
                    preprocessing_json=json.dumps(config), **metadata,
                )
    return counts


def export_patient(patient, output_root=OUTPUT_ROOT):
    print_processing_config(patient)
    records, assignments, spans = load_patient(patient)
    destination = output_root / patient
    if destination.exists():
        raise FileExistsError(destination)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{patient}_", dir=output_root) as tmp:
        stage = Path(tmp) / patient
        for kind in ("pre", "inter"):
            (stage / kind).mkdir(parents=True)
        state, previous_end, counts = None, None, Counter()
        for record in tqdm(records, desc=f"Processing {patient}"):
            source_name = record["file_name"]
            split = assignments.get(source_name)
            if split != "train":
                continue
            gap = None if previous_end is None else (record["file_start_time"] - previous_end).total_seconds()
            archive = Path(tmp) / "continuous.npz"
            # Giữ trạng thái lọc giữa EDF liền nhau; reset khi có gap như tập test.
            state, previous_end, _ = export_edf(
                SPLIT_ROOT / patient / split / source_name,
                archive, record, state, gap,
                window_seconds=WINDOW_SECONDS,
            )
            counts.update(save_labeled(archive, record, spans[source_name], stage, split))
            archive.unlink()
        totals = {kind: counts[kind] for kind in ("pre", "inter")}
        print(f"{patient}/train: {totals}", flush=True)
        if not counts["pre"] or not counts["inter"]:
            print(f"[NOTE] {patient}: only one class or no eligible windows", flush=True)
        (stage / "counts.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
        stage.rename(destination)
    return totals


if __name__ == "__main__":
    patients = PATIENTS if PATIENTS is not None else sorted(p.name for p in SPLIT_ROOT.glob("chb*") if p.is_dir())
    if not patients:
        raise ValueError(f"No patients in {SPLIT_ROOT}")
    for patient in patients:
        export_patient(patient)
