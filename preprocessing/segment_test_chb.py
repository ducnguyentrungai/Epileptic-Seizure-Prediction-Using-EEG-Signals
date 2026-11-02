"""Export continuous 4-second windows from a configurable split folder.

Read EDFs, manifest and dated summary only from edf_split_manual/chbXX/.
Edit constants and run directly. Store seizure events for later SPH/SOP alarm
evaluation, without window labels or label-based selection. Filtering is causal;
existing training exports use a different,
zero-phase filter and must be regenerated consistently for a streaming experiment.
"""

import csv
import json
import tempfile
from datetime import timedelta
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from tqdm import tqdm

from get_summary_chb import getSummary
from rules import EXCLUSION

BASE = Path(__file__).resolve().parent
SPLIT_ROOT = BASE / "edf_split_manual"
INPUT_SPLIT = "test"  # Thư mục con trong chbXX và giá trị split trong manifest: train/test/val.
OUTPUT_ROOT = BASE / "test_continuous"
PATIENTS = ("chb01", "chb05", "chb06", "chb09")  # None: tất cả bệnh nhân; hoặc ("chb01", "chb17").
WINDOW_SECONDS = 30
BATCH_WINDOWS = 128
NOTCH_BANDS_HZ = ((57.0, 63.0), (117.0, 123.0))
SESSION_GAP = timedelta(seconds=300)
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ",
]


def print_processing_config(patient):
    """Hiển thị thư mục nguồn và cấu hình trước khi xử lý bệnh nhân."""
    print(
        f"\n[{patient} / {INPUT_SPLIT}]\n"
        f"  Nguồn: {SPLIT_ROOT / patient / INPUT_SPLIT}\n"
        f"  Đích: {OUTPUT_ROOT / patient}\n"
        f"  Đầu dữ liệu: bỏ phần còn trong {EXCLUSION.total_seconds() / 3600:g} giờ sau cơn trước.\n"
        "  Phần còn lại: giữ liên tục, gồm cả các cơn trong thư mục đã chọn.\n"
        f"  Cửa sổ: {WINDOW_SECONDS:g} giây, không chồng lấn; {len(CHANNELS)} kênh.\n"
        f"  Gap > {SESSION_GAP.total_seconds() / 60:g} phút: tách phiên.\n"
        "  Bộ lọc: reset khi có gap giữa các EDF.\n",
        flush=True,
    )


def channel_plan(raw, file_name):
    """Lấy 18 kênh bipolar; tạo từ điện cực đơn cực nếu cần."""
    aliases = {"T8-P8": ("T8-P8", "T8-P8-0")}
    direct = [next((n for n in aliases.get(c, (c,)) if n in raw.ch_names), None)
              for c in CHANNELS]
    if all(direct):
        return direct, None

    def electrode(name):
        variants = (name, f"{name}-CS2")
        if name == "O1":
            variants += ("01", "01-CS2")
        return next((n for n in variants if n in raw.ch_names), None)

    pairs = []
    for channel in CHANNELS:
        pair = tuple(electrode(n) for n in channel.split("-"))
        if None in pair:
            raise ValueError(f"{file_name}: cannot construct channel {channel}")
        pairs.append(pair)
    return list(dict.fromkeys(n for pair in pairs for n in pair)), pairs


def notch_filters(sfreq):
    if sfreq / 2 <= NOTCH_BANDS_HZ[-1][1]:
        raise ValueError(f"Sampling rate {sfreq} Hz is too low for the notch bands")
    return [butter(4, band, btype="bandstop", fs=sfreq, output="sos")
            for band in NOTCH_BANDS_HZ]


def build_timeline(records):
    """Đánh số phiên theo summary đầy đủ; tách phiên khi gap > 300 giây."""
    records = [dict(r) for r in sorted(records, key=lambda r: r["file_start_time"])]
    if not records or len({r["file_name"] for r in records}) != len(records):
        raise ValueError("Empty summary or duplicate EDF names")
    session = 1
    for index, record in enumerate(records):
        if index:
            gap = record["file_start_time"] - records[index - 1]["file_end_time"]
            if gap < timedelta(0):
                raise ValueError(f"Overlapping EDF times: {record['file_name']}")
            session += gap > SESSION_GAP
        record["session_id"] = session
    return records


def export_edf(source, destination, record, state=None,
               gap_seconds=None, patient_id=None, start_seconds=0,
               window_seconds=None):
    """Return final filter state for a directly contiguous, compatible next EDF."""
    if window_seconds is None:
        window_seconds = WINDOW_SECONDS
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_edf(source, preload=False, verbose="ERROR")
    try:
        sfreq = float(raw.info["sfreq"])
        if sfreq != 256:
            raise ValueError(f"{source}: expected 256 Hz, got {sfreq}")
        width = round(window_seconds * sfreq)
        if width < 1:
            raise ValueError("window_seconds must correspond to at least one sample")
        first_sample = int(np.ceil(start_seconds * sfreq))
        if not 0 <= first_sample <= raw.n_times:
            raise ValueError(f"{source}: invalid start_seconds={start_seconds}")
        duration = raw.n_times / sfreq
        expected = (record["file_end_time"] - record["file_start_time"]).total_seconds()
        if abs(duration - expected) > 1 / sfreq:
            raise ValueError(f"{source}: EDF duration differs from summary")
        selected, pairs = channel_plan(raw, source.name)
        raw.pick(selected)
        indices = {name: i for i, name in enumerate(raw.ch_names)}
        sos = np.concatenate(notch_filters(sfreq))
        reset = state is None or gap_seconds != 0 or first_sample > 0
        zi = None if reset else state
        starts = np.arange(first_sample, raw.n_times - width + 1, width, dtype=np.int64)
        seizure_intervals = np.array([
            [round((a - record["file_start_time"]).total_seconds() * sfreq),
             round((b - record["file_start_time"]).total_seconds() * sfreq)]
            for a, b in zip(record["seizure_start_datetime"], record["seizure_end_datetime"])
        ], dtype=np.int64).reshape(-1, 2)
        config = {
            "schema_version": 2, "filter": "causal Butterworth bandstop, order 4 per band",
            "notch_bands_hz": NOTCH_BANDS_HZ, "sos": sos.tolist(),
            "initial_state": "steady state from first sample after reset",
            "filter_state_reset": reset, "future_padding": False,
            "dc_removal": "per channel per window", "normalization": "none",
            "units": "V", "augmentation": False,
            "window_seconds": width / sfreq,
            "window_boundary": "never cross EDF; discard trailing incomplete window",
            "trimmed_prefix_seconds": first_sample / sfreq,
            "evaluation": "event-based alarms with SPH/SOP; configured separately",
            "prediction_time": "file_start_datetime + available_sample / sfreq; processing latency excluded",
        }
        with tempfile.TemporaryDirectory(prefix=".test_", dir=destination.parent) as tmp:
            shape = (len(starts), len(CHANNELS), width)
            eeg = (np.lib.format.open_memmap(Path(tmp) / "eeg.npy", mode="w+",
                                            dtype=np.float32, shape=shape)
                   if len(starts) else np.empty(shape, dtype=np.float32))
            chunk_size = BATCH_WINDOWS * width
            for first in range(first_sample, raw.n_times, chunk_size):
                last = min(first + chunk_size, raw.n_times)
                signal = raw.get_data(start=first, stop=last)
                if pairs is not None:
                    signal = np.stack([signal[indices[a]] - signal[indices[b]] for a, b in pairs])
                if zi is None:
                    zi = sosfilt_zi(sos)[:, None, :] * signal[None, :, :1]
                signal, zi = sosfilt(sos, signal, axis=-1, zi=zi)
                count = (last - first) // width
                if count:
                    windows = signal[:, :count * width].reshape(len(CHANNELS), count, width).transpose(1, 0, 2)
                    windows = windows - windows.mean(axis=-1, keepdims=True)
                    row = (first - first_sample) // width
                    eeg[row:row + count] = windows.astype(np.float32)
            archive = Path(tmp) / "result.npz"
            np.savez_compressed(
                archive, eeg=eeg, start_sample=starts, available_sample=starts + width,
                file_start_datetime=record["file_start_time"].isoformat(),
                file_end_datetime=record["file_end_time"].isoformat(),
                sfreq=sfreq, step_samples=width, source_edf=source.name,
                patient_id=patient_id or source.parent.parent.name, channels=np.asarray(CHANNELS),
                session_id=record["session_id"], n_samples=raw.n_times,
                dropped_prefix_samples=first_sample,
                dropped_tail_samples=(raw.n_times - first_sample) % width,
                previous_gap_seconds=np.nan if gap_seconds is None else gap_seconds,
                seizure_intervals=seizure_intervals,
                preprocessing_json=json.dumps(config),
            )
            del eeg  # Release Windows memory map before removing temporary files.
            archive.rename(destination)
        return zi, record["file_start_time"] + timedelta(seconds=duration), len(starts)
    finally:
        raw.close()


def records_for_patient(patient):
    """Đối chiếu manifest, summary và thư mục INPUT_SPLIT trước khi xuất."""
    folder = SPLIT_ROOT / patient
    manifest = folder / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [r["file_name"] for r in rows if r["split"] == INPUT_SPLIT]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{manifest}: no {INPUT_SPLIT} EDF or duplicate {INPUT_SPLIT} entries")
    if set(names) & {r["file_name"] for r in rows if r["split"] != INPUT_SPLIT}:
        raise ValueError(f"{manifest}: EDF assigned to multiple splits")
    records = getSummary(folder / f"{patient}-summary-dated.txt")
    records = build_timeline(records)
    names_set = set(names)
    selected_records = [r for r in records if r["file_name"] in names_set]
    if len(selected_records) != len(names):
        raise ValueError(f"{manifest}: {INPUT_SPLIT} EDF absent from full summary")
    actual = {p.name for p in (folder / INPUT_SPLIT).glob("*.edf")}
    if actual != names_set:
        raise ValueError(f"{patient}: {INPUT_SPLIT} directory differs from manifest; "
                         f"missing={sorted(names_set - actual)}, extra={sorted(actual - names_set)}")
    split_start = selected_records[0]["file_start_time"]
    previous_end = max((end for r in records for end in r["seizure_end_datetime"]
                        if end <= split_start), default=None)
    # Chỉ cắt đầu dữ liệu theo cơn trước mốc chia; giữ các cơn về sau.
    cutoff = max(split_start, previous_end + EXCLUSION) if previous_end else split_start
    for record in selected_records:
        record["trim_start_seconds"] = max(
            0, (cutoff - record["file_start_time"]).total_seconds())
    return selected_records


def export_patient(patient, output_root=OUTPUT_ROOT):
    print_processing_config(patient)
    folder = SPLIT_ROOT / patient
    selected_records = records_for_patient(patient)
    for r in selected_records:
        source = folder / INPUT_SPLIT / r["file_name"]
        destination = output_root / patient / f"{source.stem}.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
    state, previous_end = None, None
    for r in selected_records:
        trim = r["trim_start_seconds"]
        duration = (r["file_end_time"] - r["file_start_time"]).total_seconds()
        if trim >= duration:
            print(f"{patient}/{r['file_name']}: bỏ toàn bộ (postictal cơn trước)", flush=True)
            continue
        if trim:
            print(f"{patient}/{r['file_name']}: bỏ {trim:g} giây đầu "
                  f"({trim / 60:.2f} phút postictal cơn trước)", flush=True)
        source = folder / INPUT_SPLIT / r["file_name"]
        destination = output_root / patient / f"{source.stem}.npz"
        gap = None if previous_end is None else (r["file_start_time"] - previous_end).total_seconds()
        state, previous_end, count = export_edf(
            source, destination, r, state, gap, start_seconds=trim)
        print(f"{patient}/{source.name}: {count} windows; {len(r['seizure_start_datetime'])} seizures", flush=True)


if __name__ == "__main__":
    if WINDOW_SECONDS <= 0 or BATCH_WINDOWS < 1:
        raise ValueError("Window and batch sizes must be positive")
    patients = PATIENTS if PATIENTS is not None else sorted(
        p.name for p in SPLIT_ROOT.glob("chb*") if p.is_dir())
    if not patients:
        raise ValueError(f"No patients found in {SPLIT_ROOT}")
    for patient in tqdm(patients):
        export_patient(patient)
