"""Read CHB-MIT summaries that include full dates from EDF headers."""

from datetime import datetime
from pathlib import Path
import re


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
FILE_HEADER = re.compile(r"^File Name:\s*(\S+\.edf)\s*$", re.MULTILINE)


def _field(block, name):
    match = re.search(rf"^{re.escape(name)}:\s*([^\r\n]+)", block, re.MULTILINE)
    if match is None:
        raise ValueError(f"Thiếu trường '{name}' trong summary")
    return match.group(1).strip()


def _seizure_times(block, kind):
    pattern = rf"^Seizure(?: \d+)? {kind} Date and Time:\s*([^\r\n]+)"
    return [
        datetime.strptime(value.strip(), DATE_FORMAT)
        for value in re.findall(pattern, block, re.MULTILINE)
    ]


def getSummary(path):
    """Return EDF metadata with full file and seizure datetimes."""
    text = Path(path).read_text(encoding="utf-8")
    headers = list(FILE_HEADER.finditer(text))
    if not headers:
        raise ValueError(f"Không tìm thấy file EDF trong {path}")

    results = []
    for index, header in enumerate(headers):
        block_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[header.end():block_end]
        file_name = header.group(1)
        file_start = datetime.strptime(_field(block, "File Start Date and Time"), DATE_FORMAT)
        file_end = datetime.strptime(_field(block, "File End Date and Time"), DATE_FORMAT)
        count = int(_field(block, "Number of Seizures in File"))
        if file_end <= file_start:
            raise ValueError(f"{file_name}: thời điểm kết thúc không sau thời điểm bắt đầu")

        start_datetimes = _seizure_times(block, "Start")
        end_datetimes = _seizure_times(block, "End")
        if len(start_datetimes) != count or len(end_datetimes) != count:
            raise ValueError(f"{file_name}: số mốc thời gian cơn không khớp số cơn")
        for seizure_start, seizure_end in zip(start_datetimes, end_datetimes):
            if not file_start <= seizure_start < seizure_end <= file_end:
                raise ValueError(f"{file_name}: cơn nằm ngoài thời gian ghi EDF")

        results.append({
            "file_name": file_name,
            "file_start_time": file_start,
            "file_end_time": file_end,
            "number_of_seizure": count,
            "seizure_start_datetime": start_datetimes,
            "seizure_end_datetime": end_datetimes,
        })
    return results


if __name__ == "__main__":
    dataset_root = Path(__file__).resolve().parents[4] / "chb-mit-scalp-eeg-database-1.0.0"
    summary_path = dataset_root / "chb01" / "chb01-summary-dated.txt"
    from pprint import pprint

    pprint(getSummary(summary_path)[14])
