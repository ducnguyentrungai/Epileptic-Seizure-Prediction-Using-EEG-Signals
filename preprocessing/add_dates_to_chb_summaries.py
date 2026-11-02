"""Create compact dated CHB-MIT summaries without changing the originals."""

import re
from datetime import datetime, timedelta
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[4] / "chb-mit-scalp-eeg-database-1.0.0"
FILE_NAME = re.compile(r"^File Name:[ \t]*(\S+\.edf)[ \t]*$", re.MULTILINE)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def edf_times(path: Path) -> tuple[datetime, datetime]:
    with path.open("rb") as file:
        header = file.read(256)
    if len(header) < 256:
        raise ValueError(f"Invalid EDF header: {path}")

    day, month, yy = map(int, header[168:176].decode("ascii").split("."))
    year = (2000 if yy <= 84 else 1900) + yy  # EDF two-digit year rule
    hour, minute, second = map(int, header[176:184].decode("ascii").split("."))
    start = datetime(year, month, day, hour, minute, second)
    records = int(header[236:244].decode("ascii"))
    seconds_per_record = float(header[244:252].decode("ascii"))
    if records < 0 or seconds_per_record <= 0:
        raise ValueError(f"Invalid EDF duration: {path}")
    return start, start + timedelta(seconds=records * seconds_per_record)


def render_summary(source: Path) -> bytes:
    original = source.read_text(encoding="ascii")
    headers = list(FILE_NAME.finditer(original))
    if not headers:
        raise ValueError(f"No EDF files listed in {source}")

    lines = ["Dates below are anonymized surrogate dates from EDF headers.", ""]
    for index, header in enumerate(headers):
        block_end = headers[index + 1].start() if index + 1 < len(headers) else len(original)
        block = original[header.end():block_end]
        file_name = header.group(1)
        start, end = edf_times(source.parent / file_name)
        count_match = re.search(r"^Number of Seizures in File:[ \t]*(\d+)", block, re.MULTILINE)
        if count_match is None:
            raise ValueError(f"{file_name}: missing seizure count")
        count = int(count_match.group(1))
        start_offsets = [int(value) for value in re.findall(
            r"^Seizure(?: \d+)? Start Time:[ \t]*(\d+) seconds", block, re.MULTILINE
        )]
        end_offsets = [int(value) for value in re.findall(
            r"^Seizure(?: \d+)? End Time:[ \t]*(\d+) seconds", block, re.MULTILINE
        )]
        if len(start_offsets) != count or len(end_offsets) != count:
            raise ValueError(f"{file_name}: seizure count does not match timestamps")

        lines.extend((
            f"File Name: {file_name}",
            f"File Start Date and Time: {start:{DATE_FORMAT}}",
            f"File End Date and Time: {end:{DATE_FORMAT}}",
            f"Number of Seizures in File: {count}",
        ))
        for seizure_number, (start_offset, end_offset) in enumerate(
            zip(start_offsets, end_offsets), start=1
        ):
            seizure_start = start + timedelta(seconds=start_offset)
            seizure_end = start + timedelta(seconds=end_offset)
            if not start <= seizure_start < seizure_end <= end:
                raise ValueError(f"{file_name}: seizure outside EDF time range")
            prefix = f"Seizure {seizure_number}" if count > 1 else "Seizure"
            lines.extend((
                f"{prefix} Start Date and Time: {seizure_start:{DATE_FORMAT}}",
                f"{prefix} End Date and Time: {seizure_end:{DATE_FORMAT}}",
            ))
        lines.append("")
    return ("\n".join(lines) + "\n").encode("ascii")


def create_summary(source: Path) -> Path:
    destination = source.with_name(source.stem + "-dated.txt")
    destination.write_bytes(render_summary(source))
    return destination


if __name__ == "__main__":
    sources = sorted(DATA_DIR.glob("chb*/chb*-summary.txt"))
    if not sources:
        raise SystemExit(f"No summaries found in {DATA_DIR}")
    rendered = [(source, render_summary(source)) for source in sources]
    for source, content in rendered:
        destination = source.with_name(source.stem + "-dated.txt")
        destination.write_bytes(content)
        print(destination)
