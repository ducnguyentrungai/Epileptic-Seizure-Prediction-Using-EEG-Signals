# Tạo các khoảng có nhãn EEG từ thông tin thời gian của EDF và các cơn.

"""
Thuật ngữ: onset/offset là thời điểm bắt đầu/kết thúc cơn; với cụm là đầu/cuối
cụm. Preictal là trước cơn, ictal là đang có cơn, postictal là sau cơn.
Interictal là dữ liệu xa các cụm cơn và gap dài theo quy tắc trong tệp này.
Gap là khoảng thiếu bản ghi giữa hai EDF.

Các khoảng được xét theo dạng [bắt đầu, kết thúc): hai khoảng chỉ chạm nhau
không được coi là giao nhau. Độ phủ dữ liệu chỉ dựa vào thời gian của EDF,
không phản ánh chất lượng tín hiệu bên trong tệp.
"""

from datetime import timedelta

# Một giờ được dùng cho ba quy tắc: gộp các cơn gần nhau thành cụm,
# xác định giai đoạn sau cơn cuối cụm và vùng không chắc chắn sau gap dài.
POST = timedelta(hours=1)

# Không lấy interictal trong 4 giờ trước onset đầu và 4 giờ sau offset cuối của cụm cơn.
EXCLUSION = timedelta(hours=4)

# Không lấy interictal trong gap dài, 1 giờ trước đầu gap và 1 giờ sau cuối gap.
GAP_EXCLUSION = timedelta(hours=1)

# Bắt đầu vùng preictal tại thời điểm (32 phút - 2h2m) trước onset đầu của cụm cơn.
PRE_START = timedelta(minutes=122)

# Kết thúc preictal 2 phút trước cơn đầu cụm. Khoảng dự kiến dài (30 phút - > 2 giờ),
# nhưng thời lượng thực có EDF có thể ngắn hơn nếu thiếu dữ liệu được chấp nhận.
PRE_END = timedelta(minutes=2)

# Gap giữa EDF >300 giây được coi là gap dài và tách phiên ghi;
# cũng là ngưỡng thiếu dữ liệu ở biên khiến vùng preictal bị loại.
# Đúng 300 giây chưa vượt ngưỡng; các điều kiện loại khác vẫn được xét.
SESSION_GAP = timedelta(seconds=300)


def seconds(start, end):
    """Tính số giây từ start đến end, giữ cả phần giây lẻ nếu có."""
    return (end - start).total_seconds()



def intersects(a, b):
    """Kiểm tra hai khoảng có phần giao với thời lượng lớn hơn 0 hay không.

        [10:00, 11:00) và [10:30, 11:30) → True
        [10:00, 11:00) và [11:00, 12:00) → False
    """
    return max(a[0], b[0]) < min(a[1], b[1])


def union(intervals):
    """
        Hợp nhất các khoảng thời gian bị chồng lấn hoặc chạm nhau, giữ nguyên các khoảng trống.
        Sắp xếp các khoảng trước khi gộp; bỏ qua khoảng rỗng hoặc có độ dài âm.
        Đầu vào:
        [10:00, 10:30)
        [10:20, 11:00)
        [11:00, 11:20)
        [11:25, 12:00)

        Kết quả:
        [10:00, 11:20)
        [11:25, 12:00)
    """
    merged = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def subtract(interval, exclusions):
    """ 
        Trả về các phần còn lại của interval sau khi trừ các vùng exclusions.
        Gộp các vùng loại trước để xử lý đúng khi chúng chồng lấn nhau.
        
        Khoảng gốc: [08:00, 14:00)
        
        Loại:
        [09:00, 10:00)
        [11:00, 12:00)
        
        Còn:
        [08:00, 09:00)
        [10:00, 11:00)
        [12:00, 14:00)
    """
    start, end = interval
    result = []
    for left, right in union(exclusions):
        if right <= start:
            continue
        if left >= end:
            break
        if start < left:
            result.append((start, left))
            
        # Dời đầu phần còn lại đến sau vùng vừa loại; không lùi về trước.
        start = max(start, right)
        if start >= end:
            break
    if start < end:
        result.append((start, end))
    return result


def covered_parts(interval, records):
    """
        Tìm phần của khoảng cần xét được các bản ghi EDF bao phủ.
        Chỉ dựa vào thời gian của tệp, chưa kiểm tra tín hiệu EEG bên trong.
        - interval: khoảng thời gian cần xét.
        - records: danh sách thông tin các file EDF.
        - Kết quả: danh sách các khoảng có dữ liệu, giữ nguyên khoảng trống giữa các file. 
        
        Khoảng cần xét: 11:08 - 11:58
        
        Thời gian được ghi lại trong các file edf
        EDF A: 10:00 - 11:20
        EDF B: 11:25 - 12:30
        
        Khoảng có dữ liệu:
        11:08 - 11:20
        11:25 - 11:58
    """
    
    return union([
        (max(interval[0], r["file_start_time"]),
         min(interval[1], r["file_end_time"]))
        for r in records
        if intersects(interval, (r["file_start_time"], r["file_end_time"]))
    ])


def build_timeline(records):
    """Sắp xếp EDF, kiểm tra thời gian, ghi lại gap và gán mã phiên ghi.

    Gap dài hơn 300 giây mở phiên mới; gap ngắn vẫn được lưu nhưng không tách phiên.
    Giữ tất cả gap trong danh sách chung để xét ảnh hưởng của chúng qua các phiên.
    Thông tin không hợp lệ gây lỗi và dừng xử lý, không bị âm thầm bỏ qua.
    
    Ví dụ đầu vào:
    Các mốc giờ dưới đây thuộc cùng một ngày và có kiểu datetime.
    Viết gọn thành giờ:phút để dễ theo dõi.

    records = [
        {"file_name": "C.edf",
            "file_start_time": 10:05, "file_end_time": 11:00},

        {"file_name": "A.edf",
            "file_start_time": 08:00, "file_end_time": 09:00},

        {"file_name": "D.edf",
            "file_start_time": 11:10, "file_end_time": 12:00},

        {"file_name": "B.edf",
            "file_start_time": 09:00, "file_end_time": 10:00},
    ]


    Đầu ra là hai danh sách: records, gaps.
    records:
        Giữ các trường ban đầu, sắp xếp lại và thêm session_id:

        [
            {"file_name": "A.edf",
             "file_start_time": 08:00, "file_end_time": 09:00,
             "session_id": 1},

            {"file_name": "B.edf",
             "file_start_time": 09:00, "file_end_time": 10:00,
             "session_id": 1},

            {"file_name": "C.edf",
             "file_start_time": 10:05, "file_end_time": 11:00,
             "session_id": 1},

            {"file_name": "D.edf",
             "file_start_time": 11:10, "file_end_time": 12:00,
             "session_id": 2},
        ]

    gaps:
        [
            {"gap_id": "gap_001",
             "start": 10:00, "end": 10:05,
             "duration_seconds": 300.0,
             "long_gap": False,
             "previous_edf": "B.edf",
             "next_edf": "C.edf"},

            {"gap_id": "gap_002",
             "start": 11:00, "end": 11:10,
             "duration_seconds": 600.0,
             "long_gap": True,
             "previous_edf": "C.edf",
             "next_edf": "D.edf"},
        ]

    """
    # Sắp theo thời điểm bắt đầu; sao chép từng từ điển để không thêm mã phiên vào bản gốc.
    records = [dict(r) for r in sorted(records, key=lambda r: r["file_start_time"])]
    if not records:
        raise ValueError("Empty recording timeline")
    if len({r["file_name"] for r in records}) != len(records):
        raise ValueError("Duplicate file names in summary")
    
    gaps = []
    session = 1
    for index, record in enumerate(records):
        # Báo lỗi nếu thời lượng EDF bằng 0 hoặc âm.
        if record["file_end_time"] <= record["file_start_time"]:
            raise ValueError(f"Invalid duration: {record['file_name']}")
        if index:
            previous = records[index - 1]
            start, end = previous["file_end_time"], record["file_start_time"]   # lấy thời gian kết thúc file trước và bắt đầu file sau
            if end < start:
                raise ValueError(
                    f"Overlapping EDF times: {previous['file_name']} / {record['file_name']}; "
                    "resolve chronology before assigning labels"
                )
            if end > start:
                long_gap = end - start > SESSION_GAP    # Chỉ gap lớn hơn 300 giây mới là gap dài.
                
                gaps.append({"gap_id": f"gap_{len(gaps)+1:03d}",
                             "start": start, "end": end,
                             "duration_seconds": seconds(start, end),
                             "long_gap": long_gap,
                             "previous_edf": previous["file_name"],
                             "next_edf": record["file_name"]})
                
                if long_gap:    # gap lớn tách thành session mới
                    session += 1
        record["session_id"] = session
    return records, gaps


def build_clusters(records):
    """Kiểm tra các cơn rồi gộp những cơn gần nhau thành cụm theo thời gian.

    Gộp nếu cơn tiếp theo bắt đầu không muộn hơn cuối cụm hiện tại cộng POST.
    Cuối cụm được cập nhật sau mỗi lần gộp, nên một cụm có thể dài hơn 1 giờ.
    Việc gộp không bị giới hạn bởi mã phiên ghi. offset của cụm là cuối cơn
    muộn nhất, chưa cộng thêm thời gian sau cơn.
    
    Ví dụ đầu vào (viết gọn datetime thành giờ, các mốc cùng ngày):
        records = [{
            "file_name": "A.edf",
            "file_start_time": 08:00,
            "file_end_time": 14:00,
            "number_of_seizure": 3,
            "seizure_start_datetime": [10:00, 10:40, 12:00],
            "seizure_end_datetime":   [10:02, 10:42, 12:02],
        }]

    Đầu ra (viết gọn danh sách seizures):
    [
        {
            "cluster_id": "cluster_001",
            "onset": 10:00,
            "offset": 10:42,
            "seizures": [
                {
                    "source_edf": "A.edf",
                    "seizure_number_in_file": 1,
                    "onset": 10:00,
                    "offset": 10:02,
                },
                {
                    "source_edf": "A.edf",
                    "seizure_number_in_file": 2,
                    "onset": 10:40,
                    "offset": 10:42,
                },
            ],
        },
        {
            "cluster_id": "cluster_002",
            "onset": 12:00,
            "offset": 12:02,
            "seizures": [
                {
                    "source_edf": "A.edf",
                    "seizure_number_in_file": 3,
                    "onset": 12:00,
                    "offset": 12:02,
                },
            ],
        },
    ]
    """
    seizures = []
    # Kiểm tra thông tin cơn; gặp dữ liệu không hợp lệ thì báo lỗi, không bỏ qua.
    for record in records:
        starts, ends = record["seizure_start_datetime"], record["seizure_end_datetime"] # Danh sách đầu/cuối các cơn.
        
        # Số thời điểm bắt đầu, kết thúc và số cơn khai báo phải khớp nhau.
        if len(starts) != len(ends) or len(starts) != record["number_of_seizure"]:
            raise ValueError(f"Invalid seizure count: {record['file_name']}")
        
        for number, (onset, offset) in enumerate(zip(starts, ends), 1): # lấy ra thời gian từng cơn
            
            # Cơn phải dài hơn 0 và nằm trọn trong EDF; nếu sai thì báo lỗi.
            if not record["file_start_time"] <= onset < offset <= record["file_end_time"]:
                raise ValueError(f"Invalid seizure interval: {record['file_name']}")
            
            seizures.append({"source_edf"            : record["file_name"],
                             "seizure_number_in_file": number,
                             "onset"                 : onset, 
                             "offset"                : offset})
            
    clusters = []
    for seizure in sorted(seizures, key=lambda s: (s["onset"], s["offset"])):
        # Gộp nếu đầu cơn không muộn hơn cuối cụm cộng 1 giờ, kể cả đúng mốc đó.
        if clusters and seizure["onset"] <= clusters[-1]["offset"] + POST:
            clusters[-1]["seizures"].append(seizure)    # lưu lại cơn hiện tại 
            clusters[-1]["offset"] = max(clusters[-1]["offset"], seizure["offset"]) # Giữ cuối cơn muộn nhất, chưa cộng POST.
            
        else:
            clusters.append({"cluster_id": f"cluster_{len(clusters)+1:03d}",    
                             "onset": seizure["onset"], 
                             "offset": seizure["offset"],
                             "seizures": [seizure]})
    return clusters


def analyze_patient(records, patient_id, min_clusters=3, min_interictal_minutes=0):
    """Đánh giá từng cụm, tạo các đoạn có nhãn và tổng hợp điều kiện chọn bệnh nhân.

    Trả về lần lượt: thống kê bệnh nhân, kết quả kiểm tra từng cụm, danh sách
    đoạn có nhãn và danh sách gap. Mỗi đoạn chỉ thuộc một EDF.
    selected chỉ là kết quả đánh giá; dù selected=False, hàm vẫn trả các đoạn
    đã tạo. Bên gọi tự quyết định có sử dụng dữ liệu bệnh nhân đó hay không.
    """
    records, gaps = build_timeline(records)      # Sắp theo thời gian, gán phiên và tìm gap.
    clusters      = build_clusters(records)      # sắp xếp các cơn và phân thành các cụm
    
    long_gaps = [g for g in gaps if g["long_gap"]]  # Chỉ giữ gap dài hơn 300 giây.
    
    # Mỗi vùng kéo dài từ đầu cụm đến 1 giờ sau cuối cụm: gồm các cơn, khoảng giữa các cơn trong cùng cụm và giai đoạn sau cơn cuối.
    # Dùng các vùng này để tránh gán preictal chồng lên sự kiện đã ghi nhận.
    known_event_regions = [(c["onset"], c["offset"] + POST) for c in clusters]
    audits = []
    for cluster in clusters:
        # Xác định khoảng từ 52 đến 2 phút trước cơn đầu cụm; chưa đọc dữ liệu EEG.
        pre = (cluster["onset"] - PRE_START, cluster["onset"] - PRE_END)
        
        # Tìm các phần trong khoảng preictal có dữ liệu EDF.
        parts = covered_parts(pre, records) 
        
        # Tìm các phần thiếu dữ liệu bằng cách trừ các phần có dữ liệu khỏi khoảng pre-ictal. 
        missing = subtract(pre, parts)
       
        # Nếu preictal có phần nằm trong gap giữa hai EDF:
        # - Gap dài tối đa 5 phút: bỏ phần thiếu dữ liệu, các phần còn lại vẫn có thể dùng nếu đạt các điều kiện kiểm tra khác.
        # - Gap dài hơn 5 phút: loại cả khoảng preictal của cụm này.
        # So sánh 5 phút với độ dài toàn bộ gap, không chỉ phần nằm trong preictal.
        long_gaps_in_pre = [g["gap_id"] for g in long_gaps
                            if intersects(pre, (g["start"], g["end"]))]
        
        # Lấy các khoảng thiếu dữ liệu không giao với bất kỳ gap nào giữa các EDF.
        # Đây là phần thiếu ở biên: trước EDF đầu tiên hoặc sau EDF cuối cùng.
        boundary_missing = [(a, b) for a, b in missing
                            if not any(intersects((a, b), (g["start"], g["end"]))
                                       for g in gaps)]
        
        
        # Loại preictal vì thiếu dữ liệu nếu gặp một trong hai trường hợp:
        # 1. Preictal giao với gap giữa hai EDF dài hơn 5 phút.
        # 2. Preictal bị thiếu hơn 5 phút trước EDF đầu tiên hoặc sau EDF cuối cùng.
        # Chỉ cần một trường hợp xảy ra thì partial=True.
        # partial=False vẫn có thể thiếu dữ liệu, nhưng chưa vượt ngưỡng trên.
        partial = bool(long_gaps_in_pre or any(
            seconds(a, b) > SESSION_GAP.total_seconds()
            for a, b in boundary_missing
        ))

        # Lấy ID các gap dài có vùng 1 giờ sau khi gap kết thúc giao với preictal.
        # Preictal trong vùng này bị coi là không chắc chắn vì gap có thể chứa cơn chưa ghi nhận.
        uncertain_gaps = [
            g["gap_id"] for g in long_gaps
            if intersects(pre, (g["end"], g["end"] + POST))
        ]

        # Lấy ID các gap dài mà onset đầu cụm nằm trong vòng 1 giờ sau cuối gap.
        # Tính cả hai mốc biên; cụm này bị coi là không chắc chắn.
        cluster_gaps = [
            g["gap_id"] for g in long_gaps
            if g["end"] <= cluster["onset"] <= g["end"] + POST
        ]

        # Lấy ID các cụm có vùng từ onset đầu đến 1 giờ sau offset cuối giao với preictal.
        # Vùng này gồm các cơn, khoảng giữa các cơn trong cụm và giai đoạn sau cơn.
        event_overlap = [
            c["cluster_id"] for c in clusters
            if intersects(pre, (c["onset"], c["offset"] + POST))
        ]

        # Ghi lại các lý do loại preictal của cụm hiện tại.
        # Một cụm có thể có nhiều lý do; danh sách rỗng nghĩa là preictal hợp lệ.
        reasons = []

        if partial:
            # Thiếu dữ liệu vượt ngưỡng cho phép.
            reasons.append("partial_preictal")

        if uncertain_gaps:
            # Preictal giao vùng không chắc chắn sau gap dài.
            reasons.append("uncertain_preictal")

        if event_overlap:
            # Preictal chồng lên vùng cơn, giữa các cơn hoặc sau cơn.
            reasons.append("overlap_confirmed_ictal_between_seizures_or_postictal")

        if cluster_gaps:
            # Onset đầu cụm nằm trong vùng không chắc chắn sau gap dài.
            reasons.append("uncertain_cluster")
            
        # Lưu thông tin để giải thích vì sao nhận/loại preictal của từng cụm.
        # Thời lượng có EDF được lưu cả khi cụm bị loại; danh sách tệp chỉ biểu thị
        # giao thời gian, không bảo đảm preictal đã được chấp nhận.
        audits.append({
            "patient_id": patient_id, "cluster_id": cluster["cluster_id"],
            "onset": cluster["onset"], "offset": cluster["offset"],
            "seizure_count": len(cluster["seizures"]),
            "seizure_files": sorted({s["source_edf"] for s in cluster["seizures"]}),
            "preictal_start": pre[0], "preictal_end": pre[1],
            "preictal_files": [r["file_name"] for r in records
                               if intersects(pre, (r["file_start_time"], r["file_end_time"]))],
            "covered_preictal_seconds": sum(seconds(a, b) for a, b in parts),
            "missing_intervals": missing,
            "long_gap_ids_in_preictal": long_gaps_in_pre,
            "partial_preictal": partial,
            "uncertain_preictal": bool(uncertain_gaps),
            "uncertain_cluster": bool(cluster_gaps),
            "uncertain_preictal_gap_ids": uncertain_gaps,
            "uncertain_cluster_gap_ids": cluster_gaps,
            "overlap_cluster_ids": event_overlap,
            "eligible_preictal": not reasons,
            "exclusion_reasons": reasons,
        })

    # Loại khỏi interictal vùng từ 2 giờ trước đầu cụm đến 2 giờ sau cuối cụm.
    # Áp dụng với mọi cụm, kể cả cụm có preictal không hợp lệ.
    inter_exclusions = [(c["onset"] - EXCLUSION, c["offset"] + EXCLUSION)
                        for c in clusters]
    # Loại thêm gap dài cùng vùng đệm 1 giờ ở mỗi phía.
    inter_exclusions += [(g["start"] - GAP_EXCLUSION, g["end"] + GAP_EXCLUSION)
                         for g in long_gaps]
    segments = []
    def add_segments(label, interval, cluster_id=""):
        """Giao khoảng cần gán nhãn với từng EDF và lưu phần có thời lượng dương."""
        # Luôn tách tại biên EDF, kể cả khi hai tệp nối liền không có gap.
        # Chỉ tạo thông tin đoạn, chưa đọc hoặc cắt tín hiệu EEG tại đây.
        for record in records:
            start = max(interval[0], record["file_start_time"])
            end = min(interval[1], record["file_end_time"])
            if start < end:
                segments.append({
                    "patient_id": patient_id, "session_id": record["session_id"],
                    "source_edf": record["file_name"], "label": label,
                    "cluster_id": cluster_id, "start": start, "end": end,
                    # Vị trí đầu/cuối tính bằng giây từ đầu EDF, dùng để cắt tín hiệu sau này.
                    "start_seconds_in_file": seconds(record["file_start_time"], start),
                    "end_seconds_in_file": seconds(record["file_start_time"], end),
                    "duration_seconds": seconds(start, end),
                })

    for cluster, audit in zip(clusters, audits):
        cid = cluster["cluster_id"]
        
        # Chỉ gán ictal cho từng cơn, không gán ictal cho khoảng giữa các cơn.
        for seizure in cluster["seizures"]:
            add_segments("ictal", (seizure["onset"], seizure["offset"]), cid)
            
        # Xét 1 giờ sau cơn cuối cụm để tạo postictal; trừ vùng sự kiện của cụm khác
        # nếu có giao nhau. add_segments chỉ giữ phần có EDF.
        # Preictal giao vùng sự kiện đã bị loại ở bước kiểm tra phía trên.
        # Không suy đoán nhãn postictal cho cơn có thể đã xảy ra trong gap.
        other_events = [interval for c, interval in zip(clusters, known_event_regions)
                        if c["cluster_id"] != cid]
        
        for interval in subtract((cluster["offset"], cluster["offset"] + POST), other_events):
            add_segments("postictal", interval, cid)
            
        # Chỉ tạo preictal khi cụm không có lý do loại nào.
        if audit["eligible_preictal"]:
            add_segments("preictal", (audit["preictal_start"], audit["preictal_end"]), cid)
            
    # Interictal là phần EDF còn lại sau khi trừ các vùng loại.
    # Không mặc định mọi thời gian chưa có nhãn đều là interictal.
    for record in records:
        for interval in subtract((record["file_start_time"], record["file_end_time"]),
                                 inter_exclusions):
            add_segments("interictal", interval)

    # Sắp theo thời gian rồi kiểm tra các đoạn không chồng lấn.
    # Chỉ chạm nhau thì được phép; chồng lấn thì báo lỗi, không ghi đè nhãn.
    segments.sort(key=lambda s: (s["start"], s["end"], s["label"]))
    for previous, current in zip(segments, segments[1:]):
        if previous["end"] > current["start"]:
            raise ValueError(f"Overlapping labels: {patient_id}, {previous}, {current}")
    # Đếm cụm có preictal hợp lệ (True được tính là 1), không đếm số cơn riêng lẻ.
    eligible = sum(a["eligible_preictal"] for a in audits)
    inter_seconds = sum(s["duration_seconds"] for s in segments if s["label"] == "interictal")
    reasons = []
    # Tiêu chí cấp bệnh nhân: đủ số cụm hợp lệ và đủ thời lượng interictal.
    if eligible < min_clusters:
        reasons.append("fewer_than_required_eligible_clusters")
    # Dù ngưỡng phút bằng 0, vẫn yêu cầu có interictal với thời lượng dương.
    if inter_seconds <= 0 or inter_seconds < min_interictal_minutes * 60:
        reasons.append("insufficient_interictal")
    # Preictal chỉ cộng thời lượng có EDF của cụm hợp lệ, không lấy số cụm nhân 50 phút.
    # selected=False không xóa các đoạn có nhãn đã tạo.
    patient = {"patient_id": patient_id, "recorded_seizures": sum(c["seizure_count"] for c in audits),
               "clusters": len(clusters), "eligible_clusters": eligible,
               "preictal_minutes": sum(a["covered_preictal_seconds"] for a in audits
                                        if a["eligible_preictal"]) / 60,
               "interictal_minutes": inter_seconds / 60,
               "selected": not reasons, "exclusion_reasons": reasons}
    return patient, audits, segments, gaps


