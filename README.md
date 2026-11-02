# Dự đoán trạng thái tiền cơn động kinh sử dụng tín hiệu điện não EEG

**Epileptic Seizure Prediction Using EEG Signals**

Đồ án sử dụng dữ liệu CHB-MIT để xây dựng bài toán phân biệt **tiền cơn động kinh (preictal)** và **giữa các cơn (interictal)**, với kiến trúc mô hình DBConformer.

**Môi trường huấn luyện:** Kaggle Notebook. Quy trình của đồ án gồm chuẩn bị dữ liệu trên máy cá nhân, sau đó sử dụng dữ liệu đã xử lý để huấn luyện mô hình trên Kaggle.

Repository hiện có mã chuẩn bị dữ liệu và định nghĩa mô hình. Mã huấn luyện trên Kaggle được cung cấp tại [DBConformer with EEG Dataset](https://www.kaggle.com/code/nguyentduc/dbconformer-with-eeg-dataset). Các lệnh chạy trên máy cá nhân dưới đây hướng dẫn đến bước xuất dữ liệu và tạo danh sách train/validation.

## 1. Nguồn dữ liệu và mô hình

| Thành phần | Nguồn | Sử dụng trong đồ án |
| --- | --- | --- |
| CHB-MIT Scalp EEG Database, phiên bản 1.0.0 | [PhysioNet — CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) | Tín hiệu EEG trong file `.edf` và thông tin cơn trong `chbXX-summary.txt` |
| DBConformer | [wzwvv/DBConformer](https://github.com/wzwvv/DBConformer) | Mã kiến trúc tại [`models/DBConformer.py`](models/DBConformer.py) được lấy từ dự án này |

CHB-MIT có tần số lấy mẫu 256 Hz. Ngày tháng trong EDF đã được thay bằng ngày giả để bảo vệ thông tin người tham gia; quan hệ thời gian giữa các bản ghi trong cùng một case được giữ lại. Chi tiết nằm trong [mô tả dữ liệu của PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

DBConformer là **Dual-Branch Convolutional Transformer for EEG Decoding**, gồm nhánh xử lý đặc trưng thời gian và nhánh xử lý đặc trưng không gian. Đây là kiến trúc kế thừa từ công trình gốc; quy tắc gán nhãn preictal/interictal trong repository này được định nghĩa riêng trong `preprocessing/rules.py`. Xem [repository gốc](https://github.com/wzwvv/DBConformer) để biết kiến trúc và cách trích dẫn.

## 2. Cấu trúc mã nguồn

```text
Epileptic-Seizure-Prediction-Using-EEG-Signals/
├── README.md
├── models/
│   └── DBConformer.py
└── preprocessing/
    ├── add_dates_to_chb_summaries.py          # Tạo summary có đầy đủ ngày và giờ
    ├── get_summary_chb.py                   # Hàm đọc summary đã thêm ngày
    ├── rules.py                             # Quy tắc thời gian và gán nhãn
    ├── select_patients.py                   # Thống kê, sàng lọc bệnh nhân
    ├── split_chb_manual.py                  # Sao chép EDF thành train/test
    ├── segment_train_chb.py                 # Xuất cửa sổ train có nhãn
    ├── segment_test_chb.py                  # Xuất cửa sổ test liên tục
    ├── kfold_pre_clusters_balanced_inter.py # Tạo CSV train/validation
    └── segment_pretrain_chb.py              # Xuất dữ liệu pretrain, tùy chọn
```

Các thư mục kết quả được tạo bên trong `preprocessing/`, trừ file `*-summary-dated.txt` được tạo cạnh summary gốc trong thư mục dữ liệu.

## 3. Chuẩn bị trước khi chạy

### 3.1. Chuẩn bị dữ liệu gốc

Tải dữ liệu từ [CHB-MIT trên PhysioNet](https://physionet.org/content/chbmit/1.0.0/), giữ nguyên tên file và cấu trúc thư mục bệnh nhân:

```text
D:/DO_AN_TN/chb-mit-scalp-eeg-database-1.0.0/
├── chb01/
│   ├── chb01-summary.txt
│   ├── chb01_01.edf
│   └── ...
├── chb02/
│   ├── chb02-summary.txt
│   └── ...
└── ...
```

Đối với mỗi bệnh nhân xử lý, cần có summary và đủ các EDF được liệt kê trong summary đó. Bước thêm ngày quét tất cả `chb*/chb*-summary.txt` tìm thấy trong thư mục dữ liệu.

### 3.2. Cài thư viện

Mở PowerShell tại thư mục repository:

```powershell
Set-Location 'D:\DO_AN_TN\scripts\datnv2\scripts\Epileptic-Seizure-Prediction-Using-EEG-Signals'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy scipy mne tqdm
```

Nếu sử dụng mô hình, cài thêm:

```powershell
python -m pip install torch timm einops
```

Đây là các thư viện được import trong mã hiện có. Repository chưa có file khóa phiên bản môi trường; bản PyTorch dùng GPU cần phù hợp với môi trường CUDA của máy.

### 3.3. Chỉnh đường dẫn dữ liệu — bắt buộc

Các script hiện tính đường dẫn theo vị trí file bằng `parents[...]`. Với vị trí repository đang dùng, đường dẫn mặc định trỏ đến `D:/DO_AN_TN/scripts/chb-mit-scalp-eeg-database-1.0.0`, trong khi dữ liệu thực tế nằm ở `D:/DO_AN_TN/chb-mit-scalp-eeg-database-1.0.0`.

Trước khi chạy, sửa các biến sau về cùng thư mục dữ liệu thực tế:

| File | Biến cần sửa | Giá trị cho máy hiện tại |
| --- | --- | --- |
| `preprocessing/add_dates_to_chb_summaries.py` | `DATA_DIR` | `Path("D:/DO_AN_TN/chb-mit-scalp-eeg-database-1.0.0")` |
| `preprocessing/select_patients.py` | `DATA_ROOT` | `Path("D:/DO_AN_TN/chb-mit-scalp-eeg-database-1.0.0")` |
| `preprocessing/split_chb_manual.py` | `DATA_ROOT` | `Path("D:/DO_AN_TN/chb-mit-scalp-eeg-database-1.0.0")` |
| `preprocessing/get_summary_chb.py` | `dataset_root` trong khối `__main__` | Chỉ cần sửa nếu chạy riêng file này để xem thử kết quả |

`segment_pretrain_chb.py` lấy `DATA_ROOT` từ `split_chb_manual.py`, nên không cần đặt lại biến này trong script pretrain.

### 3.4. Thống nhất cấu hình trước khi xuất cửa sổ

**Mã hiện tại chưa đồng nhất giữa bước cắt cửa sổ và K-fold.** Cụ thể:

- `segment_train_chb.py` và `segment_test_chb.py` đang đặt `WINDOW_SECONDS = 30`.
- `kfold_pre_clusters_balanced_inter.py` đang kiểm tra và đếm cửa sổ **4 giây** bằng các giá trị viết trực tiếp trong mã.
- Train xuất vào `segment_train_4h_120m`, còn K-fold lại đọc `segment_train_4h_30m`.
- Theo hằng số trong `rules.py`, khoảng preictal hiện dài **120 phút**: từ 122 đến 2 phút trước onset, dù một số chú thích và tên file còn nhắc thời lượng khác.

**Để chạy toàn bộ quy trình, gồm K-fold, bằng logic hiện có**, thống nhất cấu hình như sau trước bước 4.4:

| File | Thiết lập |
| --- | --- |
| `segment_train_chb.py` | Đặt `WINDOW_SECONDS = 4`; giữ `OUTPUT_ROOT = BASE / "segment_train_4h_120m"` |
| `segment_test_chb.py` | Đặt `WINDOW_SECONDS = 4` để thống nhất độ dài cửa sổ train/test/pretrain |
| `kfold_pre_clusters_balanced_inter.py` | Đặt `DATA_ROOT = BASE / "segment_train_4h_120m"` |
| `kfold_pre_clusters_balanced_inter.py` | Đặt `CSV_PATH = BASE / "segment_train_folds" / "kfold_4h_120m.csv"` để tên kết quả phản ánh cấu hình |

Các thay đổi trên là hướng dẫn cấu hình; việc thêm README không tự thay đổi các script. Nếu tiếp tục dùng cửa sổ 30 giây, cần sửa cả logic đếm và kiểm tra cửa sổ trong K-fold trước khi chạy bước đó; chỉ đổi đường dẫn là chưa đủ.

### 3.5. Môi trường huấn luyện trên Kaggle

Đồ án sử dụng **Kaggle Notebook** làm môi trường huấn luyện DBConformer. Các lệnh PowerShell và đường dẫn `D:/DO_AN_TN/...` trong README phục vụ bước chuẩn bị dữ liệu trên máy cá nhân.

**Notebook huấn luyện:** [DBConformer with EEG Dataset — nguyentduc](https://www.kaggle.com/code/nguyentduc/dbconformer-with-eeg-dataset).

**Các bộ dữ liệu đã tiền xử lý trên Kaggle:**

| STT | Bộ dữ liệu | Liên kết |
| --- | --- | --- |
| 1 | CHB-MIT Preprocessing Dataset — Window Size 30 | [chb-mit-preprocessing-dataset-window-size30](https://www.kaggle.com/datasets/nguyentduc/chb-mit-preprocessing-dataset-window-size30) |
| 2 | EEG CHB-MIT Preprocessing — 4 Hours | [eeg-chb-mit-preprocessig-4-hours](https://www.kaggle.com/datasets/nguyentduc/eeg-chb-mit-preprocessig-4-hours/) |

Để sử dụng dữ liệu đã chuẩn bị, mở notebook huấn luyện và gắn bộ dữ liệu tương ứng với thí nghiệm. Kiểm tra cấu hình trong notebook và metadata của dữ liệu để thống nhất độ dài cửa sổ, quy tắc gán nhãn và CSV fold; các bộ dữ liệu trên Kaggle có thể thuộc những cấu hình thí nghiệm khác nhau. Nếu muốn tự tạo lại dữ liệu từ EDF gốc, thực hiện quy trình ở mục 4.

Khi tự chuẩn bị dữ liệu để huấn luyện trên Kaggle, các thành phần cần chuyển sang gồm:

- Các file NPZ train đã gán nhãn và CSV phân chia train/validation tương ứng.
- Các file NPZ test liên tục để đánh giá sau khi huấn luyện.
- Mã kiến trúc `models/DBConformer.py`.
- Dữ liệu NPZ pretrain nếu thí nghiệm có bước huấn luyện trước.

Trong notebook huấn luyện, cần cấu hình lại đường dẫn dữ liệu theo vị trí gắn dữ liệu trên Kaggle và giữ đúng cách chia train/validation/test đã chuẩn bị. Ghi lại loại GPU, phiên bản Python/PyTorch, cấu hình huấn luyện và seed của từng lần chạy để có thể tái lập thí nghiệm. Notebook được cung cấp qua liên kết Kaggle ở trên; repository hiện chưa chứa bản sao `.ipynb` của notebook này.

## 4. Thứ tự chạy và kết quả của từng bước

```text
EDF gốc + chbXX-summary.txt
          |
          v
1. add_dates_to_chb_summaries.py
          | tạo chbXX-summary-dated.txt
          v
2. select_patients.py              (thống kê, hỗ trợ lựa chọn)
          | xem CSV và quyết định danh sách/mốc chia
          v
3. split_chb_manual.py
          | tạo EDF train/test, manifest và bản sao dated summary
          +-------------------------------+
          v                               v
4. segment_train_chb.py          5. segment_test_chb.py
          | NPZ có nhãn                    | NPZ liên tục + mốc cơn
          v                               |
6. kfold_pre_clusters_balanced_inter.py    | phục vụ đánh giá sau này
          | CSV train/validation          |
          v                               v
      Huấn luyện trên Kaggle: xem notebook tại mục 3.5
```

Nhánh tùy chọn: sau khi tạo dated summary và chốt `TEST_START`, có thể chạy `segment_pretrain_chb.py` để xuất dữ liệu từ các bệnh nhân ngoài tập đích. Nhánh này đọc EDF gốc, không cần chờ sao chép train/test.

### 4.1. Chạy đầu tiên: thêm ngày tháng năm vào summary

```powershell
python preprocessing/add_dates_to_chb_summaries.py
```

**Script đọc những gì?**

1. Đọc `chbXX-summary.txt` gốc để lấy tên EDF, số cơn và thời điểm bắt đầu/kết thúc cơn tính bằng số giây kể từ đầu EDF.
2. Đọc header của từng EDF tương ứng để lấy ngày, giờ bắt đầu và tính thời lượng bản ghi.
3. Chuyển các mốc thành định dạng đầy đủ `YYYY-MM-DD HH:MM:SS`.

**Script tạo file gì?**

```text
chb01/chb01-summary.txt + chb01/*.edf
                     |
                     v
chb01/chb01-summary-dated.txt
```

Tên tổng quát của file mới là **`chbXX-summary-dated.txt`**, nằm cùng thư mục với summary gốc. Script không sửa tín hiệu EDF và không ghi đè `chbXX-summary.txt`. Nếu chạy lại, file dated cùng tên sẽ được ghi đè.

Ví dụ minh họa cách chuyển đổi, không phải trích đoạn dữ liệu thực:

```text
# Thông tin trong summary gốc
File Name: chbXX_01.edf
File Start Time: 23:30:00
File End Time: 00:30:00
Number of Seizures in File: 1
Seizure Start Time: 2400 seconds
Seizure End Time: 2460 seconds
```

Giả sử header EDF ghi thời điểm bắt đầu là `2000-01-01 23:30:00` và thời lượng 3.600 giây, khối tương ứng trong file mới là:

```text
File Name: chbXX_01.edf
File Start Date and Time: 2000-01-01 23:30:00
File End Date and Time: 2000-01-02 00:30:00
Number of Seizures in File: 1
Seizure Start Date and Time: 2000-01-02 00:10:00
Seizure End Date and Time: 2000-01-02 00:11:00
```

Công thức: **thời điểm cơn = thời điểm bắt đầu EDF + số giây ghi trong summary gốc**. Ngày kết thúc EDF được tính từ thời lượng trong header, nên xử lý được bản ghi đi qua nửa đêm.

File dated là summary rút gọn về thời gian và cơn; không sao chép phần mô tả montage/kênh của summary gốc. Các ngày này vẫn là ngày giả đã ẩn danh của CHB-MIT, không phải ngày khám thực tế.

`get_summary_chb.py` là hàm đọc file dated để các bước sau sử dụng. **Không bắt buộc chạy riêng file này**; nó không tạo thêm dữ liệu.

### 4.2. Thống kê và sàng lọc bệnh nhân

```powershell
python preprocessing/select_patients.py
```

- **Đầu vào:** dated summary và danh sách EDF của từng bệnh nhân.
- **Đầu ra:** `preprocessing/patient_selection/patient_selection.csv` và `selected_patients.csv`.
- **Mục đích:** thống kê số cơn, cụm cơn, thời lượng preictal/interictal và khả năng chia dữ liệu theo thời gian.

Điều kiện chọn hiện có gồm đủ EDF khớp summary, ít nhất 3 cụm có preictal hợp lệ, tồn tại đoạn test có ít nhất 3 giờ tín hiệu với độ phủ ít nhất 99%, test có ít nhất 1 cụm hợp lệ và phần trước test còn ít nhất 2 cụm cùng dữ liệu interictal. Xem `is_selected()` để biết đầy đủ điều kiện.

**CSV này không tự quyết định danh sách chia train/test.** Bước tiếp theo sử dụng `TEST_START` được khai báo thủ công. Thống kê `balanced_clusters` của bước sàng lọc dùng lưới 4 giây và cách gán interictal riêng; không phải kết quả K-fold cuối cùng.

### 4.3. Chia EDF thành train/test theo thời gian

Kiểm tra `DATA_ROOT`, `OUTPUT_ROOT`, `TEST_START` và `DRY_RUN` trong `split_chb_manual.py`, sau đó chạy:

```powershell
python preprocessing/split_chb_manual.py
```

Các mốc đang khai báo:

| Bệnh nhân | EDF đầu tiên thuộc test |
| --- | --- |
| `chb01` | `chb01_25.edf` |
| `chb05` | `chb05_21.edf` |
| `chb06` | `chb06_14.edf` |
| `chb07` | `chb07_15.edf` |
| `chb09` | `chb09_09.edf` |

Script sắp xếp EDF theo thời điểm bắt đầu: các file trước mốc thuộc train, file tại mốc và toàn bộ file sau mốc thuộc test. Đây là cách chia thủ công; script không tự áp lại điều kiện test liên tục của bước sàng lọc.

Kết quả cho mỗi bệnh nhân:

```text
preprocessing/edf_split_manual/chb01/
├── train/                      # Bản sao các EDF trước mốc chia
├── test/                       # Bản sao các EDF từ mốc chia trở đi
├── chb01-summary-dated.txt      # Bản sao summary đầy đủ
└── manifest.csv                # Mỗi EDF thuộc train hay test
```

Script dùng thao tác sao chép, giữ nguyên dữ liệu gốc và cần thêm dung lượng lưu trữ. Đặt `DRY_RUN = True` nếu chỉ muốn xem kế hoạch; đặt `False` để tạo dữ liệu. Script dừng nếu `OUTPUT_ROOT` đã tồn tại khi chạy thật. Bước này chưa chia validation.

### 4.4. Xuất cửa sổ train có nhãn

Hoàn tất cấu hình ở mục 3.4 rồi chạy:

```powershell
python preprocessing/segment_train_chb.py
```

- **Đầu vào:** EDF trong `edf_split_manual/chbXX/train/`, `manifest.csv` và dated summary đầy đủ.
- **Đầu ra:** `preprocessing/segment_train_4h_120m/chbXX/pre/*.npz`, `inter/*.npz` và `counts.json`.
- `pre/`: nhãn **1**, preictal.
- `inter/`: nhãn **0**, interictal.

Script xét quy tắc thời gian trên summary đầy đủ rồi chỉ đọc tín hiệu EDF thuộc train. Chỉ giữ cửa sổ nằm trọn trong đoạn preictal/interictal hợp lệ; cửa sổ không nối qua biên EDF hoặc khoảng mất bản ghi.

### 4.5. Xuất cửa sổ test liên tục

```powershell
python preprocessing/segment_test_chb.py
```

- **Đầu vào:** EDF trong `edf_split_manual/chbXX/test/`, manifest và dated summary.
- **Đầu ra:** `preprocessing/test_continuous/chbXX/*.npz`, mỗi file tương ứng một EDF được xuất.
- Giữ tín hiệu liên tục theo các cửa sổ, có cả các cơn về sau; không chọn riêng cửa sổ preictal/interictal và không tạo mảng `labels` như train.
- Lưu `seizure_intervals` và mốc thời gian để phục vụ đánh giá cảnh báo theo sự kiện sau này.

Đầu tập test có thể bị cắt để loại phần còn nằm trong 4 giờ sau cơn trước mốc chia, theo `EXCLUSION` hiện tại. EDF bị cắt hết sẽ không được xuất.

`PATIENTS` hiện chỉ gồm `chb01`, `chb05`, `chb06`, `chb09`; `chb07` có trong bước chia EDF nhưng chưa nằm trong danh sách test này. Đặt lại `PATIENTS` nếu muốn xử lý thêm bệnh nhân, hoặc dùng `None` để lấy tất cả thư mục bệnh nhân đã chia.

### 4.6. Tạo danh sách train/validation bằng K-fold

Chỉ chạy sau khi xuất train và đồng nhất cấu hình theo mục 3.4:

```powershell
python preprocessing/kfold_pre_clusters_balanced_inter.py
```

- **Đầu vào:** NPZ train, manifest và dated summary.
- **Đầu ra theo cấu hình đề xuất:** `preprocessing/segment_train_folds/kfold_4h_120m.csv`.
- Mỗi dòng mô tả bệnh nhân, fold, subset `train`/`val`, đường dẫn NPZ tương đối với `DATA_ROOT`, nhãn, số cửa sổ, EDF nguồn và mã cụm liên quan.

Preictal được giữ theo cụm cơn; các cụm dùng chung EDF được liên kết thành nhóm. Preictal/interictal trong cùng EDF luôn đi cùng subset để tránh chia một EDF vào cả train và validation của cùng fold. Các EDF interictal còn lại được phân bổ để giảm chênh lệch số cửa sổ interictal giữa các fold.

**Script không lấy mẫu để cân bằng chính xác hai lớp 1:1**, không sửa NPZ và không huấn luyện mô hình. `K = None` chọn số fold tối đa khả thi theo dữ liệu; có thể đặt số nguyên từ 2 trở lên làm giới hạn. Bệnh nhân không đủ ít nhất 2 nhóm/fold phù hợp sẽ bị bỏ qua. `PATIENTS` hiện giống danh sách test ở bước 4.5.

### 4.7. Tùy chọn: xuất dữ liệu pretrain

```powershell
python preprocessing/segment_pretrain_chb.py
```

- **Đầu vào:** EDF gốc và dated summary của các bệnh nhân pretrain.
- **Đầu ra:** `preprocessing/segment_pretrain/chbXX/{pre,inter}/*.npz`, `counts.json` và `segment_pretrain/patients.json`.
- Lấy độ dài cửa sổ từ `segment_test_chb.py` và dùng chung hàm lọc/xuất với train/test.
- Loại các bệnh nhân đích trong `TEST_START`; đồng thời loại `chb21` khi `chb01` là bệnh nhân đích vì đây là cùng một người, theo [mô tả CHB-MIT](https://physionet.org/content/chbmit/1.0.0/).

`PATIENTS` đang là danh sách cố định trong script. Đặt `PATIENTS = None` để tự chọn các thư mục ngoài nhóm bị loại. Với EDF có trên đĩa nhưng không được liệt kê trong summary, bước này không gán nhãn và ghi nhận tên file bị loại trong báo cáo. Đây là bước chuẩn bị dữ liệu pretrain, chưa thực hiện huấn luyện trước.

## 5. Quy tắc gán nhãn hiện tại

Các giá trị thực thi nằm trong [`preprocessing/rules.py`](preprocessing/rules.py):

| Quy tắc | Giá trị hiện tại |
| --- | --- |
| Gộp cụm cơn | Gộp khi cơn tiếp theo bắt đầu không muộn hơn 1 giờ sau kết thúc cụm trước |
| Preictal | `[onset đầu cụm − 122 phút, onset đầu cụm − 2 phút)`, dự kiến 120 phút |
| Postictal | 1 giờ sau kết thúc cơn cuối cụm |
| Loại khỏi interictal quanh cụm | Từ 4 giờ trước onset đầu cụm đến 4 giờ sau offset cuối cụm |
| Gap dài, tách phiên | Khoảng thiếu bản ghi lớn hơn 300 giây |
| Loại khỏi interictal quanh gap dài | Toàn bộ gap và 1 giờ đệm mỗi phía |

Preictal còn phải qua kiểm tra dữ liệu thiếu, giao với vùng sự kiện khác và vùng không chắc chắn sau gap dài. Không phải mọi cụm đều được dùng. `MAX_MISSING_PRE_SECONDS = 300` trong K-fold là điều kiện bổ sung về tổng thời gian thiếu preictal khi chọn nhóm validation.

`rules.py` có thể tạo cả nhãn ictal và postictal để mô tả timeline, nhưng script xuất train/pretrain chỉ lưu hai lớp preictal và interictal. Không mặc định mọi đoạn chưa có nhãn là interictal.

## 6. Tín hiệu và định dạng NPZ

Hàm `export_edf()` trong `segment_test_chb.py` được cả train, test và pretrain sử dụng:

- Chọn hoặc dựng **18 kênh bipolar**, yêu cầu tần số lấy mẫu **256 Hz**.
- Lọc Butterworth bandstop nhân quả ở **57–63 Hz** và **117–123 Hz**.
- Giữ trạng thái lọc giữa hai EDF nối tiếp không có gap; reset khi có gap. Ngưỡng reset lọc khác với ngưỡng 300 giây dùng để tách phiên.
- Trừ trung bình từng kênh trong từng cửa sổ; chưa z-score và chưa tăng cường dữ liệu.
- Cửa sổ không chồng lấn, bỏ phần cuối EDF không đủ một cửa sổ; tín hiệu lưu bằng đơn vị volt.

| Trường | Ý nghĩa |
| --- | --- |
| `eeg` | Mảng `float32` có dạng `(N, 18, T)`; `T = window_seconds × 256` |
| `labels` | Chỉ có trong train/pretrain: 0 là interictal, 1 là preictal |
| `start_sample`, `available_sample` | Vị trí bắt đầu và kết thúc cửa sổ tính từ đầu EDF |
| `sfreq`, `channels`, `source_edf` | Tần số lấy mẫu, thứ tự kênh, tên EDF nguồn |
| `file_start_datetime`, `file_end_datetime`, `session_id` | Thời gian bản ghi và phiên ghi |
| `seizure_intervals` | Mốc bắt đầu/kết thúc cơn theo chỉ số mẫu tính từ đầu EDF |
| `preprocessing_json` | Tham số và mô tả tiền xử lý |

Ví dụ đọc một file NPZ sau khi đã xuất dữ liệu:

```python
import numpy as np

path = "preprocessing/segment_train_4h_120m/chb01/pre/ten_edf_da_xuat.npz"
with np.load(path, allow_pickle=False) as data:
    print(data.files)
    print(data["eeg"].shape)
    print(np.unique(data["labels"], return_counts=True))
```

Thay `ten_edf_da_xuat.npz` bằng file thực tế. Với cấu hình cửa sổ 4 giây, `T = 1024`; với 30 giây, `T = 7680`.

Mô hình DBConformer nhận tensor dạng `(B, 1, C, T)` và trả về `(đặc_trưng, logits)`. Khi bổ sung phần huấn luyện, cần cấu hình số kênh, chiều thời gian, patch size và số lớp phù hợp dữ liệu; việc chạy riêng `models/DBConformer.py` chưa thực hiện huấn luyện.

## 7. Lưu ý khi chạy lại

- Các script dùng hằng số ở đầu file để cấu hình, chưa có giao diện tham số dòng lệnh.
- Khi đổi quy tắc nhãn hoặc độ dài cửa sổ, xuất lại dữ liệu vào thư mục mới và tạo lại CSV fold tương ứng.
- Nhiều bước dừng nếu thư mục/file đầu ra đã tồn tại. Đổi `OUTPUT_ROOT` và cập nhật đường dẫn bước sau để giữ được kết quả cũ.
- `rules.py` và `get_summary_chb.py` là module hỗ trợ; không cần chạy chúng như một bước độc lập.
- `load_kfold_csv.py` được nhắc trong chú thích script K-fold nhưng hiện chưa có trong repository này; cần bổ sung bộ đọc CSV khi xây dựng phần huấn luyện.

## 8. Ghi nhận nguồn và trích dẫn

**Dữ liệu:** Guttag, J. (2010). *CHB-MIT Scalp EEG Database (version 1.0.0).* PhysioNet. DOI: [10.13026/C2K01R](https://doi.org/10.13026/C2K01R). Xem thêm hướng dẫn trích dẫn trên [trang dữ liệu](https://physionet.org/content/chbmit/1.0.0/).

**Mô hình:** Ziwei Wang, Hongbin Wang, Tianwang Jia, Xingyi He, Siyang Li và Dongrui Wu. *DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding.* IEEE Journal of Biomedical and Health Informatics, 2026. Nguồn mã và thông tin bài báo: [wzwvv/DBConformer](https://github.com/wzwvv/DBConformer).

Giấy phép của dữ liệu và mã mô hình cần được xem riêng tại nguồn tương ứng. Repository DBConformer gốc hiện công bố **PolyForm Noncommercial License 1.0.0** cho sử dụng học thuật và nghiên cứu phi thương mại; xem [mục License của dự án gốc](https://github.com/wzwvv/DBConformer#license) trước khi phân phối lại mã kế thừa hoặc lựa chọn giấy phép cho đồ án.
