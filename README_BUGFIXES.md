# 🐛 Bug Fixes: Vì sao agent không học được (và đã sửa thế nào)

Tài liệu này tổng hợp toàn bộ thay đổi so với commit `f6266a7`, giải thích vì sao
bản cũ **về mặt toán học không thể học được**, còn bản mới đạt sức mạnh tương đương
repo tham khảo [Tetris-deep-Q-learning-pytorch](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch).

**Kết quả:** bản cũ score luôn = 0 sau 300 epochs. Bản sửa đạt best score **1673**
(57 lines/ván) sau 3000 epochs.

---

## 📄 `code/tetris.py` — Môi trường & Features

### Fix 1 (nặng nhất): Feature `lines` tính theo tổng cộng dồn

| | |
|---|---|
| **Trước** | `lines = self.cleared_lines` — tổng số hàng đã xóa **cộng dồn cả ván** |
| **Sau** | `_get_state_features(lines_cleared=0)` nhận tham số — số hàng xóa bởi **đúng nước đi đang xét** |

**Vì sao chí mạng:** trong `get_next_states()`, board được simulate nhưng
`self.cleared_lines` không đổi → feature `lines` **giống hệt nhau cho mọi action
ứng viên**. Network không bao giờ phân biệt được nước đi xóa hàng với nước đi
thường — đúng feature quan trọng nhất thì bị vô hiệu hóa. Ngoài ra counter tăng
dần theo thời gian làm input non-stationary.

### Fix 2: Chiều cao cột scan sai hướng

| | |
|---|---|
| **Trước** | Scan **từ dưới lên**, `break` ở khối đầu tiên → lấy khối **thấp nhất** |
| **Sau** | Scan từ trên xuống → khối cao nhất = đỉnh cột thật |

**Vì sao chí mạng:** cột nào có khối chạm đáy thì `height` luôn = 1 dù chồng cao
15 tầng. Cả `total_height` lẫn `bumpiness` (tính từ heights) đều thành rác —
agent chỉ còn 2/4 feature có nghĩa, không biết "sợ" chồng cao.

### Fix 3: Xóa nhiều hàng cùng lúc trong `get_next_states` bị lệch index

| | |
|---|---|
| **Trước** | `del temp_board[r]` + `insert(0, ...)` **trong cùng vòng lặp** |
| **Sau** | `del` hết các hàng đầy trước, rồi mới insert đủ số hàng trống |

**Vì sao quan trọng:** insert ở đầu board làm index các hàng còn lại dịch xuống 1
→ khi clear 2-4 hàng cùng lúc thì xóa **nhầm hàng**. Afterstate của chính những
nước có reward lớn nhất bị tính sai. (`_clear_full_lines` dùng trong `step` vốn
đã đúng — chỉ bản simulate bị lỗi.)

### Fix 4: Reward gần như luôn âm / luôn bằng 0

| | |
|---|---|
| **Trước** | `reward = points − 0.5×height − 0.36×holes − 0.2×bumpiness`, game over −10 |
| **Sau** | `reward = 1 + lines² × 10` (**+1 sống sót mỗi khối**), game over −2 |

**Vì sao chí mạng:** với height tổng thường 20-100+, reward cũ **gần như luôn âm
nặng**. Công thức mới (giống reference) cho tín hiệu dày đặc mỗi step: agent học
được "sống lâu = tốt" ngay cả khi chưa biết xóa hàng, và xóa 1 hàng = 11 điểm,
4 hàng = 161 điểm.

---

## 📄 `code/train.py` — Vòng lặp training

### Fix 5: Lọc `if reward >= 0` vứt sạch replay buffer

| | |
|---|---|
| **Trước** | `if reward >= 0: memory.append(...)` |
| **Sau** | Luôn lưu mọi transition |

**Vì sao chí mạng (combo với Fix 4):** reward cũ gần như luôn âm → **hầu như
không có transition nào lọt vào buffer**, đặc biệt toàn bộ transition game-over
(`done=True`) bị vứt. Agent train trên buffer gần rỗng và không bao giờ học được
rằng thua là xấu (value bootstrap mãi không có điểm kết thúc).

### Fix 6: Train quá dày → policy không ổn định

| | |
|---|---|
| **Trước** | Train **mỗi khối đặt** (~40-100 gradient updates/ván), bắt đầu khi buffer mới có 3 samples |
| **Sau** | Train **1 lần/episode** (giống reference), chờ buffer đủ 3000 samples (10%) |

**Triệu chứng đã quan sát được:** loss leo thang 66 → 96 trong giai đoạn exploit,
score dao động dữ dội 0 ↔ 720 không ổn định thêm.

### Fix 7: Target network bị đóng băng

| | |
|---|---|
| **Trước** | Chỉ update khi `epsilon != 0.001` → **đóng băng vĩnh viễn** sau epoch 2000, chu kỳ 100 episodes |
| **Sau** | Update mỗi 10 episodes, vô điều kiện |

### Fix 8 + 9: Logic save model sai

| | |
|---|---|
| **Trước** | Best model chỉ save khi `epsilon == 0.001` (chạy < 2000 epochs → không bao giờ save); final save `target_net` (bản copy cũ) |
| **Sau** | Save best mỗi khi có score cao mới; final save `q_net` |

### Fix 10: Số epochs không khớp epsilon decay

| | |
|---|---|
| **Trước** | Default 100 epochs, hướng dẫn chạy 300 — lúc dừng epsilon vẫn ≈ **0.85** (85% nước đi random) |
| **Sau** | Default **3000 epochs** = 2000 decay + 1000 exploit |

---

## ✅ Đã verify thế nào

- **Cross-check với reference:** 200 board random → feature `(holes, bumpiness,
  height)` khớp `get_state_properties` của repo reference **200/200**.
- **Test xóa 2 hàng:** feature simulate trong `get_next_states` khớp đúng feature
  thực tế sau `step`, reward = 41 (= 1 + 2²×10) đúng công thức.
- **Run thật 3000 epochs:** best score 1673, nhiều ván 20-57 lines (bản cũ: 0).

## 🚀 Chạy lại

```bash
python code/train.py                    # default 3000 epochs
python code/train.py --num_epochs 5000  # muốn đẩy tiếp giai đoạn exploit
```

> ⚠️ Model save trước các fix này (train trên feature sai) không dùng lại được —
> phải train lại từ đầu.

## 📌 Bài học chung khi RL "không cải thiện"

1. **Kiểm tra feature trên afterstate simulate trước** — feature phải phân biệt
   được các action ứng viên với nhau.
2. **Kiểm tra transition terminal có vào buffer không** — không có `done=True`
   thì value không có điểm neo.
3. **Reward phải có tín hiệu dày đặc** — chỉ thưởng sự kiện hiếm (xóa hàng) thì
   agent random không bao giờ nhận được tín hiệu.
4. Hyperparameter (lr, batch size, ...) chỉ tune **sau khi** 3 điều trên đúng.
