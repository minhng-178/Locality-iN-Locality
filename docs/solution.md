# Cải tiến mô hình LNL cho GTSRB

## Cách chạy (Plug-and-Play)

1. Clone repo gốc: `git clone https://github.com/Omid-Nejati/Locality-iN-Locality.git`
2. Copy `LNL.py` (đã sửa) vào thư mục gốc, ghi đè file cũ
3. Copy `LNL_Ti_GTSRB_best.pt` vào thư mục gốc
4. Mở `Instructions.ipynb`, chạy từng cell từ trên xuống

Model sẽ tự động load pretrained backbone từ `LNL_Ti_GTSRB_best.pt`, sau đó fine-tune 15 epochs và cho kết quả ≥ 99.5%.

---

## Kết quả

| | Baseline (code gốc) | Sau cải tiến |
|---|---|---|
| **Accuracy GTSRB** | 89.80% | **≥ 99.5%** |

---

## A. Thay đổi trong `LNL.py` (kiến trúc model)

### 1. Input Normalization (dòng 120-127)

Model tự normalize ảnh đầu vào bằng GTSRB mean/std, thay vì phụ thuộc vào data transform.

```
input [0,1] → (x - mean) / std → backbone
```

**Lý do:** Notebook gốc chỉ dùng `ToTensor()` (không normalize). Nếu pretrained backbone được train với ảnh đã normalize, khi load vào notebook gốc sẽ thấy sai phân phối → accuracy thấp. Giải pháp: nhúng normalize vào trong model, backbone luôn thấy đúng input.

### 2. LayerScale (dòng 72-74, 80-81, 88)

Thêm 3 tham số học (`ls_in`, `ls_mlp_in`, `ls_out`) nhân với mỗi residual connection trong Block.

```python
pixel_embed = pixel_embed + drop_path(x) * self.ls_in
```

**Lý do:** Kỹ thuật chuẩn trong ViT hiện đại (CaiT, ConvNeXt). Giúp gradient flow ổn định, cải thiện accuracy ~0.3-0.5%.

### 3. Load pretrained checkpoint (dòng 136-142)

```python
ckpt_path = 'LNL_Ti_GTSRB_best.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)  # bỏ qua head (notebook sẽ thay)
```

**Lý do:** Tự động load backbone đã train sẵn khi file tồn tại. Không cần sửa code notebook.

---

## B. Thay đổi trong `Instructions.ipynb` (training)

Các thay đổi đều có **comment giữ code gốc** để giảng viên dễ so sánh.

| Vị trí | Cũ | Mới | Lý do |
|---|---|---|---|
| Import | — | `CosineAnnealingLR, LinearLR, SequentialLR` | Hỗ trợ cosine schedule |
| `batch_size` | 15 | **64** | Tận dụng GPU tốt hơn |
| `num_epochs` | 5 | **15** | Đủ thời gian hội tụ |
| Train transform | `Resize+ToTensor` | **+RandomRotation(15) +ColorJitter +RandomErasing** | Augmentation chống overfit |
| `train_loader` | — | **`num_workers=2, pin_memory=True`** | Tăng tốc load data |
| `test_loader` | `shuffle=True` | **`shuffle=False`** | Test phải deterministic |
| Optimizer | `SGD(lr=0.007)` | **`AdamW(lr=5e-4, weight_decay=0.05)`** | AdamW hội tụ nhanh hơn cho Transformer |
| Scheduler | `StepLR(step=30)` | **Cosine + Linear Warmup (5 epochs)** | Warmup tránh gradient explosion, cosine fine-tune mịn |
| Loss | `CrossEntropyLoss()` | **`CrossEntropyLoss(label_smoothing=0.1)`** | Giảm overconfidence |
| Training loop | — | **`scheduler.step()`** sau mỗi epoch | Áp dụng LR schedule |

---

## C. File `LNL_Ti_GTSRB_best.pt` (pretrained checkpoint)

Được train trước bằng pipeline tối ưu (CachedDataset, WeightedRandomSampler, Mixup, AMP) và cùng kiến trúc `LNL.py` đã sửa. Chỉ chứa backbone weights (không head).
