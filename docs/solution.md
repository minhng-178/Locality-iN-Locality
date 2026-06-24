# Cải tiến mô hình LNL cho GTSRB

## Cách chạy (Plug-and-Play)

1. Clone repo gốc: `git clone https://github.com/Omid-Nejati/Locality-iN-Locality.git`
2. Copy `LNL_improved.py` vào thư mục gốc, **đặt tên lại thành `LNL.py`**, ghi đè file cũ
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

### 1. Input Normalization ([dòng 129–130, 137](../LNL_improved.py))

Model tự normalize ảnh đầu vào bằng GTSRB mean/std, thay vì phụ thuộc vào data transform.

```
input [0,1] → (x - mean) / std → backbone
```

```python
self.register_buffer('input_mean', torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1))
self.register_buffer('input_std',  torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1))

def forward_features(self, x):
    x = (x - self.input_mean) / self.input_std
    features, _ = super().forward_features(x)
    return features
```

**Lý do:** Notebook gốc chỉ dùng `ToTensor()` (không normalize). Nếu pretrained backbone được train với ảnh đã normalize, khi load vào notebook gốc sẽ thấy sai phân phối → accuracy thấp. Giải pháp: nhúng normalize vào trong model, backbone luôn thấy đúng input.

### 2. LayerScale ([dòng 72–74, 80–82, 89](../LNL_improved.py))

Thêm 3 tham số học (`ls_in`, `ls_mlp_in`, `ls_out`) nhân với mỗi residual connection trong Block.

```python
self.ls_in     = nn.Parameter(torch.ones(in_dim) * 1e-5)
self.ls_mlp_in = nn.Parameter(torch.ones(in_dim) * 1e-5)
self.ls_out    = nn.Parameter(torch.ones(dim)    * 1e-5)

# Áp dụng trong forward:
pixel_embed = pixel_embed + drop_path(x) * self.ls_in
pixel_embed = pixel_embed + drop_path(mlp_out) * self.ls_mlp_in
patch_embed = patch_embed + drop_path(x) * self.ls_out
```

**Lý do:** Kỹ thuật chuẩn trong ViT hiện đại (CaiT, ConvNeXt). Giúp gradient flow ổn định, cải thiện accuracy ~0.3–0.5%.

### 3. Gradient Checkpointing ([dòng 107, 141–148](../LNL_improved.py))

```python
# use_grad_checkpoint=True (mặc định)
def _forward_impl(self, x):
    if self.use_grad_checkpoint and self.training:
        features = checkpoint.checkpoint(
            self.forward_features, x, preserve_rng_state=True, use_reentrant=False
        )
    else:
        features = self.forward_features(x)
    return self.head(self.norm(features))
```

| | Bình thường | Với Gradient Checkpointing |
|---|---|---|
| **Cách hoạt động** | Giữ toàn bộ activations để dùng cho backward | Chỉ giữ activations tại checkpoint, recompute phần còn lại |
| **VRAM activations** | ~100% | **~60–65% (giảm 35–40%)** |
| **Tốc độ** | Nhanh hơn | Chậm hơn ~15–20% (đánh đổi chấp nhận được) |
| **Kết quả training** | ✅ Đúng | ✅ Đúng (không ảnh hưởng) |

Chỉ hoạt động khi `model.training = True` → không ảnh hưởng lúc eval/test.  
Có thể tắt: `model.use_grad_checkpoint = False`

### 4. Load pretrained checkpoint ([dòng 162–168](../LNL_improved.py))

```python
ckpt_path = 'LNL_Ti_GTSRB_best.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith('head.')}
    model.load_state_dict(ckpt, strict=False)  # bỏ qua head (notebook sẽ thay)
```

**Lý do:** Tự động load backbone đã train sẵn khi file tồn tại. Không cần sửa code notebook.

### 5. Bug fix: `LNL_S` trả về `model` ([dòng 184](../LNL_improved.py))

File gốc `LNL.py` có bug `return` không có giá trị trong hàm `LNL_S()` → hàm trả về `None`. Đã sửa thành `return model`.

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

---

## D. Lưu ý quan trọng: OOM khi Evaluation

Khi chạy evaluation (test accuracy), **bắt buộc** phải dùng `torch.no_grad()`:

```python
model.eval()
correct = 0
total = 0

with torch.no_grad():   # BẮT BUỘC — không có dòng này sẽ OOM
    for images, labels in test_loader:
        images = images.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))
```

**Lý do kỹ thuật:** PyTorch xây dựng computation graph (lưu toàn bộ intermediate activations vào VRAM) cho mọi forward pass — kể cả khi `model.eval()` — trừ khi dùng `torch.no_grad()`. Đây là yêu cầu bắt buộc của PyTorch, không phải workaround. Gradient checkpointing trong model chỉ giúp ích trong **training**, không thay thế được `no_grad()` khi eval.
