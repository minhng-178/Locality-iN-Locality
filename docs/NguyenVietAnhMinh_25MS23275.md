# BÁO CÁO CẢI TIẾN MÔ HÌNH NHẬN DẠNG BIỂN BÁO GIAO THÔNG

**Họ và tên:** Nguyễn Viết Anh Minh  
**Mã số sinh viên:** 25MS23275  
**Mô hình gốc:** [Locality-iN-Locality (LNL)](https://github.com/Omid-Nejati/Locality-iN-Locality)  
**Tập dữ liệu:** GTSRB (German Traffic Sign Recognition Benchmark)

---

## Kết quả

| Metric | Baseline | Sau cải tiến |
|---|---|---|
| **Top-1 Accuracy (GTSRB)** | 89.80% | **99.78%** |

---

## Cách chạy (Plug-and-Play)

1. Clone repo gốc: `git clone https://github.com/Omid-Nejati/Locality-iN-Locality.git`
2. Copy `LNL_improved.py` vào thư mục gốc, **đổi tên thành `LNL.py`** (ghi đè file cũ)
3. Copy `pretrained-model.pt` vào thư mục `pytorch/`
4. Mở `Instructions.ipynb`, chạy từng cell từ trên xuống

> `LNL_improved.py` là file kiến trúc model đã tối ưu — phải dùng file này thay cho `LNL.py` gốc.  
> Model sẽ tự động load pretrained backbone → fine-tune 15 epochs → **99.78%**.

---

## A. Thay đổi trong `LNL_improved.py` (kiến trúc model)

### 1. Input Normalization (dòng 129–130, 137)

Nhúng normalize GTSRB mean/std vào model — backbone luôn thấy đúng phân phối đầu vào, bất kể pipeline transform của notebook.

```python
self.register_buffer('input_mean', torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1))
self.register_buffer('input_std',  torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1))

def forward_features(self, x):
    x = (x - self.input_mean) / self.input_std
    features, _ = super().forward_features(x)
    return features
```

### 2. LayerScale (dòng 72–74, 80–82, 89)

3 scalar học được nhân vào residual connections (khởi tạo `1e-5`). Kỹ thuật của CaiT/ConvNeXt — ổn định gradient, tăng accuracy ~0.3–0.5%.

```python
self.ls_in     = nn.Parameter(torch.ones(in_dim) * 1e-5)
self.ls_mlp_in = nn.Parameter(torch.ones(in_dim) * 1e-5)
self.ls_out    = nn.Parameter(torch.ones(dim)    * 1e-5)

# forward:
pixel_embed = pixel_embed + self.drop_path(x) * self.ls_in
pixel_embed = pixel_embed + self.drop_path(mlp_out) * self.ls_mlp_in
patch_embed = patch_embed + self.drop_path(x) * self.ls_out
```

### 3. Gradient Checkpointing (dòng 107, 141–148)

Giảm VRAM activations ~35–40% khi training (batch_size=64 trên T4 16 GB). Không ảnh hưởng accuracy.

```python
def _forward_impl(self, x):
    if self.use_grad_checkpoint and self.training:
        features = checkpoint.checkpoint(
            self.forward_features, x, preserve_rng_state=True, use_reentrant=False
        )
    else:
        features = self.forward_features(x)
    return self.head(self.norm(features))
```

### 4. Auto-load pretrained checkpoint (dòng 162–168)

Model tự load backbone khi file tồn tại, bỏ qua `head` (notebook tạo head mới).

```python
ckpt_path = 'pytorch/pretrained-model.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith('head.')}
    model.load_state_dict(ckpt, strict=False)
```

### 5. Bug fix: `LNL_S` thiếu `return model` (dòng 184)

File gốc trả về `None`. Đã sửa thành `return model`.

---

## B. Thay đổi trong `Instructions.ipynb` (training)

| Tham số | Gốc | Mới |
|---|---|---|
| `batch_size` | 15 | **64** |
| `num_epochs` | 5 | **15** |
| Train transform | `Resize+ToTensor` | `+RandomRotation(15) +ColorJitter +RandomErasing` |
| `test_loader` | `shuffle=True` | `shuffle=False` |
| Optimizer | `SGD(lr=0.007)` | `AdamW(lr=5e-4, wd=0.05)` |
| Scheduler | `StepLR(step=30)` | `Cosine + Linear Warmup (5 ep)` |
| Loss | `CrossEntropyLoss()` | `CrossEntropyLoss(label_smoothing=0.1)` |

---

## C. Pretrained checkpoint (`pytorch/pretrained-model.pt`)

Train trước bằng pipeline mạnh hơn: `CachedDataset` + `WeightedRandomSampler` + `Mixup` + `AMP`. Chỉ chứa backbone weights (không head) — plug-and-play với notebook gốc.

---

## D. Lưu ý: OOM khi Evaluation

Bắt buộc dùng `torch.no_grad()` khi test, kể cả sau `model.eval()`:

```python
model.eval()
with torch.no_grad():   # BẮT BUỘC — không có dòng này sẽ OOM
    for images, labels in test_loader:
        images = images.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))
```
