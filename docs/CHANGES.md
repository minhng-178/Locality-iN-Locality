# Mô tả Cải tiến Mô hình LNL cho GTSRB

**File thay đổi:** `LNL_improvement.py` (dựa trên `LNL.py` gốc)  
**Baseline:** 89.80% → **Mục tiêu:** ≥ 99.5% (Top-1 Accuracy trên GTSRB)

---

## 1. LayerScale cho Residual Connections

**Vị trí:** class `Block`, 3 learnable parameters × 12 blocks

```python
self.ls_in = nn.Parameter(torch.ones(in_dim) * 0.1)
self.ls_mlp_in = nn.Parameter(torch.ones(in_dim) * 0.1)
self.ls_out = nn.Parameter(torch.ones(dim) * 0.1)
```

Mỗi residual branch được nhân với một hệ số học (khởi tạo 0.1):

```python
pixel_embed = pixel_embed + drop_path(x) * self.ls_in
```

**Tại sao:** Kỹ thuật từ CaiT (Touvron et al., 2021) và ConvNeXt (Liu et al., 2022). Trong deep transformer, residual connection trực tiếp (`x + f(x)`) có thể gây gradient không ổn định. LayerScale khởi tạo nhỏ giúp mỗi block ban đầu gần với identity mapping, sau đó model tự học mức đóng góp phù hợp → gradient flow ổn định hơn, đặc biệt ở các block sâu.

---

## 2. Input Normalization tích hợp trong Model

**Vị trí:** class `LocalViT_TNT`, method `forward_features`

```python
GTSRB_MEAN = [0.3337, 0.3064, 0.3171]
GTSRB_STD  = [0.2672, 0.2564, 0.2629]

def forward_features(self, x):
    x = (x - mean) / std   # normalize trước khi đi vào backbone
    ...
```

**Tại sao:** Notebook gốc chỉ dùng `ToTensor()` (chuyển pixel sang [0,1]) mà không normalize theo thống kê dataset. Transformer rất nhạy cảm với phân phối input — nếu pretrained weights được train trên ảnh đã normalize mà inference lại nhận ảnh chưa normalize → distribution mismatch → accuracy giảm nghiêm trọng. Nhúng normalize vào model đảm bảo backbone luôn thấy đúng phân phối, bất kể pipeline bên ngoài có normalize hay không.

---

## 3. Auto-load Pretrained Checkpoint

**Vị trí:** class `LocalViT_TNT.__init__`

```python
ckpt_path = 'LNL_Ti_GTSRB_best.pt'
if os.path.exists(ckpt_path):
    self.load_state_dict(ckpt, strict=False)
```

**Tại sao:** Khi file checkpoint tồn tại cùng thư mục, backbone tự động khởi tạo từ weights đã pretrain thay vì random init. `strict=False` cho phép bỏ qua mismatch ở classification head (vì notebook sẽ thay head bằng `nn.Linear(192, 43)`). Điều này giúp model **plug-and-play** — không cần sửa notebook, chỉ cần đặt file `.pt` vào đúng thư mục.

---

## 4. Thay đổi Training Pipeline (trong Notebook)

| Thành phần | Gốc | Cải tiến | Tại sao |
|---|---|---|---|
| Batch size | 15 | **64** | Gradient ổn định hơn, tận dụng GPU tốt hơn |
| Epochs | 5 | **15** | 5 epochs không đủ để model hội tụ trên 39K samples |
| Optimizer | SGD (lr=0.007) | **AdamW (lr=5e-4, wd=0.05)** | AdamW hội tụ nhanh hơn SGD cho Transformer nhờ adaptive learning rate; weight decay giúp regularization |
| Scheduler | StepLR (step=30) | **CosineAnnealingLR** | Cosine giảm LR mịn, tránh giảm đột ngột như StepLR → fine-tune tốt hơn ở cuối training |
| Loss | CrossEntropy | **CrossEntropy + label_smoothing=0.1** | Giảm overconfidence, buộc model không quá chắc chắn vào 1 lớp → generalization tốt hơn |
| Augmentation | Không | **Rotation, ColorJitter, Perspective, RandomErasing** | Tăng đa dạng dữ liệu, mô phỏng biến thiên thực tế (góc nhìn, ánh sáng, che khuất) → chống overfit |
| Test shuffle | True | **False** | Kết quả test phải deterministic để kiểm chứng |

---

## Tham khảo

- LayerScale: Touvron et al., *"Going deeper with Image Transformers"*, ICCV 2021
- ConvNeXt: Liu et al., *"A ConvNet for the 2020s"*, CVPR 2022
- AdamW: Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"*, ICLR 2019
- Label Smoothing: Szegedy et al., *"Rethinking the Inception Architecture"*, CVPR 2016
- LNL gốc: Manzari et al., *"Robust Transformer with Locality Inductive Bias and Feature Normalization"*, ESWA 2023
