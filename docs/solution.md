Solution đề xuất: GTSRB 99.5% Top-1 với LNL trên Colab Free GPU
Sau khi phân tích kỹ code và bài toán, đây là chiến lược tôi đề xuất:
Vấn đề chính của notebook gốc
1. Không có pretrained weights → train từ scratch 6M params trên ~39k ảnh → dễ overfit, hội tụ chậm
2. Augmentation yếu → chỉ có Affine, GaussianBlur, Perspective. Thiếu color jitter, CLAHE, normalize đúng
3. Optimizer kém → SGD lr=0.007 chỉ 10 epochs, không warmup, không cosine schedule
4. Không normalizeảnh → ToTensor() cho range 0,1 nhưng model expect mean/std = 0.5/0.5
5. Head thay thô → thay head rồi train toàn bộ model không có strategy
Chiến lược đề xuất (xếp theo mức ảnh hưởng)
1. Transfer Learning từ TNT-S pretrained (quan trọng nhất)
TNT-S có pretrained weights trên ImageNet (tnt_s_patch16_224.pth.tar). Dù LNL thay MLP bằng LocalityFeedForward, phần lớn weights vẫn compatible:
- pixel_embed, cls_token, patch_pos, pixel_pos → dùng được
- atn_in, attn_out, norm_*, proj → dùng được  
- mlp (outer MLP) → không dùng được cho LNL (khác shape)
Cách làm: Load TNT-S pretrained → copy weights sang LNL_S với strict=False → các layer LocalityFeedForward init random, còn lại đã pretrained. Điều này cho convergence nhanh hơn rất nhiều.
# Ý tưởng:
tnt_s = tnt_s_patch16_224(pretrained=True)  # tải weight ImageNet
lnl_s = LNL_S(pretrained=False, num_classes=43)

# Copy matching weights
tnt_state = tnt_s.state_dict()
lnl_state = lnl_s.state_dict()
pretrained_dict = {k: v for k, v in tnt_state.items() 
                   if k in lnl_state and v.shape == lnl_state[k].shape}
lnl_state.update(pretrained_dict)
lnl_s.load_state_dict(lnl_state)
2. Data Augmentation mạnh hơn + Normalize đúng
# GTSRB-specific transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomRotation(15),           # nhẹ, vì biển báo có hướng cố định
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(0.3, p=0.5),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.337, 0.3064171],   # GTSRB actual stats
                         std=[0.2672, 0.2564, 0.2629]),
    transforms.RandomErasing(p=0.25),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.337, 0.3064, 0.3171],
                std=[0.2672, 0.2564, 0.2629]),
])
Lưu ý: KHÔNG dùng RandomHorizontalFlip — lật ngang biển báo sẽ tạo class sai (rẽ trái thành rẽ phải).
3. Training Schedule tốt hơn
# AdamW thay SGD — hội tụ nhanh hơn nhiều cho Transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)

# Cosine anealing with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

num_epochs = 50  # nhiều hơn 10 nhưng vẫn OK với Colab free (~2-3h cho LNL_Ti)
4. Discriminative Learning Rate
Backbone pretrained dùng LR thấp, head mới dùng LR cao:
param_groups = [
    {'params': model.head.parameters(), 'lr': 5e-3},           # head mới
    {'params': [p for n, p in model.named_parameters() 
                if 'head' not in n], 'lr': 5e-4},              # backbone
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
5. Label Smoothing + Mixup
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixup đơn giản (không cần thêm lib)
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
6. Model Architecture Tweaks (nhỏ nhưng có ích)
- Dùng LNL_Ti (6M params) thay vì LNL_S (24M) →ít overfit hơn trên dataset nhỏ
- Tăng drop_path_rate lên 0.1–0.15 để regularize
- Fix SE reduction: hiện tại reduction=dim → squeeze 1 neuron, nên đổi thành reduction=4 để SE module hoạt động tốt hơn
7. Class-balanced Sampling (GTSRB bị imbalanced nặng)
from torch.utils.data import WeightedRandomSampler

class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in sorted(os.listdir(train_dir))]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[trainset.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(trainset, batch_size=64, sampler=sampler)
Thứ tự ưu tiên (cho Colab Free GPU T4, ~12GB VRAM, ~4h max)
| # | Cải tiến | Dự kiến tăng | Effort |
|---|----------|--------|
| 1 | Transfer learning TNT-S → LNL | +3-5% | Medium |
| 2 | Augmentation + Normalize đúng | +2-3% | Low |
| 3 | AdamW + Cosine + Warmup (50 epochs) | +2-3% | Low |
| 4 | Label smoothing + Mixup | +0.5-1% | Low |
| 5 | Class-balanced sampling | +0.5-1% | Low |
| 6 | Discriminative LR | +0.3-0.5% | Low |
| 7 | Drop path 0.1 | +0.2-0.5% | Trivial |
Kết hợp tất cả, từ baseline ~95-96% (train from scratch 10 epochs) → dự kiến đạt 99.0-99.5%.
Tài nguyên Colab Free
- LNL_Ti: ~6M params, batch_size=64, 50 epochs GTSRB → ~1.5-2h trên T4
- LNL_S: ~24M params, batch_size=32→ ~3-4h, cần chặt chẽ hơn
- Khuyến nghị dùng LNL_Ti vì dataset nhỏ,ít risk overfit, train nhanh hơn, dễ iterate
Deliverables cần nộp
1. File model (code .py) — LNL.py có sửa nhẹ (fix bug LNL_S return, thêm drop_path param)
2. Pretrained model (.pth) — checkpoint sau khi train xong
3. Screenshot kết quả — cell output accuracy
4. File giải thích — mô tả các cải tiến
Bạn muốn tôi triển khai cụ thể hướng nào trước? Ví dụ:
- Viết script training hoàn chỉnh (1 file .py chạy trên Colab)?
- Fix bug LNL_S + thêm drop_path parameter?
- Viết code transfer learning TNT-S → LNL?