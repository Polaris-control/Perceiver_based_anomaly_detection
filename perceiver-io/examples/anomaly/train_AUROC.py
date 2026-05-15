from perceiver.data.vision.mvtec import MVTecDataModule

dm = MVTecDataModule(
    dataset_dir="C:/Users/20763/Desktop/zero-shot/MVtec_ad/data",
    train_categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
    ],
    test_categories = ["tile", "toothbrush", "transistor", "wood", "zipper"],
    include_test_good=True,
    image_size=256,
    batch_size=8,
    num_workers=0,
)
dm.setup()

val_loader = dm.val_dataloader()

pos_ratio = 0.0
for batch_idx, batch in enumerate(val_loader):
    if batch_idx > 20:
        break
    m = batch["mask"]
    pos_ratio = (m > 0.5).float().mean().item()
    print(f"Batch {batch_idx:2d} | pos_ratio = {pos_ratio:.6f}")

print("mask shape:", m.shape)
print("mask unique approx:", (m.min().item(), m.max().item()))