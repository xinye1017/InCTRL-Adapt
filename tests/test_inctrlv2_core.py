import numpy as np
import torch
from PIL import Image

from datasets.inctrlv2_dataset import InCTRLv2DirectoryDataset
from models.inctrlv2.dasl import DASLBranch
from models.inctrlv2.fusion import fuse_image_score, fuse_pixel_maps
from models.inctrlv2.losses import DiceLoss
from models.inctrlv2.metrics import compute_pixel_pro
from models.inctrlv2.oasl import OASLBranch


def test_dasl_and_oasl_patch_adapters_are_independent():
    dasl = DASLBranch(patch_dim=8, text_dim=6, selected_layers=[7, 9])
    oasl = OASLBranch(patch_dim=8, text_dim=6, selected_layers=[7, 9])

    dasl_ptrs = {param.data_ptr() for param in dasl.parameters()}
    oasl_ptrs = {param.data_ptr() for param in oasl.parameters()}

    assert dasl_ptrs.isdisjoint(oasl_ptrs)


def test_empty_mask_dice_loss_is_prediction_mean():
    pred = torch.tensor([[[[0.2, 0.6], [0.8, 0.4]]]], dtype=torch.float32)
    target = torch.zeros_like(pred)

    loss = DiceLoss()(pred, target)

    assert torch.isclose(loss, pred.mean())


def test_alpha_beta_fusion_formulas():
    s_i = torch.tensor([0.2])
    s_q = torch.tensor([0.6])
    s_p = torch.tensor([0.8])
    image_score = fuse_image_score(s_i=s_i, s_q=s_q, s_p=s_p, alpha=0.5)

    dasl_map = torch.full((1, 1, 2, 2), 0.2)
    oasl_map = torch.full((1, 1, 2, 2), 0.8)
    pixel_map = fuse_pixel_maps(dasl_map=dasl_map, oasl_map=oasl_map, beta=0.75)

    assert torch.isclose(image_score, torch.tensor([0.6])).all()
    assert torch.isclose(pixel_map, torch.full((1, 1, 2, 2), 0.65)).all()


def test_directory_dataset_returns_zero_normal_mask_and_real_abnormal_mask(tmp_path):
    root = tmp_path / "data" / "mvtec" / "bottle"
    (root / "train" / "good").mkdir(parents=True)
    (root / "test" / "good").mkdir(parents=True)
    (root / "test" / "broken").mkdir(parents=True)
    (root / "ground_truth" / "broken").mkdir(parents=True)

    Image.new("RGB", (8, 8), color="white").save(root / "train" / "good" / "prompt.png")
    Image.new("RGB", (8, 8), color="white").save(root / "test" / "good" / "normal.png")
    Image.new("RGB", (8, 8), color="black").save(root / "test" / "broken" / "bad.png")
    mask = Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8).repeat(4, axis=0).repeat(4, axis=1))
    mask.save(root / "ground_truth" / "broken" / "bad_mask.png")

    dataset = InCTRLv2DirectoryDataset(
        dataset_root=tmp_path / "data" / "mvtec",
        split="test",
        shots=1,
        input_size=8,
        seed=0,
    )

    normal = next(sample for sample in dataset if sample["label"].item() == 0)
    abnormal = next(sample for sample in dataset if sample["label"].item() == 1)

    assert normal["query_image"].shape == (3, 8, 8)
    assert normal["prompt_images"].shape == (1, 3, 8, 8)
    assert normal["mask"].shape == (1, 8, 8)
    assert normal["mask"].sum().item() == 0
    assert abnormal["mask"].sum().item() > 0
    assert abnormal["class_name"] == "bottle"


def test_pixel_pro_returns_one_for_perfect_prediction():
    pred = np.zeros((1, 4, 4), dtype=np.float32)
    mask = np.zeros((1, 4, 4), dtype=np.uint8)
    pred[:, 1:3, 1:3] = 1.0
    mask[:, 1:3, 1:3] = 1

    pro = compute_pixel_pro(pred, mask, max_fpr=0.3, num_thresholds=20)

    assert pro == 1.0
