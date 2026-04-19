import torch

from open_clip.inctrl_pqa_losses import compute_pqa_mask_loss, compute_training_loss
from train_local import TEST_DATASETS_BY_TRAIN


def test_cross_domain_eval_mapping_matches_experiment_design():
    assert TEST_DATASETS_BY_TRAIN["mvtec"] == ["aitex", "elpv", "visa"]
    assert TEST_DATASETS_BY_TRAIN["visa"] == ["mvtec"]


def test_pqa_fused_training_loss_defaults_to_final_and_pqa_terms():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7]),
        "image_logit": torch.tensor([0.1, 0.8]),
        "pqa_logit": torch.tensor([0.4, 0.9]),
    }

    total_loss, parts = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
    )

    expected_final = loss_fn(outputs["final_logit"], labels)
    expected_pqa = loss_fn(outputs["pqa_logit"], labels)

    assert torch.allclose(total_loss, expected_final + expected_pqa)
    assert torch.allclose(parts["final_loss"], expected_final.detach())
    assert torch.allclose(parts["image_loss"], torch.tensor(0.0))
    assert torch.allclose(parts["pqa_loss"], expected_pqa.detach())
    assert torch.allclose(parts["pqa_mask_loss"], torch.tensor(0.0))
    assert torch.allclose(parts["total_loss"], total_loss.detach())


def test_pqa_fused_training_loss_keeps_image_auxiliary_as_ablation_knob():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7]),
        "image_logit": torch.tensor([0.1, 0.8]),
        "pqa_logit": torch.tensor([0.4, 0.9]),
    }

    total_loss, parts = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        image_loss_weight=0.25,
        pqa_loss_weight=0.5,
    )

    expected_final = loss_fn(outputs["final_logit"], labels)
    expected_image = loss_fn(outputs["image_logit"], labels)
    expected_pqa = loss_fn(outputs["pqa_logit"], labels)

    assert torch.allclose(total_loss, expected_final + 0.25 * expected_image + 0.5 * expected_pqa)
    assert torch.allclose(parts["image_loss"], expected_image.detach())


def test_pqa_mask_loss_returns_zero_without_masks():
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7]),
        "pqa_local_logits": torch.randn(2, 2, 8, 8),
    }

    assert torch.allclose(compute_pqa_mask_loss(outputs, masks=None), torch.tensor(0.0))


def test_pqa_only_training_loss_adds_mask_supervision_when_masks_are_available():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7]),
        "image_logit": torch.tensor([0.1, 0.8]),
        "pqa_logit": torch.tensor([0.4, 0.9]),
        "pqa_local_logits": [torch.randn(2, 2, 8, 8)],
    }
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 2:6, 2:6] = 1.0

    total_loss, parts = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        masks=masks,
        image_loss_weight=0.0,
        pqa_loss_weight=0.5,
        mask_loss_weight=2.0,
    )

    expected_without_mask = (
        loss_fn(outputs["final_logit"], labels)
        + 0.5 * loss_fn(outputs["pqa_logit"], labels)
    )

    assert parts["pqa_mask_loss"] > 0
    assert torch.allclose(total_loss, expected_without_mask + 2.0 * parts["pqa_mask_loss"])
