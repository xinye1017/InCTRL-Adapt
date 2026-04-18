import torch

from train_local import TEST_DATASETS_BY_TRAIN, compute_training_loss


def test_cross_domain_eval_mapping_matches_experiment_design():
    assert TEST_DATASETS_BY_TRAIN["mvtec"] == ["aitex", "elpv", "visa"]
    assert TEST_DATASETS_BY_TRAIN["visa"] == ["mvtec"]


def test_visual_training_loss_uses_logit_branches():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7]),
        "base_logit": torch.tensor([0.3, 0.6]),
        "image_logit": torch.tensor([0.1, 0.8]),
        "pqa_logit": torch.tensor([0.4, 0.9]),
        "static_text_logit": torch.tensor([0.25, 0.75]),
        "adaptive_text_logit": torch.tensor([0.5, 0.5]),
        "text_logit": torch.tensor([0.5, 0.5]),
        "text_static_reg": torch.tensor(0.25),
    }

    total_loss, parts = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        phase="visual",
        image_loss_weight=1.0,
        pqa_loss_weight=0.5,
        text_reg_weight=0.1,
    )

    expected_final = loss_fn(outputs["final_logit"], labels)
    expected_base = loss_fn(outputs["base_logit"], labels)
    expected_image = loss_fn(outputs["image_logit"], labels)
    expected_pqa = loss_fn(outputs["pqa_logit"], labels)
    expected_static_text = loss_fn(outputs["static_text_logit"], labels)

    assert torch.allclose(
        total_loss,
        expected_final + expected_base + expected_image + 0.5 * expected_pqa + expected_static_text,
    )
    assert torch.allclose(parts["final_loss"], expected_final.detach())
    assert torch.allclose(parts["base_loss"], expected_base.detach())
    assert torch.allclose(parts["image_loss"], expected_image.detach())
    assert torch.allclose(parts["pqa_loss"], expected_pqa.detach())
    assert torch.allclose(parts["static_text_loss"], expected_static_text.detach())
    assert torch.allclose(parts["adaptive_text_loss"], torch.tensor(0.0))
    assert torch.allclose(parts["text_loss"], expected_static_text.detach())


def test_visual_training_loss_uses_pqa_mask_supervision_when_masks_are_available():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7]),
        "base_logit": torch.tensor([0.3, 0.6]),
        "image_logit": torch.tensor([0.1, 0.8]),
        "pqa_logit": torch.tensor([0.4, 0.9]),
        "static_text_logit": torch.tensor([0.25, 0.75]),
        "adaptive_text_logit": torch.tensor([0.5, 0.5]),
        "text_logit": torch.tensor([0.5, 0.5]),
        "text_static_reg": torch.tensor(0.25),
        "pqa_local_logits": [torch.randn(2, 2, 8, 8)],
    }
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 2:6, 2:6] = 1.0

    total_loss, parts = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        phase="visual",
        masks=masks,
        image_loss_weight=1.0,
        pqa_loss_weight=0.5,
        mask_loss_weight=2.0,
        text_reg_weight=0.1,
    )

    expected_without_mask = (
        loss_fn(outputs["final_logit"], labels)
        + loss_fn(outputs["base_logit"], labels)
        + loss_fn(outputs["image_logit"], labels)
        + 0.5 * loss_fn(outputs["pqa_logit"], labels)
        + loss_fn(outputs["static_text_logit"], labels)
    )

    assert parts["pqa_mask_loss"] > 0
    assert torch.allclose(total_loss, expected_without_mask + 2.0 * parts["pqa_mask_loss"])


def test_text_training_loss_uses_text_logit_and_static_regularizer():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7]),
        "base_logit": torch.tensor([0.3, 0.6]),
        "image_logit": torch.tensor([0.1, 0.8]),
        "pqa_logit": torch.tensor([0.4, 0.9]),
        "static_text_logit": torch.tensor([0.25, 0.75]),
        "adaptive_text_logit": torch.tensor([0.5, 0.5]),
        "text_logit": torch.tensor([0.5, 0.5]),
        "text_static_reg": torch.tensor(0.25),
    }

    total_loss, parts = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        phase="text",
        image_loss_weight=1.0,
        pqa_loss_weight=0.5,
        text_reg_weight=0.1,
    )

    expected_text = loss_fn(outputs["adaptive_text_logit"], labels)
    expected_reg = 0.1 * outputs["text_static_reg"]

    assert torch.allclose(total_loss, expected_text + expected_reg)
    assert torch.allclose(parts["adaptive_text_loss"], expected_text.detach())
    assert torch.allclose(parts["static_text_loss"], torch.tensor(0.0))
    assert torch.allclose(parts["text_loss"], expected_text.detach())
    assert torch.allclose(parts["text_reg_loss"], expected_reg.detach())
    assert torch.allclose(parts["final_loss"], torch.tensor(0.0))
