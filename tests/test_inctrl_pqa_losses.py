import torch

from open_clip.inctrl_pqa_losses import (
    compute_pqa_local_mil_loss,
    compute_pqa_mask_loss,
    compute_text_prior_loss,
    compute_training_loss,
)


def test_compute_training_loss_uses_simplified_logits_only():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 2:6, 2:6] = 1.0

    outputs = {
        "final_logit": torch.tensor([0.2, 0.7], requires_grad=True),
        "patch_logit": torch.tensor([0.3, 0.6], requires_grad=True),
        "pqa_logit": torch.tensor([0.4, 0.9], requires_grad=True),
        "image_logit": torch.tensor([0.1, 0.8], requires_grad=True),
        "pqa_local_logits": torch.randn(2, 2, 8, 8, requires_grad=True),
        "base_logit": torch.tensor([3.0, 3.0], requires_grad=True),
        "holistic_logit": torch.tensor([4.0, 4.0], requires_grad=True),
    }

    total_loss, metrics = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        masks=masks,
        pqa_loss_weight=0.5,
        image_loss_weight=0.25,
        mask_loss_weight=2.0,
    )

    expected_final_loss = loss_fn(outputs["final_logit"], labels)
    expected_pqa_loss = loss_fn(outputs["pqa_logit"], labels)
    expected_image_loss = loss_fn(outputs["image_logit"], labels)
    expected_mask_loss = compute_pqa_mask_loss(outputs, masks)
    expected_total_loss = (
        expected_final_loss
        + 0.5 * expected_pqa_loss
        + 0.25 * expected_image_loss
        + 2.0 * expected_mask_loss
    )

    assert set(metrics.keys()) == {
        "final_loss",
        "image_loss",
        "pqa_loss",
        "pqa_mask_loss",
        "pqa_local_mil_loss",
        "prior_loss",
        "total_loss",
    }
    assert total_loss.requires_grad
    assert torch.allclose(total_loss, expected_total_loss, atol=1e-6)
    assert torch.allclose(metrics["final_loss"], expected_final_loss.detach(), atol=1e-6)
    assert torch.allclose(metrics["pqa_loss"], expected_pqa_loss.detach(), atol=1e-6)
    assert torch.allclose(metrics["image_loss"], expected_image_loss.detach(), atol=1e-6)
    assert torch.allclose(metrics["pqa_mask_loss"], expected_mask_loss.detach(), atol=1e-6)
    assert torch.equal(metrics["pqa_local_mil_loss"], outputs["final_logit"].new_zeros(()))
    assert torch.equal(metrics["prior_loss"], outputs["final_logit"].new_zeros(()))
    assert torch.allclose(metrics["total_loss"], expected_total_loss.detach(), atol=1e-6)

    total_loss.backward()

    assert outputs["final_logit"].grad is not None
    assert outputs["pqa_logit"].grad is not None
    assert outputs["image_logit"].grad is not None
    assert outputs["pqa_local_logits"].grad is not None
    assert outputs["patch_logit"].grad is None
    assert outputs["base_logit"].grad is None
    assert outputs["holistic_logit"].grad is None


def test_compute_training_loss_allows_missing_optional_heads_when_weights_disabled():
    loss_fn = torch.nn.MSELoss()
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7], requires_grad=True),
    }
    labels = torch.tensor([0.0, 1.0])

    total_loss, metrics = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        masks=None,
        pqa_loss_weight=0.0,
        image_loss_weight=0.0,
        mask_loss_weight=0.0,
    )

    expected_final_loss = loss_fn(outputs["final_logit"], labels)

    assert torch.allclose(total_loss, expected_final_loss, atol=1e-6)
    assert torch.allclose(metrics["final_loss"], expected_final_loss.detach(), atol=1e-6)
    assert torch.equal(metrics["image_loss"], outputs["final_logit"].new_zeros(()))
    assert torch.equal(metrics["pqa_loss"], outputs["final_logit"].new_zeros(()))
    assert torch.equal(metrics["pqa_mask_loss"], outputs["final_logit"].new_zeros(()))
    assert torch.equal(metrics["pqa_local_mil_loss"], outputs["final_logit"].new_zeros(()))


def test_compute_pqa_mask_loss_returns_zero_without_masks():
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7]),
        "pqa_local_logits": torch.randn(2, 2, 8, 8),
    }

    expected_zero = outputs["pqa_local_logits"].new_zeros(())
    actual = compute_pqa_mask_loss(outputs, masks=None)

    assert torch.equal(actual, expected_zero)


def test_compute_pqa_local_mil_loss_uses_topk_anomaly_probability():
    labels = torch.tensor([0.0, 1.0])
    logits = torch.zeros(2, 2, 2, 2, requires_grad=True)
    logits.data[0, 1] = -2.0
    logits.data[1, 1, 0, 0] = 4.0
    outputs = {"pqa_local_logits": logits}

    loss = compute_pqa_local_mil_loss(outputs, labels, topk_ratio=0.25)
    loss.backward()

    assert loss.requires_grad
    assert logits.grad is not None


def test_compute_training_loss_adds_local_mil_when_masks_are_missing():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7], requires_grad=True),
        "image_logit": torch.tensor([0.1, 0.8], requires_grad=True),
        "pqa_logit": torch.tensor([0.4, 0.9], requires_grad=True),
        "pqa_local_logits": torch.randn(2, 2, 8, 8, requires_grad=True),
    }

    total_loss, metrics = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        masks=None,
        local_mil_loss_weight=0.75,
        local_mil_topk_ratio=0.1,
    )
    expected_mil = compute_pqa_local_mil_loss(outputs, labels, topk_ratio=0.1)
    expected_total = (
        loss_fn(outputs["final_logit"], labels)
        + loss_fn(outputs["image_logit"], labels)
        + loss_fn(outputs["pqa_logit"], labels)
        + 0.75 * expected_mil
    )

    assert torch.allclose(total_loss, expected_total, atol=1e-6)
    assert torch.allclose(metrics["pqa_local_mil_loss"], expected_mil.detach(), atol=1e-6)

    total_loss.backward()

    assert outputs["pqa_local_logits"].grad is not None


def test_compute_training_loss_keeps_local_mil_off_by_default_without_masks():
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7], requires_grad=True),
        "image_logit": torch.tensor([0.1, 0.8], requires_grad=True),
        "pqa_logit": torch.tensor([0.4, 0.9], requires_grad=True),
        "pqa_local_logits": torch.randn(2, 2, 8, 8, requires_grad=True),
    }

    total_loss, metrics = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        masks=None,
    )
    total_loss.backward()

    assert metrics["pqa_local_mil_loss"].item() == 0.0
    assert outputs["pqa_local_logits"].grad is None


def test_compute_text_prior_loss_returns_zero_when_text_score_missing():
    """Phase 1-D (2026-04-21): missing text_score -> graceful zero loss, not raise."""
    outputs = {"final_logit": torch.tensor([0.0, 0.0])}
    loss = compute_text_prior_loss(outputs)
    assert torch.equal(loss, outputs["final_logit"].new_zeros(()))


def test_compute_text_prior_loss_is_zero_when_final_matches_text_score():
    """The KL anchor must vanish when the final decision already matches the prior."""
    text_score = torch.tensor([0.2, 0.8])
    # Inverse-sigmoid of 0.2 / 0.8 gives the final_logit that produces identical probs.
    final_logit = torch.log(text_score / (1.0 - text_score))
    outputs = {"final_logit": final_logit, "text_score": text_score}
    loss = compute_text_prior_loss(outputs)
    assert float(loss) < 1e-5


def test_compute_text_prior_loss_grows_when_final_disagrees_with_prior():
    """Regression guard: disagreement must produce strictly positive loss."""
    text_score = torch.tensor([0.9, 0.9])
    outputs = {
        "final_logit": torch.tensor([-5.0, -5.0], requires_grad=True),
        "text_score": text_score,
    }
    loss = compute_text_prior_loss(outputs)
    assert loss > 0.1
    loss.backward()
    # Gradient should flow into final_logit but not into text_score (detached).
    assert outputs["final_logit"].grad is not None


def test_compute_training_loss_adds_prior_when_weighted():
    """Phase 1-D (2026-04-21): prior_loss_weight > 0 injects the KL anchor into total_loss."""
    loss_fn = torch.nn.MSELoss()
    labels = torch.tensor([0.0, 1.0])
    outputs = {
        "final_logit": torch.tensor([0.2, 0.7], requires_grad=True),
        "image_logit": torch.tensor([0.1, 0.8], requires_grad=True),
        "pqa_logit": torch.tensor([0.4, 0.9], requires_grad=True),
        "pqa_local_logits": torch.randn(2, 2, 8, 8, requires_grad=True),
        "text_score": torch.tensor([0.1, 0.9]),
    }

    total_loss, metrics = compute_training_loss(
        outputs=outputs,
        labels=labels,
        loss_fn=loss_fn,
        masks=None,
        prior_loss_weight=0.5,
    )

    expected_prior = compute_text_prior_loss(outputs)
    expected_total = (
        loss_fn(outputs["final_logit"], labels)
        + loss_fn(outputs["image_logit"], labels)
        + loss_fn(outputs["pqa_logit"], labels)
        + 0.5 * expected_prior
    )

    assert torch.allclose(total_loss, expected_total, atol=1e-6)
    assert torch.allclose(metrics["prior_loss"], expected_prior.detach(), atol=1e-6)

    total_loss.backward()
    assert outputs["final_logit"].grad is not None
