import torch
import torch.nn.functional as F

from open_clip.config.defaults import get_cfg
from open_clip.inctrl_pqa import InCTRLPQA, PQAGlobalHead, PQAdapter


def _build_args(image_size=32, shot=2):
    cfg = get_cfg()
    cfg.image_size = image_size
    cfg.shot = shot
    return cfg


def _build_model():
    args = _build_args()
    vision_cfg = {
        "image_size": 32,
        "layers": 12,
        "width": 64,
        "patch_size": 16,
        "head_width": 32,
        "mlp_ratio": 4.0,
        "output_tokens": True,
    }
    text_cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 32,
        "heads": 4,
        "layers": 2,
    }
    return InCTRLPQA(
        args,
        embed_dim=32,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=False,
        cast_dtype=None,
        patch_layers=(7, 9, 11),
        hidden_dim=16,
    )


def test_prompt_query_adapter_matches_bruteforce_nearest_neighbor():
    adapter = PQAdapter(
        feature_dim=4,
        hidden_dim=8,
        num_layers=1,
        image_size=32,
        beta=1.0,
    )
    adapter.eval()

    query = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]],
        dtype=torch.float32,
    )
    prompt = torch.tensor(
        [[[[1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )

    outputs = adapter(
        query_patch_levels=[query],
        prompt_patch_levels=[prompt],
    )
    aligned_indices = outputs["aligned_indices"][0]
    residual_map = outputs["residual_maps"][0]

    prompt_flat = prompt.reshape(1, -1, 4)
    q_norm = torch.nn.functional.normalize(query, dim=-1)
    p_norm = torch.nn.functional.normalize(prompt_flat, dim=-1)
    brute_sim = torch.einsum("bnc,bmc->bnm", q_norm, p_norm)
    brute_max, brute_idx = brute_sim.max(dim=-1)
    brute_residual = 1.0 - brute_max

    assert torch.equal(aligned_indices, brute_idx)
    assert torch.allclose(residual_map, brute_residual, atol=1e-5)
    assert outputs["pqa_global_logits"][0].shape == (1,)
    assert outputs["pqa_global_logits_2c"][0].shape == (1, 2)
    assert outputs["patch_logits"][0].shape == (1, 4)
    assert outputs["patch_scores"][0].shape == (1, 4)
    assert outputs["pqa_local_logits"][0].shape == (1, 2, 32, 32)
    assert outputs["pqa_local_scores"][0].shape == (1, 2, 32, 32)
    assert outputs["pqa_patch_maps"][0].shape == (1, 4)
    assert "patch_evidence_maps" not in outputs


def test_pqa_global_head_accepts_patch_sequences():
    head = PQAGlobalHead(feature_dim=4, hidden_dim=8, global_topk=3)
    head.eval()

    patch_features = torch.randn(2, 4, 9)
    logits = head(patch_features)

    assert logits.shape == (2, 2)


def test_pqa_global_head_projects_patches_before_pooling():
    torch.manual_seed(5)
    head = PQAGlobalHead(feature_dim=3, hidden_dim=6, global_topk=2)
    head.eval()

    patch_features = torch.tensor(
        [
            [[1.0, 0.0, 2.0, 1.5], [0.5, 1.5, 0.0, 1.0], [2.0, 1.0, 0.5, 0.0]],
            [[0.2, 1.2, 0.5, 1.5], [1.0, 0.0, 1.0, 0.0], [0.3, 0.8, 1.6, 0.4]],
        ],
        dtype=torch.float32,
    )

    projected = head.project_patch_logits(patch_features)
    manual_logits = head.pool_projected_logits(projected)

    assert projected.shape == (2, 2, 4)
    assert manual_logits.shape == (2, 2)
    assert torch.allclose(head(patch_features), manual_logits, atol=1e-6)


def test_prompt_query_adapter_patch_outputs_come_from_local_head():
    torch.manual_seed(13)
    adapter = PQAdapter(
        feature_dim=4,
        hidden_dim=8,
        num_layers=1,
        image_size=32,
        beta=1.0,
    )
    adapter.eval()

    query = torch.randn(1, 4, 4)
    prompt = torch.randn(1, 2, 4, 4)

    outputs = adapter(
        query_patch_levels=[query],
        prompt_patch_levels=[prompt],
    )

    expected_patch_logits_2c = F.interpolate(
        outputs["pqa_local_logits"][0],
        size=(2, 2),
        mode="bilinear",
        align_corners=False,
    )
    expected_patch_scores_2c = torch.softmax(expected_patch_logits_2c, dim=1)
    expected_patch_logits = (expected_patch_logits_2c[:, 1] - expected_patch_logits_2c[:, 0]).flatten(1)
    expected_patch_scores = expected_patch_scores_2c[:, 1].flatten(1)

    assert torch.allclose(outputs["patch_logits"][0], expected_patch_logits, atol=1e-6)
    assert torch.allclose(outputs["patch_scores"][0], expected_patch_scores, atol=1e-6)
    assert torch.allclose(outputs["pqa_patch_maps"][0], expected_patch_scores, atol=1e-6)


def test_model_exposes_only_pqa_training_interface():
    model = _build_model()

    assert hasattr(model, "enable_pqa_training")
    assert not hasattr(model, "set_train_phase")


def test_forward_with_prompt_images_returns_pqa_only_outputs():
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)

    outputs = model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )

    assert outputs["final_score"].shape == (2,)
    assert outputs["final_logit"].shape == (2,)
    assert outputs["base_logit"].shape == (2,)
    assert outputs["image_logit"].shape == (2,)
    assert outputs["text_logit"].shape == (2,)
    assert outputs["pqa_logit"].shape == (2,)
    assert outputs["holistic_score"].shape == (2,)
    assert outputs["image_score"].shape == (2,)
    assert outputs["text_score"].shape == (2,)
    assert outputs["pqa_score"].shape == (2,)
    assert outputs["pqa_patch_logit"].shape == (2, 4)
    assert outputs["pqa_patch_score"].shape == (2, 4)
    assert outputs["patch_map"].shape == (2, 4)
    assert outputs["max_patch_score"].shape == (2,)
    assert outputs["max_patch_logit"].shape == (2,)
    assert outputs["aux"]["patch_map_2d"].shape == (2, 2, 2)
    assert outputs["base_patch_map"].shape == (2, 4)
    assert outputs["hybrid_patch_map"].shape == (2, 4)
    assert torch.allclose(outputs["hybrid_patch_map"], outputs["pqa_patch_score"], atol=1e-6)
    assert "static_text_logit" not in outputs
    assert "adaptive_text_logit" not in outputs
    assert "branch_weights" not in outputs
    assert "text_static_reg" not in outputs


def test_default_forward_keeps_inctrl_patch_map_as_final_patch_map():
    torch.manual_seed(17)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)

    outputs = model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )

    expected_max = outputs["base_patch_map"].max(dim=-1).values

    assert torch.allclose(outputs["patch_map"], outputs["base_patch_map"], atol=1e-6)
    assert torch.allclose(outputs["max_patch_score"], expected_max, atol=1e-6)
    assert torch.allclose(outputs["max_base_patch_score"], expected_max, atol=1e-6)
    assert torch.allclose(outputs["final_logit"], outputs["base_logit"], atol=1e-6)


def test_image_residual_uses_original_prompt_minus_query_direction():
    torch.manual_seed(19)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)

    outputs = model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )

    expected = outputs["aux"]["prompt_global_proto"] - outputs["aux"]["raw_query_global"]

    assert torch.allclose(outputs["aux"]["image_residual"], expected, atol=1e-6)


def test_text_prototype_cache_is_fixed_and_matches_direct_outputs():
    torch.manual_seed(11)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    normal_list = torch.randn(2, 3, 32, 32)
    prompt_feature_cache = model.build_prompt_feature_cache(normal_list=normal_list)

    direct_outputs = model(
        query_image=query_image,
        prompt_feature_cache=prompt_feature_cache,
        obj_types=["candle", "candle"],
        return_aux=False,
        return_dict=True,
    )
    text_prototype_cache = model.build_text_prototype_cache(
        obj_types=["candle"],
        device=torch.device("cpu"),
    )
    cached_outputs = model(
        query_image=query_image,
        prompt_feature_cache=prompt_feature_cache,
        obj_types=["candle", "candle"],
        text_prototype_cache=text_prototype_cache,
        return_aux=False,
        return_dict=True,
    )

    assert set(text_prototype_cache.keys()) == {"normal_proto", "anomaly_proto"}
    assert torch.allclose(cached_outputs["final_score"], direct_outputs["final_score"], atol=1e-5)
    assert torch.allclose(cached_outputs["text_score"], direct_outputs["text_score"], atol=1e-5)


def test_pqa_training_gives_pqa_global_head_gradients():
    torch.manual_seed(29)
    model = _build_model()
    model.train()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)
    outputs = model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=False,
        return_dict=True,
    )

    loss = outputs["pqa_logit"].sum()
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.prompt_query_adapter.global_heads.parameters()
    )
