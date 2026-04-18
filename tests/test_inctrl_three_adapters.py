from types import SimpleNamespace

import torch

from open_clip.config.defaults import get_cfg
from open_clip.inctrl_three_adapters import InCTRLWithAdapters, PromptQueryAdapter


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
    return InCTRLWithAdapters(
        args,
        embed_dim=32,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=False,
        cast_dtype=None,
        patch_layers=(7, 9, 11),
        adapter_hidden_dim=16,
        text_adapter_ctx_len=4,
    )


def test_textual_adapter_outputs_normal_and_anomaly_prototypes():
    model = _build_model()

    normal_proto, anomaly_proto = model._build_adapted_text_prototypes(
        obj_types=["candle", "capsule"],
        device=torch.device("cpu"),
    )

    assert normal_proto.shape == (2, 32)
    assert anomaly_proto.shape == (2, 32)
    assert torch.allclose(normal_proto.norm(dim=-1), torch.ones(2), atol=1e-4)
    assert torch.allclose(anomaly_proto.norm(dim=-1), torch.ones(2), atol=1e-4)


def test_prompt_query_adapter_matches_bruteforce_nearest_neighbor():
    adapter = PromptQueryAdapter(
        feature_dim=4,
        hidden_dim=8,
        num_layers=1,
        image_size=32,
        beta=1.0,
        gamma_r=0.5,
        gamma_c=0.5,
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
    assert outputs["pqa_local_logits"][0].shape == (1, 2, 32, 32)
    assert outputs["pqa_local_scores"][0].shape == (1, 2, 32, 32)


def test_default_phase_is_visual_not_joint():
    model = _build_model()

    assert model.train_phase == "visual"


def test_set_train_phase_toggles_text_and_visual_params():
    model = _build_model()

    model.set_train_phase("text")
    assert all(parameter.requires_grad for parameter in model.get_text_parameters())
    assert all(not parameter.requires_grad for parameter in model.get_visual_parameters())

    model.set_train_phase("visual")
    assert all(not parameter.requires_grad for parameter in model.get_text_parameters())
    assert all(parameter.requires_grad for parameter in model.get_visual_parameters())

    model.set_train_phase("joint")
    assert all(parameter.requires_grad for parameter in model.get_text_parameters())
    assert all(parameter.requires_grad for parameter in model.get_visual_parameters())


def test_default_train_phase_freezes_unused_adapter_ablation_params():
    model = _build_model()

    model.set_train_phase("visual")

    assert any(parameter.requires_grad for parameter in model.prompt_query_adapter.parameters())
    assert all(not parameter.requires_grad for parameter in model.textual_adapter.parameters())

    model.set_train_phase("text")
    assert all(not parameter.requires_grad for parameter in model.prompt_query_adapter.parameters())
    assert any(parameter.requires_grad for parameter in model.textual_adapter.parameters())


def test_forward_with_prompt_images_returns_expected_output_shapes():
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
    assert outputs["branch_weights"].shape == (4,)
    assert outputs["aux"]["patch_map_2d"].shape == (2, 2, 2)
    assert outputs["base_patch_map"].shape == (2, 4)
    assert outputs["hybrid_patch_map"].shape == (2, 4)


def test_default_forward_uses_inctrl_residual_map_for_final_patch_score():
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


def test_default_final_fusion_is_protected_by_max_patch_branch():
    torch.manual_seed(23)
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

    assert outputs["branch_weights"][-1] > 0.95
    assert model.prompt_query_adapter.global_topk == 10
    assert torch.allclose(outputs["holistic_logit"], torch.zeros_like(outputs["holistic_logit"]))
    assert torch.allclose(
        outputs["max_patch_logit"],
        model._positive_scalar(model.alpha_raw) * outputs["max_base_patch_score"],
        atol=1e-6,
    )


def test_default_final_score_uses_raw_inctrl_max_patch_as_backbone():
    torch.manual_seed(29)
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

    expected_raw_max = outputs["raw_base_patch_map"].max(dim=-1).values

    assert torch.allclose(outputs["raw_max_patch_score"], expected_raw_max, atol=1e-6)
    assert torch.allclose(outputs["final_logit"], outputs["raw_max_patch_logit"], atol=1e-6)


def test_image_residual_uses_query_minus_prompt_direction():
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

    expected = outputs["aux"]["adapted_query_global"] - outputs["aux"]["prompt_global_proto"]

    assert torch.allclose(outputs["aux"]["image_residual"], expected, atol=1e-6)


def test_static_residual_text_adapter_starts_from_static_prototypes():
    torch.manual_seed(23)
    model = _build_model()
    model.eval()

    static_normal, static_anomaly = model._build_static_text_prototypes(
        obj_types=["candle"],
        device=torch.device("cpu"),
    )
    adapted_normal, adapted_anomaly = model._build_adapted_text_prototypes(
        obj_types=["candle"],
        device=torch.device("cpu"),
    )

    assert model.textual_adapter.mode == "static_residual"
    assert torch.allclose(adapted_normal, static_normal, atol=1e-6)
    assert torch.allclose(adapted_anomaly, static_anomaly, atol=1e-6)


def test_visual_phase_gives_pqa_global_head_gradients():
    torch.manual_seed(29)
    model = _build_model()
    model.train()
    model.set_train_phase("visual")

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


def test_forward_accepts_shared_normal_list():
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    normal_list = torch.randn(2, 3, 32, 32)

    outputs = model(
        query_image=query_image,
        normal_list=normal_list,
        obj_types=["candle", "candle"],
        return_aux=False,
        return_dict=True,
    )

    assert outputs["final_score"].shape == (2,)
    assert outputs["patch_map"].shape == (2, 4)


def test_prompt_feature_cache_matches_shared_normal_list_outputs():
    torch.manual_seed(7)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    normal_list = torch.randn(2, 3, 32, 32)

    direct_outputs = model(
        query_image=query_image,
        normal_list=normal_list,
        obj_types=["candle", "candle"],
        return_aux=False,
        return_dict=True,
    )
    prompt_feature_cache = model.build_prompt_feature_cache(normal_list=normal_list)
    cached_outputs = model(
        query_image=query_image,
        prompt_feature_cache=prompt_feature_cache,
        obj_types=["candle", "candle"],
        return_aux=False,
        return_dict=True,
    )

    assert torch.allclose(cached_outputs["final_score"], direct_outputs["final_score"], atol=1e-5)
    assert torch.allclose(cached_outputs["patch_map"], direct_outputs["patch_map"], atol=1e-5)


def test_text_prototype_cache_matches_direct_text_outputs():
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

    assert torch.allclose(cached_outputs["final_score"], direct_outputs["final_score"], atol=1e-5)
    assert torch.allclose(cached_outputs["text_score"], direct_outputs["text_score"], atol=1e-5)
