from types import SimpleNamespace

import torch

from open_clip.inctrl_three_adapters import InCTRLWithAdapters, PromptQueryAdapter


def _build_args(image_size=32, shot=2):
    return SimpleNamespace(
        image_size=image_size,
        shot=shot,
    )


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
        beta=1.0,
        gamma_r=0.5,
        gamma_c=0.5,
    )

    query = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
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
    brute_residual = 0.5 * (1.0 - brute_max)

    assert torch.equal(aligned_indices, brute_idx)
    assert torch.allclose(residual_map, brute_residual, atol=1e-5)


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
    assert outputs["holistic_score"].shape == (2,)
    assert outputs["image_score"].shape == (2,)
    assert outputs["text_score"].shape == (2,)
    assert outputs["patch_map"].shape == (2, 4)
    assert outputs["max_patch_score"].shape == (2,)
    assert outputs["aux"]["patch_map_2d"].shape == (2, 2, 2)


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
