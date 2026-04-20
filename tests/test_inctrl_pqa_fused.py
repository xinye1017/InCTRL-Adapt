import torch

from open_clip.config.defaults import get_cfg
from open_clip.inctrl_pqa_fused import InCTRLPQA, PQAGlobalHead
from open_clip.inctrl_pqa_losses import compute_pqa_mask_loss


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
        feature_is_projected=False,
    )


def _forward(model):
    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)
    return model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )


def test_fused_forward_exposes_new_contract():
    torch.manual_seed(31)
    model = _build_model()
    model.eval()

    outputs = _forward(model)

    required_keys = {
        "final_score",
        "final_logit",
        "patch_score",
        "patch_logit",
        "pqa_score",
        "pqa_logit",
        "image_score",
        "image_logit",
        "text_score",
        "text_logit",
        "fused_patch_map",
        "pqa_local_logits",
        "pqa_local_scores",
        "patch_map_fusion_weights",
        "aux",
    }

    assert required_keys.issubset(outputs.keys())
    assert outputs["final_logit"].shape == (2,)
    assert outputs["patch_logit"].shape == (2,)
    assert outputs["pqa_logit"].shape == (2,)
    assert outputs["image_logit"].shape == (2,)
    assert outputs["text_logit"].shape == (2,)
    assert outputs["pqa_local_logits"].shape == (2, 2, 32, 32)
    assert outputs["pqa_local_scores"].shape == (2, 2, 32, 32)
    assert outputs["fused_patch_map"].shape == (2, 4)


def test_pqadapter_returns_new_patch_payload():
    torch.manual_seed(35)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)

    _, query_patch_tokens = model._encode_visual_features(query_image)
    query_patch_levels = model._as_query_level_list(
        model._prepare_patch_levels(query_patch_tokens, batch_size=2, num_shots=1)
    )

    _, prompt_patch_levels, _ = model._encode_prompt_features(
        query_image=query_image,
        prompt_images=prompt_images,
    )

    outputs = model.prompt_query_adapter(
        query_patch_levels=query_patch_levels,
        prompt_patch_levels=prompt_patch_levels,
        beta=torch.tensor(model.beta, device=query_patch_levels[0].device, dtype=query_patch_levels[0].dtype),
    )

    required_keys = {
        "residual_maps",
        "inctrl_patch_maps",
        "pqa_patch_maps",
        "patch_logits",
        "patch_scores",
        "pqa_global_logits",
        "pqa_local_logits",
        "aligned_indices",
        "aligned_prompt_features",
    }

    assert required_keys.issubset(outputs.keys())
    assert "layer_weights" not in outputs

    num_layers = len(model.patch_layers)
    assert len(outputs["residual_maps"]) == num_layers
    assert len(outputs["inctrl_patch_maps"]) == num_layers
    assert len(outputs["pqa_patch_maps"]) == num_layers
    assert len(outputs["patch_logits"]) == num_layers
    assert len(outputs["patch_scores"]) == num_layers
    assert len(outputs["pqa_global_logits"]) == num_layers
    assert len(outputs["pqa_local_logits"]) == num_layers
    assert len(outputs["aligned_indices"]) == num_layers
    assert len(outputs["aligned_prompt_features"]) == num_layers

    assert outputs["residual_maps"][0].shape == (2, 4)
    assert outputs["patch_logits"][0].shape == (2, 4)
    assert outputs["patch_scores"][0].shape == (2, 4)
    assert outputs["pqa_global_logits"][0].shape == (2,)
    assert outputs["pqa_local_logits"][0].shape == (2, 2, 32, 32)
    assert outputs["aligned_indices"][0].shape == (2, 4)
    assert outputs["aligned_prompt_features"][0].shape == (2, 4, 32)


def test_forward_aux_contains_new_diagnostics():
    torch.manual_seed(36)
    model = _build_model()
    model.eval()

    outputs = _forward(model)
    aux = outputs["aux"]

    expected_aux_keys = {
        "patch_map_2d",
        "per_layer_pqa_patch_map",
        "per_layer_pqa_patch_logit",
        "per_layer_residual",
        "aligned_indices",
        "raw_query_global",
        "prompt_global_proto",
        "image_residual",
        "pqa_layer_weights",
        "patch_layer_weights",
        "patch_map_fusion_weights",
        "pqa_global_pool_weights",
        "text_prototypes",
    }

    assert expected_aux_keys.issubset(aux.keys())

    patch_side = int(model.num_patches ** 0.5)
    assert aux["patch_map_2d"].shape == (2, patch_side, patch_side)

    num_layers = len(model.patch_layers)
    assert len(aux["per_layer_pqa_patch_map"]) == num_layers
    assert len(aux["per_layer_residual"]) == num_layers

    pqa_wts = aux["pqa_layer_weights"]
    patch_wts = aux["patch_layer_weights"]
    assert pqa_wts.shape == (num_layers,)
    assert patch_wts.shape == (num_layers,)
    assert torch.allclose(pqa_wts.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(patch_wts.sum(), torch.tensor(1.0), atol=1e-6)


def test_final_logit_backprop_updates_trainable_path():
    torch.manual_seed(37)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["final_logit"].sum().backward()

    assert any(parameter.grad is not None for parameter in model.get_trainable_parameters())


def test_patch_logit_backprop_updates_patch_map_fusion_logits():
    torch.manual_seed(41)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["patch_logit"].sum().backward()

    assert model.patch_map_fusion_logits.grad is not None


def test_pqa_logit_backprop_updates_global_heads():
    torch.manual_seed(47)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["pqa_logit"].sum().backward()

    assert any(
        parameter.grad is not None
        for parameter in model.prompt_query_adapter.global_heads.parameters()
    )


def test_pqa_mask_loss_backprop_updates_local_heads():
    torch.manual_seed(43)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    masks = torch.zeros(2, 1, 32, 32)
    masks[1, :, 8:24, 8:24] = 1.0
    loss = compute_pqa_mask_loss(outputs, masks)
    loss.backward()

    assert loss > 0
    assert any(
        parameter.grad is not None
        for parameter in model.prompt_query_adapter.local_heads.parameters()
    )


def test_return_dict_false_forces_tuple_output():
    torch.manual_seed(53)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 2, 3, 32, 32)
    obj_types = ["candle", "candle"]

    dict_outputs = model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=obj_types,
        return_aux=True,
        return_dict=True,
    )
    tuple_outputs = model(
        query_image=query_image,
        prompt_images=prompt_images,
        obj_types=obj_types,
        return_aux=True,
        return_dict=False,
    )

    assert isinstance(tuple_outputs, tuple)
    assert len(tuple_outputs) == 2
    assert tuple_outputs[0].shape == (2,)
    assert tuple_outputs[1].shape == (2, 4)
    assert torch.allclose(tuple_outputs[0], dict_outputs["final_score"], atol=1e-6)
    assert torch.allclose(tuple_outputs[1], dict_outputs["fused_patch_map"], atol=1e-6)


def test_return_dict_none_uses_model_default_output_dict():
    torch.manual_seed(57)
    model = _build_model()
    model.eval()

    outputs = model(
        query_image=torch.randn(2, 3, 32, 32),
        prompt_images=torch.randn(2, 2, 3, 32, 32),
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=None,
    )

    assert isinstance(outputs, dict)
    assert outputs["final_logit"].shape == (2,)


def test_return_dict_none_can_use_tuple_model_default():
    torch.manual_seed(58)
    model = _build_model()
    model.output_dict = False
    model.eval()

    outputs = model(
        query_image=torch.randn(2, 3, 32, 32),
        prompt_images=torch.randn(2, 2, 3, 32, 32),
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=None,
    )

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2


def test_shared_5d_prompt_images_expand_to_query_batch():
    torch.manual_seed(59)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    shared_prompt_images = torch.randn(1, 2, 3, 32, 32)
    coerced = model._coerce_prompt_images(query_image, prompt_images=shared_prompt_images)

    assert coerced.shape == (2, 2, 3, 32, 32)

    outputs = model(
        query_image=query_image,
        prompt_images=shared_prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )
    assert outputs["final_logit"].shape == (2,)


def test_4d_prompt_images_broadcast_for_multi_query_batch():
    torch.manual_seed(61)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 3, 32, 32)
    coerced = model._coerce_prompt_images(query_image, prompt_images=prompt_images)

    assert coerced.shape == (2, 2, 3, 32, 32)


def test_global_head_supports_single_item_training_batches():
    torch.manual_seed(63)
    head = PQAGlobalHead(feature_dim=32, hidden_dim=16)
    head.train()

    logits = head(torch.randn(1, 4, 32))
    logits.sum().backward()

    assert logits.shape == (1,)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_global_topk_is_not_part_of_fused_global_head_state():
    head = PQAGlobalHead(feature_dim=32, hidden_dim=16)

    assert not hasattr(head, "global_topk")


def test_prompt_feature_cache_round_trip_still_works():
    torch.manual_seed(67)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 3, 32, 32)
    cache = model.build_prompt_feature_cache(prompt_images=prompt_images)

    assert cache["prompt_global"].shape == (2, 32)
    assert len(cache["prompt_patch_levels"]) == 3
    assert cache["prompt_patch_levels"][0].shape == (2, 4, 32)
    assert cache["num_shots"] == 2

    direct_outputs = model(
        query_image=query_image,
        normal_list=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )
    cached_outputs = model(
        query_image=query_image,
        prompt_feature_cache=cache,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )

    assert torch.allclose(cached_outputs["final_logit"], direct_outputs["final_logit"], atol=1e-6)
    assert torch.allclose(cached_outputs["fused_patch_map"], direct_outputs["fused_patch_map"], atol=1e-6)


def test_prompt_feature_cache_rejects_wrong_layer_count():
    torch.manual_seed(68)
    model = _build_model()
    model.eval()
    cache = {
        "prompt_global": torch.randn(2, 32),
        "prompt_patch_levels": [torch.randn(2, 4, 32)],
        "num_shots": 2,
    }

    try:
        model(
            query_image=torch.randn(2, 3, 32, 32),
            prompt_feature_cache=cache,
            obj_types=["candle", "candle"],
            return_dict=True,
        )
    except ValueError as error:
        assert "prompt_feature_cache" in str(error)
        assert "layers" in str(error)
    else:
        raise AssertionError("Expected ValueError for invalid prompt cache layer count")


def test_prompt_feature_cache_rejects_category_mismatch():
    torch.manual_seed(69)
    model = _build_model()
    model.eval()
    cache = model.build_prompt_feature_cache(
        prompt_images=torch.randn(2, 3, 32, 32),
        category="candle",
    )

    try:
        model(
            query_image=torch.randn(2, 3, 32, 32),
            prompt_feature_cache=cache,
            obj_types=["capsule", "capsule"],
            return_dict=True,
        )
    except ValueError as error:
        assert "prompt_feature_cache" in str(error)
        assert "category" in str(error)
    else:
        raise AssertionError("Expected ValueError for prompt cache category mismatch")


def test_legacy_prompt_cache_without_category_rejects_mixed_batches():
    torch.manual_seed(70)
    model = _build_model()
    model.eval()
    cache = model.build_prompt_feature_cache(prompt_images=torch.randn(2, 3, 32, 32))

    try:
        model(
            query_image=torch.randn(2, 3, 32, 32),
            prompt_feature_cache=cache,
            obj_types=["candle", "capsule"],
            return_dict=True,
        )
    except ValueError as error:
        assert "prompt_feature_cache" in str(error)
        assert "mixed-category" in str(error)
    else:
        raise AssertionError("Expected ValueError for category-less cache with mixed batch")


def test_text_prototype_cache_round_trip_still_works():
    torch.manual_seed(71)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    prompt_images = torch.randn(2, 3, 32, 32)
    cache = model.build_text_prototype_cache(
        obj_types=["candle", "candle"],
        device=torch.device("cpu"),
    )

    assert cache["normal_proto"].shape == (2, 32)
    assert cache["anomaly_proto"].shape == (2, 32)

    direct_outputs = model(
        query_image=query_image,
        normal_list=prompt_images,
        obj_types=["candle", "candle"],
        return_aux=True,
        return_dict=True,
    )
    cached_outputs = model(
        query_image=query_image,
        normal_list=prompt_images,
        obj_types=["candle", "candle"],
        text_prototype_cache=cache,
        return_aux=True,
        return_dict=True,
    )

    assert torch.allclose(cached_outputs["text_logit"], direct_outputs["text_logit"], atol=1e-6)
    assert torch.allclose(cached_outputs["final_logit"], direct_outputs["final_logit"], atol=1e-6)


def test_text_prototype_cache_rejects_mismatched_shapes():
    torch.manual_seed(72)
    model = _build_model()
    model.eval()
    cache = {
        "normal_proto": torch.randn(2, 32),
        "anomaly_proto": torch.randn(2, 31),
    }

    try:
        model(
            query_image=torch.randn(2, 3, 32, 32),
            normal_list=torch.randn(2, 3, 32, 32),
            obj_types=["candle", "candle"],
            text_prototype_cache=cache,
            return_dict=True,
        )
    except ValueError as error:
        assert "text_prototype_cache" in str(error)
        assert "shape" in str(error)
    else:
        raise AssertionError("Expected ValueError for mismatched text prototype cache")


def test_compute_patch_residuals_method_exists():
    torch.manual_seed(73)
    model = _build_model()
    model.eval()

    query_image = torch.randn(2, 3, 32, 32)
    _, query_patch_tokens = model._encode_visual_features(query_image)
    query_patch_levels = model._as_query_level_list(
        model._prepare_patch_levels(query_patch_tokens, batch_size=2, num_shots=1)
    )
    _, prompt_patch_levels, _ = model._encode_prompt_features(
        query_image=query_image,
        normal_list=torch.randn(2, 3, 32, 32),
    )

    raw_residual_outputs = model._compute_patch_residuals(
        query_patch_levels=query_patch_levels,
        prompt_patch_levels=prompt_patch_levels,
    )

    assert "residual_maps" in raw_residual_outputs
    assert len(raw_residual_outputs["residual_maps"]) == len(model.patch_layers)
    assert raw_residual_outputs["residual_maps"][0].shape == (2, 4)


def test_get_layer_weights_method_exists():
    torch.manual_seed(74)
    model = _build_model()
    model.eval()

    weights = model._get_layer_weights(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert weights.shape == (len(model.patch_layers),)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_forward_legacy_returns_original_tuple_contract():
    torch.manual_seed(75)
    model = _build_model()
    model.eval()
    query_image = torch.randn(2, 3, 32, 32)
    normal_1 = torch.randn(2, 3, 32, 32)
    normal_2 = torch.randn(2, 3, 32, 32)
    obj_types = ["candle", "candle"]

    direct_outputs = model(
        query_image=query_image,
        prompt_images=torch.stack([normal_1, normal_2], dim=1),
        obj_types=obj_types,
        return_dict=True,
    )
    legacy_score, legacy_image_score = model.forward_legacy(
        tokenizer=None,
        image=[query_image, normal_1, normal_2],
        text=obj_types,
    )

    assert torch.allclose(legacy_score, direct_outputs["final_score"], atol=1e-6)
    assert legacy_image_score.shape == (2,)


def test_get_patch_layer_weights_method_exists():
    torch.manual_seed(76)
    model = _build_model()
    model.eval()

    weights = model._get_patch_layer_weights(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert weights.shape == (len(model.patch_layers),)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_reduce_patch_map_to_logit():
    torch.manual_seed(77)
    model = _build_model()
    model.eval()

    patch_map = torch.rand(2, 4)
    logit = model._reduce_patch_map_to_logit(patch_map, topk=3)

    assert logit.shape == (2,)
    max_score = patch_map.max(dim=-1).values
    topk_mean = patch_map.topk(3, dim=-1).values.mean(dim=-1)
    expected_score = 0.5 * max_score + 0.5 * topk_mean
    expected_logit = model._score_to_logit(expected_score)
    assert torch.allclose(logit, expected_logit, atol=1e-6)


def test_decision_head_backprop_updates_all_branches():
    torch.manual_seed(78)
    model = _build_model()
    model.train()

    outputs = _forward(model)
    outputs["final_logit"].sum().backward()

    assert model.decision_head.net[1].weight.grad is not None
    assert model.patch_map_fusion_logits.grad is not None
    assert model.layer_weights_logits.grad is not None
    assert model.patch_layer_weights_logits.grad is not None
