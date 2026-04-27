import torch
import torch.nn.functional as F

from open_clip.object_agnostic_text import ObjectAgnosticTextBranch


class FakeTextEncoder:
    def __call__(self, tokens, normalize=False):
        feats = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=tokens.device)
        return F.normalize(feats, dim=-1) if normalize else feats


def fake_tokenizer(texts):
    assert texts == ["a photo of a normal object.", "a photo of a damaged object."]
    return torch.ones(len(texts), 77, dtype=torch.long)


def test_object_agnostic_text_branch_scores_global_and_patch_features():
    branch = ObjectAgnosticTextBranch(
        templates=["a photo of a normal object.", "a photo of a damaged object."],
        logit_scale=10.0,
    )
    global_feat = F.normalize(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), dim=-1)
    patch_feat = F.normalize(
        torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ]
        ),
        dim=-1,
    )

    outputs = branch(
        encode_text=FakeTextEncoder(),
        tokenizer=fake_tokenizer,
        global_feat=global_feat,
        patch_feat=patch_feat,
        image_size=32,
    )

    assert outputs["text_logit"].shape == (2,)
    assert outputs["text_score"].shape == (2,)
    assert outputs["text_map"].shape == (1, 1, 32, 32)
    assert outputs["text_score"][0] > 0.99
    assert outputs["text_score"][1] < 0.01
