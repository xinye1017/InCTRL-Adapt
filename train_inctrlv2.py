#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.inctrlv2_dataset import (
    BalancedBatchSampler,
    InCTRLv2DirectoryDataset,
    InCTRLv2NormalDataset,
    resolve_dataset_root,
)
from models.inctrlv2 import build_inctrlv2_model
from models.inctrlv2.losses import compute_inctrlv2_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train InCTRLv2 with DASL and OASL.")
    parser.add_argument("--train_dataset", type=str, required=True, choices=["mvtec", "visa"])
    parser.add_argument("--test_datasets", type=str, nargs="*", default=[])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/inctrlv2")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--oasl_batch_size", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240")
    parser.add_argument("--clip_checkpoint", type=str, default="./vit_b_16_plus_240-laion400m_e32-699c4b84.pt")
    parser.add_argument("--allow_random_init", action="store_true", help="Only for smoke tests; not for paper experiments.")
    parser.add_argument("--input_size", type=int, default=240)
    parser.add_argument("--selected_layers", type=int, nargs="+", default=[7, 9, 11])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--lambda_image", type=float, default=1.0)
    parser.add_argument("--lambda_pixel", type=float, default=1.0)
    parser.add_argument("--lambda_oasl", type=float, default=1.0)
    parser.add_argument("--disable_dasl", action="store_true")
    parser.add_argument("--disable_oasl", action="store_true")
    parser.add_argument("--disable_pixel_loss", action="store_true")
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_normal_json", type=str, nargs="*", default=None)
    parser.add_argument("--train_outlier_json", type=str, nargs="*", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def build_loaders(args, shot: int, seed: int):
    dataset_root = resolve_dataset_root(args.data_root, args.train_dataset)
    main_dataset = InCTRLv2DirectoryDataset(
        dataset_root=dataset_root,
        split="train",
        shots=shot,
        input_size=args.input_size,
        seed=seed,
        normal_json_paths=args.train_normal_json,
        outlier_json_paths=args.train_outlier_json,
    )
    oasl_dataset = InCTRLv2NormalDataset(
        dataset_root=dataset_root,
        split="train",
        shots=shot,
        input_size=args.input_size,
        seed=seed + 1009,
        normal_json_paths=args.train_normal_json,
    )
    steps_per_epoch = args.steps_per_epoch or max(1, len(main_dataset) // args.batch_size)
    batch_sampler = BalancedBatchSampler(main_dataset.labels, args.batch_size, steps_per_epoch, seed=seed)
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs = {"persistent_workers": args.num_workers > 1, "prefetch_factor": 1}
    main_loader = DataLoader(
        main_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        **worker_kwargs,
    )
    oasl_loader = DataLoader(
        oasl_dataset,
        batch_size=args.oasl_batch_size or args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        **worker_kwargs,
    )
    return main_loader, oasl_loader


def train_one_run(args, shot: int, seed: int) -> Path:
    seed_everything(seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = build_inctrlv2_model(
        backbone=args.backbone,
        clip_checkpoint=args.clip_checkpoint,
        device=device,
        selected_layers=args.selected_layers,
        alpha=args.alpha,
        beta=args.beta,
        disable_dasl=args.disable_dasl,
        disable_oasl=args.disable_oasl,
        allow_random_init=args.allow_random_init,
    )
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    main_loader, oasl_loader = build_loaders(args, shot=shot, seed=seed)
    oasl_iter = cycle(oasl_loader)
    train_log = []

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        progress = tqdm(main_loader, desc=f"shot={shot} seed={seed} epoch={epoch + 1}/{args.epochs}")
        for main_batch in progress:
            oasl_batch = next(oasl_iter)
            main_batch = move_batch_to_device(main_batch, device)
            oasl_batch = move_batch_to_device(oasl_batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                main_outputs = model.forward_main(
                    query_image=main_batch["query_image"],
                    prompt_images=main_batch["prompt_images"],
                    class_name=main_batch["class_name"],
                )
                oasl_outputs = None
                if not args.disable_oasl:
                    oasl_outputs = model.forward_oasl(
                        normal_image=oasl_batch["normal_image"],
                        prompt_images=oasl_batch["prompt_images"],
                        class_name=oasl_batch["class_name"],
                    )
                loss_dict = compute_inctrlv2_loss(
                    main_outputs=main_outputs,
                    labels=main_batch["label"],
                    masks=main_batch["mask"],
                    oasl_outputs=oasl_outputs,
                    oasl_masks=oasl_batch.get("normal_mask"),
                    lambda_image=args.lambda_image,
                    lambda_pixel=args.lambda_pixel,
                    lambda_oasl=args.lambda_oasl,
                    disable_dasl=args.disable_dasl,
                    disable_oasl=args.disable_oasl,
                    disable_pixel_loss=args.disable_pixel_loss,
                )
                loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(loss.detach().cpu())
            epoch_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")

        train_log.append({"epoch": epoch + 1, "loss": float(np.mean(epoch_losses))})

    run_dir = Path(args.output_dir) / f"trained_on_{args.train_dataset}" / f"shot_{shot}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "shot": shot,
            "seed": seed,
            "train_dataset": args.train_dataset,
            "test_datasets": args.test_datasets,
            "backbone": args.backbone,
            "selected_layers": args.selected_layers,
            "alpha": args.alpha,
            "beta": args.beta,
        },
        checkpoint_path,
    )
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"shot": shot, "seed": seed}, handle, indent=2)
    with open(run_dir / "train_log.json", "w", encoding="utf-8") as handle:
        json.dump(train_log, handle, indent=2)
    return checkpoint_path


def main():
    args = parse_args()
    checkpoint_paths = []
    for seed in args.seeds:
        for shot in args.shots:
            checkpoint_paths.append(str(train_one_run(args, shot=shot, seed=seed)))
    print(json.dumps({"checkpoints": checkpoint_paths}, indent=2))


if __name__ == "__main__":
    main()
