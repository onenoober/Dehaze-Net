import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import optim

from model import Backbone, DEANet


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated training checkpoint initialized from an "
            "official DEA-Net inference .pth file."
        )
    )
    parser.add_argument(
        "--official_checkpoint",
        type=str,
        default=str(ROOT_DIR / "trained_models" / "HAZE4K" / "PSNR3426_SSIM9885.pth"),
        help="Official Backbone .pth checkpoint. This is a model state_dict, not a train resume checkpoint.",
    )
    parser.add_argument("--exp_dir", type=str, default="../experiment")
    parser.add_argument("--dataset", type=str, default="HAZE4K")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--output_checkpoint",
        type=str,
        default="",
        help="Optional explicit output .pk path. Defaults to exp_dir/dataset/model_name/saved_model/official_warmstart_step0.pk.",
    )
    parser.add_argument("--start_lr", type=float, default=0.00002)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data_parallel_model_keys",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
        help="Save model keys with module. prefix when the later train.py resume will wrap DataParallel.",
    )
    parser.add_argument("--check_forward", action="store_true")
    parser.add_argument("--check_size", type=int, default=64)
    parser.add_argument("--check_tolerance", type=float, default=1e-5)
    parser.add_argument("--allow_forward_mismatch", action="store_true")

    parser.add_argument("--use_lf_prior", action="store_true")
    parser.add_argument("--lf_prior_channels", type=int, default=8)
    parser.add_argument("--lf_prior_pool", type=int, default=8)
    parser.add_argument("--lf_prior_gate_init", type=float, default=0.0)
    parser.add_argument("--lf_prior_residual_center", action="store_true")
    parser.add_argument("--lf_prior_train_dropout", type=float, default=0.0)
    parser.add_argument("--lf_prior_gate_max", type=float, default=0.0)
    parser.add_argument("--lf_prior_injection", type=str, default="pre_mix", choices=["pre_mix", "post_mix"])
    parser.add_argument("--lf_conditional_mask", action="store_true")
    parser.add_argument("--lf_mask_hidden_channels", type=int, default=8)
    parser.add_argument("--lf_mask_init_bias", type=float, default=2.0)
    parser.add_argument("--lf_haze_aware_mask", action="store_true")
    parser.add_argument("--lf_haze_mask_strength", type=float, default=1.0)
    parser.add_argument("--lf_residual_calibration", action="store_true")
    parser.add_argument("--lf_calib_hidden_channels", type=int, default=8)
    parser.add_argument("--lf_calib_alpha_max", type=float, default=1.0)
    parser.add_argument("--lf_residual_selector", action="store_true")
    parser.add_argument("--lf_selector_hidden_channels", type=int, default=8)
    parser.add_argument("--lf_selector_init_bias", type=float, default=2.0)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return strip_module_prefix(checkpoint)


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return OrderedDict((key.replace("module.", "", 1), value) for key, value in state_dict.items())


def build_model(args):
    return DEANet(
        base_dim=32,
        use_lf_prior=args.use_lf_prior,
        lf_prior_channels=args.lf_prior_channels,
        lf_prior_pool=args.lf_prior_pool,
        lf_prior_gate_init=args.lf_prior_gate_init,
        lf_prior_residual_center=args.lf_prior_residual_center,
        lf_prior_train_dropout=args.lf_prior_train_dropout,
        lf_prior_gate_max=args.lf_prior_gate_max,
        lf_prior_injection=args.lf_prior_injection,
        lf_conditional_mask=args.lf_conditional_mask,
        lf_mask_hidden_channels=args.lf_mask_hidden_channels,
        lf_mask_init_bias=args.lf_mask_init_bias,
        lf_haze_aware_mask=args.lf_haze_aware_mask,
        lf_haze_mask_strength=args.lf_haze_mask_strength,
        lf_residual_calibration=args.lf_residual_calibration,
        lf_calib_hidden_channels=args.lf_calib_hidden_channels,
        lf_calib_alpha_max=args.lf_calib_alpha_max,
        lf_residual_selector=args.lf_residual_selector,
        lf_selector_hidden_channels=args.lf_selector_hidden_channels,
        lf_selector_init_bias=args.lf_selector_init_bias,
    )


def zero_deconv_side_branches(target_state, prefix, stats):
    for branch in ("conv1_1", "conv1_2", "conv1_3", "conv1_4"):
        for suffix in ("weight", "bias"):
            key = "{}conv1.{}.conv.{}".format(prefix, branch, suffix)
            if key in target_state:
                target_state[key] = torch.zeros_like(target_state[key])
                stats["zeroed_deconv_side_branch_keys"].append(key)


def transfer_official_weights(official_state, target_state):
    stats = {
        "exact_loaded_keys": [],
        "deconv_mapped_keys": [],
        "zeroed_deconv_side_branch_keys": [],
        "shape_mismatch_keys": [],
        "unused_official_keys": [],
        "target_keys_not_from_official": [],
    }

    loaded_targets = set()
    used_official = set()

    for key, value in official_state.items():
        if key in target_state:
            if target_state[key].shape == value.shape:
                target_state[key] = value.clone()
                stats["exact_loaded_keys"].append(key)
                loaded_targets.add(key)
                used_official.add(key)
            else:
                stats["shape_mismatch_keys"].append(
                    {
                        "key": key,
                        "official_shape": list(value.shape),
                        "target_shape": list(target_state[key].shape),
                    }
                )

    for key, value in official_state.items():
        if key.endswith(".conv1.weight"):
            prefix = key[: -len("conv1.weight")]
            target_key = "{}conv1.conv1_5.weight".format(prefix)
            if target_key in target_state and target_state[target_key].shape == value.shape:
                target_state[target_key] = value.clone()
                loaded_targets.add(target_key)
                used_official.add(key)
                stats["deconv_mapped_keys"].append({"official": key, "target": target_key})
                zero_deconv_side_branches(target_state, prefix, stats)
            elif target_key in target_state:
                stats["shape_mismatch_keys"].append(
                    {
                        "key": key,
                        "target_key": target_key,
                        "official_shape": list(value.shape),
                        "target_shape": list(target_state[target_key].shape),
                    }
                )
        elif key.endswith(".conv1.bias"):
            prefix = key[: -len("conv1.bias")]
            target_key = "{}conv1.conv1_5.bias".format(prefix)
            if target_key in target_state and target_state[target_key].shape == value.shape:
                target_state[target_key] = value.clone()
                loaded_targets.add(target_key)
                used_official.add(key)
                stats["deconv_mapped_keys"].append({"official": key, "target": target_key})
            elif target_key in target_state:
                stats["shape_mismatch_keys"].append(
                    {
                        "key": key,
                        "target_key": target_key,
                        "official_shape": list(value.shape),
                        "target_shape": list(target_state[target_key].shape),
                    }
                )

    stats["unused_official_keys"] = [key for key in official_state.keys() if key not in used_official]
    stats["target_keys_not_from_official"] = [
        key for key in target_state.keys() if key not in loaded_targets and key not in stats["zeroed_deconv_side_branch_keys"]
    ]
    return target_state, stats


def maybe_prefix_data_parallel(model_state, args):
    if args.data_parallel_model_keys == "yes":
        prefix = True
    elif args.data_parallel_model_keys == "no":
        prefix = False
    else:
        prefix = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if not prefix:
        return model_state, False
    return OrderedDict(("module." + key, value) for key, value in model_state.items()), True


def forward_equivalence_check(args, official_state, warm_state):
    if not args.check_forward:
        return None
    if not torch.cuda.is_available():
        return {
            "status": "skipped",
            "reason": "CUDA is unavailable; DEConv forward allocates CUDA tensors in this fork.",
        }

    device = torch.device("cuda")
    official_model = Backbone(base_dim=32).to(device)
    official_model.load_state_dict(official_state)
    official_model.eval()

    warm_model = build_model(args).to(device)
    warm_model.load_state_dict(warm_state)
    warm_model.eval()

    torch.manual_seed(args.seed + 1000)
    sample = torch.rand(1, 3, args.check_size, args.check_size, device=device)
    with torch.no_grad():
        official_out = official_model(sample)
        warm_out = warm_model(sample)
    max_abs = torch.max(torch.abs(official_out - warm_out)).item()
    result = {
        "status": "passed" if max_abs <= args.check_tolerance else "failed",
        "max_abs_diff": max_abs,
        "tolerance": args.check_tolerance,
        "check_size": args.check_size,
    }
    if result["status"] == "failed" and not args.allow_forward_mismatch:
        raise RuntimeError("Forward equivalence check failed: {}".format(result))
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    official_path = Path(args.official_checkpoint)
    if not official_path.is_file():
        raise FileNotFoundError("Official checkpoint not found: {}".format(official_path))

    model_dir = Path(args.exp_dir) / args.dataset / args.model_name
    output_path = Path(args.output_checkpoint) if args.output_checkpoint else model_dir / "saved_model" / "official_warmstart_step0.pk"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (model_dir / "saved_data").mkdir(parents=True, exist_ok=True)

    official_state = load_checkpoint(str(official_path))
    model = build_model(args)
    target_state = OrderedDict((key, value.clone()) for key, value in model.state_dict().items())
    warm_state, stats = transfer_official_weights(official_state, target_state)
    model.load_state_dict(warm_state, strict=True)

    check = forward_equivalence_check(args, official_state, warm_state)
    optimizer = optim.Adam(
        params=filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.start_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    saved_model_state, used_data_parallel_keys = maybe_prefix_data_parallel(model.state_dict(), args)
    checkpoint = {
        "epoch": 0,
        "step": 0,
        "max_psnr": 0,
        "max_ssim": 0,
        "ssims": [],
        "psnrs": [],
        "losses": [],
        "loss_log": {},
        "psnr_log": [],
        "early_stop_state": {},
        "model": saved_model_state,
        "optimizer": optimizer.state_dict(),
        "warmstart_meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "official_checkpoint": str(official_path),
            "output_checkpoint": str(output_path),
            "model_name": args.model_name,
            "dataset": args.dataset,
            "start_lr": args.start_lr,
            "seed": args.seed,
            "used_data_parallel_model_keys": used_data_parallel_keys,
            "architecture": {
                "use_lf_prior": args.use_lf_prior,
                "lf_prior_gate_init": args.lf_prior_gate_init,
                "lf_prior_injection": args.lf_prior_injection,
                "lf_residual_calibration": args.lf_residual_calibration,
                "lf_residual_selector": args.lf_residual_selector,
                "lf_conditional_mask": args.lf_conditional_mask,
                "lf_haze_aware_mask": args.lf_haze_aware_mask,
            },
            "transfer_stats": {
                "exact_loaded_count": len(stats["exact_loaded_keys"]),
                "deconv_mapped_count": len(stats["deconv_mapped_keys"]),
                "zeroed_deconv_side_branch_count": len(stats["zeroed_deconv_side_branch_keys"]),
                "shape_mismatch_count": len(stats["shape_mismatch_keys"]),
                "unused_official_count": len(stats["unused_official_keys"]),
                "target_keys_not_from_official_count": len(stats["target_keys_not_from_official"]),
            },
            "forward_equivalence_check": check,
        },
    }
    torch.save(checkpoint, output_path)

    report = {
        "checkpoint": checkpoint["warmstart_meta"],
        "transfer_stats": stats,
    }
    report_path = model_dir / "saved_data" / "official_warmstart_report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("Official checkpoint:", official_path)
    print("Warm-start checkpoint:", output_path)
    print("Report:", report_path)
    print("Exact keys loaded:", len(stats["exact_loaded_keys"]))
    print("DEConv keys mapped:", len(stats["deconv_mapped_keys"]))
    print("DEConv side-branch keys zeroed:", len(stats["zeroed_deconv_side_branch_keys"]))
    print("Shape mismatches:", len(stats["shape_mismatch_keys"]))
    print("Target keys not from official:", len(stats["target_keys_not_from_official"]))
    if check is not None:
        print("Forward equivalence:", check)


if __name__ == "__main__":
    main()
