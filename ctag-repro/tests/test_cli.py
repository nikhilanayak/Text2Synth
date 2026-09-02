from ctag_repro.cli import _base_config, build_parser


def test_runs_override_maps_to_runs_per_prompt():
    args = build_parser().parse_args(
        ["generate", "--profile", "smoke", "--prompt", "train horn", "--runs", "10"]
    )

    config = _base_config(args)

    assert config.runs_per_prompt == 10


def test_doctor_can_require_gpu():
    args = build_parser().parse_args(["doctor", "--require-gpu"])

    assert args.command == "doctor"
    assert args.require_gpu is True


def test_direct_inference_defaults_to_strict_head_zero_selection():
    args = build_parser().parse_args(
        ["infer-direct", "--bundle", "model", "--prompt", "train horn"]
    )
    assert args.selection == "direct"
    assert args.device == "cuda"


def test_train_live_has_resumable_balanced_defaults():
    args = build_parser().parse_args(["train-live", "--workspace", "drive/run"])
    assert args.profile == "balanced"
    assert args.handler.__name__ == "command_train_live"
