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
