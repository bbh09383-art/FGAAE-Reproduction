from pathlib import Path

import full_model


ROOT = Path(__file__).resolve().parent


def main():
    parser = full_model.build_parser({
        "output": str(ROOT / "runs/full_fgaae_v5_stabilized"),
        "pretrained_run": str(ROOT / "runs/full_fgaae_v4"),
        "encoder_lr": 2e-6,
        "freeze_encoder_bn": True,
        "retention_end": 0.85,
        "monitor_encoder_gradients": True,
        "monitor_feature_alignment": True,
        "training_profile": "complete_fgaae_v5_stabilized",
    })
    arguments = parser.parse_args()
    if arguments.command == "smoke-test":
        full_model.smoke_test(stabilized=True)
    else:
        full_model.run(arguments)


if __name__ == "__main__":
    main()
