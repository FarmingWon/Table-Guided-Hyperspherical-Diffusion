import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Table-Guided Hyperspherical Diffusion for column type annotation."
    )
    parser.add_argument("--data", default="gt-semtab22-dbpedia-all", help="Dataset name under data/.")
    parser.add_argument("--model_name", default="sti_ddpm", help="Model name for result directory layout.")
    parser.add_argument("--result_dir", default="result", help="Root directory for experiment artifacts.")
    parser.add_argument("--epoch", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--shortcut_name", default="bert-base-uncased", help="Backbone checkpoint name.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum serialized table token length.")
    parser.add_argument("--max_cols", type=int, default=16, help="Maximum number of columns per table.")
    parser.add_argument(
        "--adaptive_col_budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adjust per-column token budget based on the number of columns in a table.",
    )
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion steps.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for non-encoder modules.")
    parser.add_argument("--encoder_lr", type=float, default=2e-5, help="Learning rate for the encoder backbone.")
    parser.add_argument("--l2_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "adagrad", "rmsprop"])
    parser.add_argument("--diffuser_type", default="mlp1", choices=["mlp1", "mlp2"])
    parser.add_argument("--split_mode", default="auto", choices=["auto", "random", "cv", "presplit"])
    parser.add_argument("--cv_fold", type=int, default=-1, help="-1 runs all CV folds when CV files exist.")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="Validation ratio for split creation.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio for random split mode.")
    parser.add_argument("--eval_every", type=int, default=5, help="Validation interval in epochs.")
    parser.add_argument("--fast_steps", type=int, default=100, help="Reverse diffusion steps for evaluation.")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Classifier-free guidance scale.")
    parser.add_argument("--mc_eval", type=int, default=10, help="Monte Carlo repeats at evaluation.")
    parser.add_argument("--cond_drop_prob", type=float, default=0.2, help="Condition dropout probability.")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Beta schedule start.")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta schedule end.")
    parser.add_argument("--beta_schedule", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--y_dim", type=int, default=768, help="Label embedding dimension.")
    parser.add_argument("--cond_dim", type=int, default=768, help="Condition embedding dimension.")
    parser.add_argument("--time_dim", type=int, default=768, help="Time embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Denoiser hidden dimension.")
    parser.add_argument(
        "--freeze_encoder_epochs",
        type=int,
        default=3,
        help="Number of initial epochs with the encoder backbone frozen.",
    )
    return parser.parse_args()
