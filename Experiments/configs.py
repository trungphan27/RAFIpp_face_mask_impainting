import argparse
from pathlib import Path
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='RAFI++ training and inference configuration')

    # Data
    parser.add_argument('--data_root', type=str, default='./Dataset/CelebA/rafipp')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # Runtime
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', action='store_true')

    # Optimization
    parser.add_argument('--lr_seg', type=float, default=2e-4)
    parser.add_argument('--lr_gen', type=float, default=2e-4)
    parser.add_argument('--lr_disc', type=float, default=2e-4)
    parser.add_argument('--betas', type=float, nargs=2, default=(0.5, 0.999))
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=0.0)

    # Schedule
    parser.add_argument('--stage1_epochs', type=int, default=10)
    parser.add_argument('--stage2_epochs', type=int, default=20)
    parser.add_argument('--stage3_epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)

    # Loss weights from the RAFI++ document
    parser.add_argument('--lambda_m', type=float, default=1.0)
    parser.add_argument('--lambda_b', type=float, default=1.0)
    parser.add_argument('--lambda_c', type=float, default=0.2)
    parser.add_argument('--lambda_dice', type=float, default=1.0)
    parser.add_argument('--lambda_bdice', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=10.0)
    parser.add_argument('--lambda_ssim', type=float, default=5.0)
    parser.add_argument('--lambda_perc', type=float, default=1.0)
    parser.add_argument('--lambda_style', type=float, default=100.0)
    parser.add_argument('--lambda_id', type=float, default=2.0)
    parser.add_argument('--lambda_edge', type=float, default=2.0)
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--alpha_region', type=float, default=3.0)
    parser.add_argument('--gamma_region', type=float, default=2.0)
    parser.add_argument('--beta_conf', type=float, default=5.0)

    # I/O
    parser.add_argument('--run_name', type=str, default='rafipp_run')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--sample_dir', type=str, default='./outputs/samples')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='./outputs/test_predictions')

    return parser


def get_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    return args


def save_args(args, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False, allow_unicode=True)
