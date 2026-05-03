import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.datasets import RAFIppCelebA
from Experiments.configs import get_args
from Model import RAFIppSystem
from Utils.metrics import summarize_restoration, dice_score, iou_score
from Utils.seed import seed_everything
from Utils.visualization import save_tensor_image


@torch.no_grad()
def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = RAFIppCelebA(args.data_root, split='test', image_size=args.image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    checkpoint = args.checkpoint or args.resume
    if not checkpoint:
        raise ValueError('Please provide --checkpoint for test/inference.')

    system = RAFIppSystem(args).to(device)
    system.load_checkpoint(checkpoint, map_location=device)
    system.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sums = {'l1': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'mask_dice': 0.0, 'mask_iou': 0.0}
    count = 0

    for batch in tqdm(loader, desc='Testing'):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = system.inference(batch)
        metrics = summarize_restoration(outputs['isyn'], batch['gt'])
        sums['l1'] += metrics['l1']
        sums['psnr'] += metrics['psnr']
        sums['ssim'] += metrics['ssim']
        sums['mask_dice'] += float(dice_score(outputs['mask_pred'], batch['mask']).item())
        sums['mask_iou'] += float(iou_score(outputs['mask_pred'], batch['mask']).item())
        count += 1

        for i, name in enumerate(batch['name']):
            stem = Path(name).stem
            save_tensor_image(batch['masked'][i], save_dir / f'{stem}_input.png')
            save_tensor_image(batch['mask'][i], save_dir / f'{stem}_mask_gt.png', is_mask=True)
            save_tensor_image(outputs['mask_pred'][i], save_dir / f'{stem}_mask_pred.png', is_mask=True)
            save_tensor_image(outputs['restored'][i], save_dir / f'{stem}_restored.png')
            save_tensor_image(outputs['isyn'][i], save_dir / f'{stem}_isyn.png')
            save_tensor_image(batch['gt'][i], save_dir / f'{stem}_gt.png')

    final_metrics = {k: v / max(count, 1) for k, v in sums.items()}
    (save_dir / 'metrics.json').write_text(json.dumps(final_metrics, indent=2), encoding='utf-8')
    print(json.dumps(final_metrics, indent=2))


if __name__ == '__main__':
    main()
