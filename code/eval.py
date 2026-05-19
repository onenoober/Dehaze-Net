import os
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils import AverageMeter, pad_img, val_psnr, val_ssim
from data import ValDataset
from data.data_loader import resolve_pair_dirs
from option import opt
from model import Backbone


def eval(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for i, batch in enumerate(tqdm(val_loader, desc='evaluation')):
        if opt.max_test_batches > 0 and i >= opt.max_test_batches:
            break
        hazy_img = batch['hazy'].cuda()
        clear_img = batch['clear'].cuda()

        with torch.no_grad():
            H, W = hazy_img.shape[2:]
            hazy_img = pad_img(hazy_img, 4)
            output = network(hazy_img)
            output = output.clamp(0, 1)
            output = output[:, :, :H, :W]
            if opt.save_infer_results:
                save_image(output, os.path.join(opt.saved_infer_dir, batch['filename'][0]))

        psnr_tmp = val_psnr(output, clear_img)
        ssim_tmp = val_ssim(output, clear_img).item()
        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)

    return PSNR.avg, SSIM.avg


if __name__ == '__main__':
    network = Backbone().cuda() 

    hazy_dir, clear_dir = resolve_pair_dirs(opt.val_dataset_dir)
    print('val_hazy_dir:', hazy_dir)
    print('val_clear_dir:', clear_dir)
    val_dataset = ValDataset(hazy_dir, clear_dir)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=False)

    # load pre-trained model
    ckpt = torch.load(os.path.join('../trained_models', opt.dataset, opt.pre_trained_model), map_location='cpu')
    network.load_state_dict(ckpt)

    # start evaluation
    avg_psnr, avg_ssim = eval(val_loader, network) 
    print('Evaluation on {}\nPSNR:{}\nSSIM:{}'.format(opt.dataset, avg_psnr, avg_ssim))
