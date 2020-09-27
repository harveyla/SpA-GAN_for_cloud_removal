import numpy as np
from skimage.measure import compare_ssim as SSIM

from torch.autograd import Variable

from utils import save_image


def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    for i, batch in enumerate(test_data_loader):
        if i > 100: break
        real_cloud, real_sar, real_clean, real_mask = Variable(batch[0][0]), Variable(batch[0][1]), Variable(batch[1][1]), Variable(batch[1][2])
        if config.cuda:
            real_cloud = real_cloud.cuda()
            real_sar = real_sar.cuda()
            real_clean = real_clean.cuda()
            real_mask = real_mask.cuda()

        att, out = gen((real_cloud, real_sar))

        if epoch % config.snapshot_interval == 0:
            h = 1
            w = 3
            c = 3
            width = config.width
            height = config.height

            allim = np.zeros((h, w, c, width, height))
            real_cloud_ = real_cloud.cpu().numpy()[0]
            real_clean_ = real_clean.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = np.clip(real_cloud_[1:4], 0, 1)
            t_rgb = np.clip(real_clean_[1:4], 0, 1)
            out_rgb = np.clip(out_[1:4], 0, 1)
            allim[0, 0, :] = in_rgb * 255
            allim[0, 1, :] = out_rgb * 255
            allim[0, 2, :] = t_rgb * 255
            
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*height, w*width, c))

            save_image(config.out_dir, allim, i, epoch)

        mse = criterionMSE(out, real_clean)
        psnr = 10 * np.log10(1 / mse.item())

        img1 = np.tensordot(out.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        img2 = np.tensordot(real_clean.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        
        ssim = SSIM(img1, img2)
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim
    
    # avg_mse = avg_mse / len(test_data_loader)
    # avg_psnr = avg_psnr / len(test_data_loader)
    # avg_ssim = avg_ssim / len(test_data_loader)
    avg_mse = avg_mse / (i+1)
    avg_psnr = avg_psnr / (i+1)
    avg_ssim = avg_ssim / (i+1)

    print("===> Avg. MSE: {:.4f}".format(avg_mse))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))
    
    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test
