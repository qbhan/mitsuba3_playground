import numpy as np
import matplotlib.pyplot as plt
import os


def tonemap(c, ref=None, kInvGamma=1.0/2.2):
    # c: (W, H, C=3)
    if ref is None:
        ref = c
    luminance = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    col = np.copy(c)
    col[:,:,0] /= (1 + luminance / 1.5)
    col[:,:,1] /= (1 + luminance / 1.5)
    col[:,:,2] /= (1 + luminance / 1.5)
    col = np.clip(col, 0, None)
    return np.clip(col ** kInvGamma, 0.0, 1.0)


def save_aovs(img: np.ndarray, aovs, save_dir='result'):
    rgb = img[:, :, :3]
    os.makedirs(save_dir, exist_ok=True)
    plt.imsave(os.path.join(save_dir, 'rgb.png'), tonemap(rgb))
    idx = 3
    for aov in aovs:
        aov_name, aov_ch = aov.split(':')[0], int(aov.split(':')[1])
        print(aov_name)
        a = img[:, :, idx:idx+aov_ch]
        idx += aov_ch
        if 'normal' in aov_name:
            plt.imsave(os.path.join(save_dir, '{}.png'.format(aov_name)), np.clip(a * 0.5 + 0.5, 0.0, 1.0))
        elif 'depth' in aov_name:
            plt.imsave(os.path.join(save_dir, '{}.png'.format(aov_name)), a[:, :, 0], vmin=np.min(a), vmax=np.max(a))
        elif 'albedo' in aov_name:
            plt.imsave(os.path.join(save_dir, '{}.png'.format(aov_name)), np.clip(a, 0.0, 1.0))