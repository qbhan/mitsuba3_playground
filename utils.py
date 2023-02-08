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


def save_aovs(aov_dict: dict, aovs, save_dir='result'):
    os.makedirs(save_dir, exist_ok=True)
    for k in aov_dict:
        # print(k, full_dict[k].shape)
        aov_name = k.split(':')[0]
        if len(aov_dict[k].shape) == 4: a = aov_dict[k].mean(2)
        elif len(aov_dict[k].shape) == 3: a = aov_dict[k] 
        # print(k, a.shape)
        if 'normal' in aov_name:
            plt.imsave(os.path.join(save_dir, '{}.png'.format(aov_name)), np.clip(a * 0.5 + 0.5, 0.0, 1.0))
        elif 'depth' in aov_name:
            a = np.clip(a, a_min=0.0, a_max=a.max())
            plt.imsave(os.path.join(save_dir, '{}.png'.format(aov_name)), a[:, :, 0], vmin=np.min(a), vmax=np.max(a))
        elif 'albedo' in aov_name:
            plt.imsave(os.path.join(save_dir, '{}.png'.format(aov_name)), np.clip(a, 0.0, 1.0))
        elif 'radiance' in aov_name:
            plt.imsave(os.path.join(save_dir, '{}.png'.format(aov_name)), tonemap(a))
        else:
            if a.shape[-1] == 1: plt.imsave(os.path.join(save_dir, '{}.png'.format(aov_name)), a[:, :, 0], vmin=np.min(a), vmax=np.max(a))
            elif a.shape[-1] == 3:
                plt.imsave(os.path.join(save_dir, '{}_x.png'.format(aov_name)), a, vmin=np.min(a), vmax=np.max(a))
            else:
                INDEX = ['x', 'y', 'z', 'w']
                for i in range(a.shape[-1]):
                    plt.imsave(os.path.join(save_dir, '{}_{}.png'.format(aov_name, INDEX[i])), a[:, :, i], vmin=np.min(a), vmax=np.max(a))

# def make_probmap(aov_dict: dict, clean_dict:dict, save_dir='result'):
#     gt = tonemap(clean_dict['radiance:3f'])
#     diffuse = aov_dict(aov_dict['radiance_diff:3f'])