import os
import pickle
from pathlib import Path
import hashlib
import numpy as np
import torch
import torch.utils.data as data

import librosa
import scipy.interpolate
from scipy.signal import decimate
import re


def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  f = scipy.interpolate.splrep(i_lr, x_lr)
  x_sp = scipy.interpolate.splev(i_hr, f)
  return x_sp

# Adapted from https://github.com/kuleshov/audio-super-res/blob/master/data/vctk/prep_vctk.py

class VCTK(data.TensorDataset):

    def __init__(self, file_list, input_dir, scale_factor, sampling_rate,
                 clip_length, clip_stride, interpolate=True, low_pass=True):
        """
        Args:
            file_list: (string) path to file containing the list of file names
        """
        cache_str = f'{file_list}_{input_dir}_{scale_factor}_{sampling_rate}_{clip_length}_{clip_stride}_{interpolate}_{low_pass}'
        cache_hash = hashlib.sha256()
        cache_hash.update(cache_str.encode())
        cache_filename = Path(cache_hash.hexdigest() + '.pt')
        # cache_filename = cache_filename.replace('/', '-')
        cur_dir = Path(__file__).parent.absolute() / 'vctk'
        cur_dir.mkdir(exist_ok=True)
        cache_filename = cur_dir / cache_filename
        if cache_filename.is_file():
            lr_patches, hr_patches = torch.load(cache_filename)
        else:
            filename_list = []
            ID_list = []
            file_extensions = set(['.wav'])
            with open(file_list) as f:
                for line in f:
                    filename = line.strip()
                    ext = os.path.splitext(filename)[1]
                    if ext in file_extensions:
                        filename_list.append(os.path.join(input_dir, filename))

            num_files = len(filename_list)
            # patches to extract and their size
            if clip_length is not -1:
                if interpolate:
                    d, d_lr = clip_length, clip_length
                    s, s_lr = clip_stride, clip_stride
                else:
                    d, d_lr = clip_length, clip_length // scale_factor
                    s, s_lr = clip_stride, clip_stride // scale_factor
            hr_patches, lr_patches = list(), list()

            for j, file_path in enumerate(filename_list):
                if j % 10 == 0: print('%d/%d' % (j, num_files))
                ID = int(re.search('p\d\d\d/', file_path).group(0)[1:-1])

                # load audio file
                x, fs = librosa.load(file_path, sr=sampling_rate)

                # crop so that it works with scaling ratio
                x_len = len(x)
                x = x[ : x_len - (x_len % scale_factor)]

                # generate low-res version
                if low_pass:
                    x_lr = decimate(x, scale_factor)
                else:
                    x_lr = np.array(x[0::scale_factor])

                if interpolate:
                    x_lr = upsample(x_lr, scale_factor)
                    assert len(x) % scale_factor == 0
                    assert len(x_lr) == len(x)
                else:
                    assert len(x) % scale_factor == 0
                    assert len(x_lr) == len(x) // scale_factor

                if clip_length is not -1:
                    # generate patches
                    max_i = len(x) - d + 1
                    for i in range(0, max_i, s):
                        # # keep only a fraction of all the patches
                        # u = np.random.uniform()
                        # if u > args.sam: continue
                        if interpolate:
                            i_lr = i
                        else:
                            i_lr = i // scale_factor

                        hr_patch = np.array( x[i : i+d] )
                        lr_patch = np.array( x_lr[i_lr : i_lr+d_lr] )

                        assert len(hr_patch) == d
                        assert len(lr_patch) == d_lr

                        hr_patches.append(hr_patch.reshape((d,1)))
                        lr_patches.append(lr_patch.reshape((d_lr,1)))
                        ID_list.append(ID)
                else: # for full snr
                    # append the entire file without patching
                    x = x[:,np.newaxis]
                    x_lr = x_lr[:,np.newaxis]
                    hr_patches.append(x[:len(x) // 256 * 256])
                    lr_patches.append(x_lr[:len(x_lr) // 256 * 256])
                    ID_list.append(ID)

            # if clip_length is not -1:
            #     # crop # of patches so that it's a multiple of mini-batch size
            #     num_patches = len(hr_patches)
            #     num_to_keep = int(np.floor(num_patches / args.batch_size) * args.batch_size)
            #     hr_patches = np.array(hr_patches[:num_to_keep])
            #     lr_patches = np.array(lr_patches[:num_to_keep])
            #     ID_list = ID_list[:num_to_keep]
            # if save_examples:
            #     librosa.output.write_wav('example-hr.wav', hr_patches[40][0], fs, norm=False)
            #     librosa.output.write_wav('example-lr.wav', lr_patches[40][0], fs // scale_factor, norm=False)


            if clip_length is not -1:
                hr_patches = torch.Tensor(hr_patches)
                lr_patches = torch.Tensor(lr_patches)
                # # create the hdf5 file
                # data_set = h5_file.create_dataset('data', lr_patches.shape, np.float32)
                # label_set = h5_file.create_dataset('label', hr_patches.shape, np.float32)

                # data_set[...] = lr_patches
                # label_set[...] = hr_patches
                # pickle.dump(ID_list, open('ID_list_patches_'+str(d)+'_'+str(scale_factor), 'w'))
            else:
                assert False, 'clip_length=-1 not supported yet'
                # # pickle the data
                # pickle.dump(hr_patches, open('full-label-'+args.out[:-7],'w'))
                # pickle.dump(lr_patches, open('full-data-'+args.out[:-7],'w'))
                # pickle.dump(ID_list, open('ID_list','w'))

            if not interpolate:
                hr_patches = hr_patches.reshape(hr_patches.shape[0], hr_patches.shape[1] // scale_factor, scale_factor)
            print(f'Caching dataset to {cache_filename}')
            torch.save((lr_patches, hr_patches), cache_filename)

        super().__init__(lr_patches, hr_patches)
