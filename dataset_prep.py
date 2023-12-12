import numpy as np
import glob

def main ():
    for dstype in ['train', 'test']:

        print(f'Processing {dstype} data...')
        dstype_paths = glob.glob(f'./processed_data/{dstype}/*.npz')

        mix_mags_dstype = []
        mix_phases_dstype = []
        vocal_mags_dstype = []

        for i, path in enumerate(dstype_paths):
            print(f'Processing {i+1}/{len(dstype_paths)}')
            data = np.load(path, mmap_mode='r')
            mix_mags_dstype = mix_mags_dstype + [data['mix_mag'][:,:,:,np.newaxis]]
            mix_phases_dstype = mix_phases_dstype + [data['mix_phase'][:,:,:,np.newaxis]]
            vocal_mags_dstype = vocal_mags_dstype + [data['vocal_mag'][:,:,:,np.newaxis]]
            
        mix_mags_dstype = np.concatenate(mix_mags_dstype, axis=0)
        mix_phases_dstype = np.concatenate(mix_phases_dstype, axis=0)
        vocal_mags_dstype = np.concatenate(vocal_mags_dstype, axis=0)
        
        vocal_masks_dstype = np.copy(vocal_mags_dstype)

        np.divide(vocal_mags_dstype, mix_mags_dstype, out=vocal_masks_dstype, where=mix_mags_dstype!=0)
        
        print('Saving data...')
        # save the data to npy files
        np.save(f'./processed_data/mix_mags_{dstype}_512x128.npy', mix_mags_dstype)
        np.save(f'./processed_data/mix_phases_{dstype}_512x128.npy', mix_phases_dstype)
        np.save(f'./processed_data/vocal_mags_{dstype}_512x128.npy', vocal_mags_dstype)
        np.save(f'./processed_data/vocal_masks_{dstype}_512x128.npy', vocal_masks_dstype)
        
if __name__ == '__main__':
    main()