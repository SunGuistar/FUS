import random
import json,os

import h5py
import numpy as np


class MultiFileSSFrameDataset():  # Multi-file Subject-Scan frame loader
    """
    A dataset class that can handle combining data from multiple H5 files.
    This class maintains a list of individual SSFrameDataset instances and 
    provides a unified interface for accessing data across multiple files.
    """

    def __init__(self, datasets):
        """
        :param datasets: List of SSFrameDataset instances to combine
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        
        self.datasets = datasets
        self.dataset_offsets = []
        
        # Calculate cumulative offsets for each dataset
        total_indices = 0
        for dataset in datasets:
            self.dataset_offsets.append(total_indices)
            total_indices += len(dataset)
        
        self.num_indices = total_indices
        
        # Use parameters from the first dataset as defaults
        self.min_scan_len = datasets[0].min_scan_len
        self.num_samples = datasets[0].num_samples
        self.sample_range = datasets[0].sample_range
        self.data_path = datasets[0].data_path
        
        # Check for parameter consistency and warn if different
        for i, dataset in enumerate(datasets[1:], 1):
            if dataset.num_samples != self.num_samples:
                print(f'WARNING: Dataset {i} has different num_samples ({dataset.num_samples} vs {self.num_samples}) - using first dataset\'s value.')
            if dataset.sample_range != self.sample_range:
                print(f'WARNING: Dataset {i} has different sample_range ({dataset.sample_range} vs {self.sample_range}) - using first dataset\'s value.')

    def __len__(self):
        return self.num_indices

    def __getitem__(self, idx):
        """
        Get item by finding which dataset contains the requested index
        and delegating to that dataset.
        """
        if idx >= self.num_indices:
            raise IndexError("Index out of range")
        
        # Find which dataset contains this index
        dataset_idx = 0
        for i, offset in enumerate(self.dataset_offsets):
            if i + 1 < len(self.dataset_offsets):
                if offset <= idx < self.dataset_offsets[i + 1]:
                    dataset_idx = i
                    break
            else:  # Last dataset
                dataset_idx = i
        
        # Calculate local index within the target dataset
        local_idx = idx - self.dataset_offsets[dataset_idx]
        
        # Delegate to the appropriate dataset
        return self.datasets[dataset_idx][local_idx]

    def __add__(self, other):
        """
        Combine with another dataset (can be MultiFileSSFrameDataset or SSFrameDataset)
        """
        if isinstance(other, MultiFileSSFrameDataset):
            combined_datasets = self.datasets + other.datasets
        else:  # Assume it's SSFrameDataset
            combined_datasets = self.datasets + [other]
        
        return MultiFileSSFrameDataset(combined_datasets)


class SSFrameDataset():  # Subject-Scan frame loader

    def __init__(self, min_scan_len, data_path,h5_file_name, indices_in_use=None, num_samples=2, sample_range=None):

        """
        :param filename_h5, file path
        :param indices_in_use: 
            case 1: a list of tuples (idx_subject, idx_scans), indexing self.num_frames[indices_in_use[idx]]
            case 2: a list of two lists, [indices_subjects] and [indices_scans], meshgrid to get indices
            case 3: None (default), use all available in the file
        
        Sampling parameters
        :param num_samples: type int, number of (model input) frames, > 1. However, when num_samples=-1, sample all in the scan
        :param sample_range: type int, range of sampling frames, default is num_samples
        """
        self.min_scan_len = min_scan_len
        self.data_path=data_path
        self.h5_file_name=h5_file_name
        self.filename = data_path+'/'+h5_file_name
        self.file = h5py.File(self.filename, 'r')
        self.frame_size = self.file['frame_size'][()]
        self.num_frames = self.file['num_frames'][()]
        self.name_scan = self.file['name_scan'][()]
        
        if indices_in_use is None:
            self.indices_in_use = [(i_sub,i_scn) for i_sub in range(self.num_frames.shape[0]) for i_scn in range(self.num_frames.shape[1])]                    
            # delete indices whose scan length is less than min_scan_len
            self.indices_in_use = list(tuple(map(tuple,(np.array((np.where(self.num_frames >= self.min_scan_len))).T).tolist())))

        elif all([isinstance(t,tuple) for t in indices_in_use]):
            self.indices_in_use = indices_in_use
        elif isinstance(indices_in_use[0],list) and isinstance(indices_in_use[1],list):
            self.indices_in_use = [(i_sub,i_scn) for i_sub in indices_in_use[0] for i_scn in indices_in_use[1]]            
        else:
            raise ValueError("indices_in_use should be a list of tuples (idx_subject, idx_scans) of two lists, [indices_subjects] and [indices_scans].")
        
        if len(set(self.indices_in_use)) != len(self.indices_in_use):
            print("WARNING: Replicated indices are found - not removed.")
        
        self.indices_in_use.sort()
        self.num_indices = len(self.indices_in_use)

        # sampling parameters
        if num_samples < 2:
            if num_samples == -1:
                if sample_range is not None:
                    sample_range = None
                    print("Sampling all frames. sample_range is ignored.")
            else:
                raise ValueError('num_samples should be greater than or equal to 2, or -1 for sampling all frames.')
        self.num_samples = num_samples
        
        if sample_range is None:
            self.sample_range = self.num_samples
        elif any([self.num_frames[indices]<sample_range for indices in self.indices_in_use]):
            raise ValueError("The specified sample_range is larger than number of frames in at least one of the in-use scans.")
        else:
            self.sample_range = sample_range
        

    def __add__(self, other):
        """
        Enhanced __add__ method that supports multi-file combining.
        Returns a MultiFileSSFrameDataset when combining different files.
        """
        # Check if files are the same (original behavior)
        if self.filename == other.filename:
            # Same file - use original combining logic
            if self.num_samples != other.num_samples:
                print('WARNING: found different num_samples - the first is used.')
            if self.sample_range != other.sample_range:
                print('WARNING: found different sample_range - the first is used.')
            indices_combined = self.indices_in_use + other.indices_in_use
            return SSFrameDataset(min_scan_len = self.min_scan_len,data_path = self.data_path, h5_file_name=self.h5_file_name, indices_in_use=indices_combined, num_samples=self.num_samples, sample_range=self.sample_range)
        else:
            # Different files - create MultiFileSSFrameDataset
            print(f'INFO: Combining datasets from different files: {self.h5_file_name} + {other.h5_file_name}')
            return MultiFileSSFrameDataset([self, other])
    

    def __len__(self):
        return self.num_indices
    

    def __getitem__(self, idx):
        

        indices = self.indices_in_use[idx]
        if self.num_samples == -1:  # sample all available frames, for validation
            i_frames = range(self.num_frames[indices])
        else:
            i_frames = self.frame_sampler(self.num_frames[indices])
 
        frames = self.file['/sub{:03d}_frames{:02d}'.format(indices[0],indices[1])][()][i_frames,...]
        tforms = self.file['/sub{:03d}_tforms{:02d}'.format(indices[0],indices[1])][()][i_frames,...]
        tforms_inv = self.file['/sub{:03d}_tforms_inv{:02d}'.format(indices[0],indices[1])][()][i_frames,...]

        return frames,tforms,tforms_inv



    def frame_sampler(self, n):

        n0 = random.randint(0,n-self.sample_range)  # sample the start index for the range
        idx_frames = random.sample(range(n0,n0+self.sample_range), self.num_samples)   # sample indices
        idx_frames.sort()
        return idx_frames
    

    def write_json(self, jason_filename):
        with open(jason_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "min_scan_len" : self.min_scan_len,
                "filename" : self.filename, 
                "indices_in_use" : self.indices_in_use,
                "num_samples": self.num_samples,
                "sample_range": self.sample_range
                }, f, ensure_ascii=False, indent=4)
        print("%s written." % jason_filename)
    
    
    @staticmethod
    def read_json(data_path,jason_filename, h5_file_name, num_samples=None):
        with open(data_path+'/'+jason_filename, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            return SSFrameDataset(
                min_scan_len = obj['min_scan_len'],
                data_path=data_path,
                h5_file_name=h5_file_name,
                indices_in_use = [tuple(ids) for ids in obj['indices_in_use']], # convert to tuples from json string
                num_samples    = obj['num_samples'] if num_samples is None else num_samples,
                sample_range   = obj['sample_range'],
                )
   
   

   
   
        
            
