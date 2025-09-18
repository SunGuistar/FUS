
import random
import json,os
import glob

import h5py
import numpy as np


class SSFrameDataset():  # Subject-Scan frame loader

    def __init__(self, min_scan_len, data_path, h5_file_name, indices_in_use=None, num_samples=2, sample_range=None, split_type='train'):

        """
        :param filename_h5, file path
        :param indices_in_use: 
            case 1: a list of tuples (idx_subject, idx_scans), indexing self.num_frames[indices_in_use[idx]]
            case 2: a list of two lists, [indices_subjects] and [indices_scans], meshgrid to get indices
            case 3: None (default), use all available in the file
        
        Sampling parameters
        :param num_samples: type int, number of (model input) frames, > 1. However, when num_samples=-1, sample all in the scan
        :param sample_range: type int, range of sampling frames, default is num_samples
        :param split_type: type str, 'train', 'test', or 'val' to determine which directory to read from
        """
        self.min_scan_len = min_scan_len
        self.data_path = data_path
        self.h5_file_name = h5_file_name
        self.split_type = split_type
        
        # New approach: read from train/test/val directories
        if split_type in ['train', 'test', 'val']:
            self.h5_files = self._get_h5_files_from_split(split_type)
            self.num_scans = len(self.h5_files)
            # Initialize with first file to get frame_size
            if self.num_scans > 0:
                with h5py.File(self.h5_files[0], 'r') as f:
                    self.frame_size = f['frames'].shape[1:]  # Get height and width
            else:
                raise ValueError(f"No h5 files found in {split_type} directory")
        else:
            # Fallback to original single file approach
            self.filename = data_path + '/' + h5_file_name
            self.file = h5py.File(self.filename, 'r')
            self.frame_size = self.file['frame_size'][()]
            self.num_frames = self.file['num_frames'][()]
            self.name_scan = self.file['name_scan'][()]
            self.h5_files = [self.filename]
            self.num_scans = 1
        
        # Handle indices_in_use for new multi-file approach
        if split_type in ['train', 'test', 'val']:
            if indices_in_use is None:
                # Use all available h5 files
                self.indices_in_use = list(range(self.num_scans))
            elif all([isinstance(t, tuple) for t in indices_in_use]):
                # Convert old format to new format (just use the scan indices)
                self.indices_in_use = [idx[1] for idx in indices_in_use if idx[1] < self.num_scans]
            elif isinstance(indices_in_use, list) and all([isinstance(idx, int) for idx in indices_in_use]):
                # Direct list of scan indices
                self.indices_in_use = [idx for idx in indices_in_use if idx < self.num_scans]
            else:
                raise ValueError("indices_in_use should be a list of integers (scan indices) for the new multi-file approach")
        else:
            # Original logic for single file approach
            if indices_in_use is None:
                self.indices_in_use = [(i_sub,i_scn) for i_sub in range(self.num_frames.shape[0]) for i_scn in range(self.num_frames.shape[1])]                    
                # delete indices whose scan length is less than min_scan_len
                self.indices_in_use = list(tuple(map(tuple,(np.array((np.where(self.num_frames >= self.min_scan_len))).T).tolist())))

            elif all([isinstance(t,tuple) for t in indices_in_use]):
                self.indices_in_use = indices_in_use
            elif isinstance(indices_in_use[0],list) and isinstance(indices_in_use[1],list):
                self.indices_in_use = [(i_sub,i_scn) for i_sub in indices_in_use[0] for i_scn in indices_in_use[1]]            
            else:
                raise("indices_in_use should be a list of tuples (idx_subject, idx_scans) of two lists, [indices_subjects] and [indices_scans].")
        
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
                raise('num_samples should be greater than or equal to 2, or -1 for sampling all frames.')
        self.num_samples = num_samples
        
        if sample_range is None:
            self.sample_range = self.num_samples
        else:
            # For new approach, we'll check frame counts dynamically
            if split_type not in ['train', 'test', 'val']:
                if any([self.num_frames[indices]<sample_range for indices in self.indices_in_use]):
                    raise("The specified sample_range is larger than number of frames in at least one of the in-use scans.")
        self.sample_range = sample_range

    def _get_h5_files_from_split(self, split_type):
        """Get all h5 files from the specified split directory"""
        split_dir = os.path.join(self.data_path, split_type)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Get all h5 files recursively from the split directory
        h5_files = []
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
        
        h5_files.sort()  # Ensure consistent ordering
        return h5_files

    def _get_num_frames_for_scan(self, scan_idx):
        """Get number of frames for a specific scan"""
        if scan_idx >= len(self.h5_files):
            raise IndexError(f"Scan index {scan_idx} out of range")
        
        with h5py.File(self.h5_files[scan_idx], 'r') as f:
            return f['frames'].shape[0]

    def _get_name_scan_for_scan(self, scan_idx):
        """Get name_scan for a specific scan (use filename)"""
        if scan_idx >= len(self.h5_files):
            raise IndexError(f"Scan index {scan_idx} out of range")
        
        # Use the filename (without path and extension) as name_scan
        filename = os.path.basename(self.h5_files[scan_idx])
        return filename.replace('.h5', '')
        

    def __add__(self, other):
        # Check if both datasets use the same approach
        if hasattr(self, 'split_type') and hasattr(other, 'split_type'):
            if self.split_type != other.split_type:
                raise ValueError(f"Cannot combine datasets with different split_types: {self.split_type} and {other.split_type}")
            if self.data_path != other.data_path:
                raise ValueError(f"Cannot combine datasets with different data_paths: {self.data_path} and {other.data_path}")
        elif hasattr(self, 'filename') and hasattr(other, 'filename'):
            if self.filename != other.filename:
                raise ValueError('Currently different file combining is not supported.')
        else:
            raise ValueError('Cannot combine datasets with different data loading approaches')
        
        if self.num_samples != other.num_samples:
            print('WARNING: found different num_samples - the first is used.')
        if self.sample_range != other.sample_range:
            print('WARNING: found different sample_range - the first is used.')
        
        indices_combined = self.indices_in_use + other.indices_in_use
        
        # Create new dataset with combined indices
        if hasattr(self, 'split_type') and self.split_type in ['train', 'test', 'val']:
            return SSFrameDataset(
                min_scan_len=self.min_scan_len,
                data_path=self.data_path, 
                h5_file_name=self.h5_file_name, 
                indices_in_use=indices_combined, 
                num_samples=self.num_samples, 
                sample_range=self.sample_range,
                split_type=self.split_type
            )
        else:
            return SSFrameDataset(
                min_scan_len=self.min_scan_len,
                data_path=self.data_path, 
                h5_file_name=self.h5_file_name, 
                indices_in_use=indices_combined, 
                num_samples=self.num_samples, 
                sample_range=self.sample_range
            )
    

    def __len__(self):
        return self.num_indices
    

    def __getitem__(self, idx):
        if self.split_type in ['train', 'test', 'val']:
            # New multi-file approach
            scan_idx = self.indices_in_use[idx]
            h5_file_path = self.h5_files[scan_idx]
            
            with h5py.File(h5_file_path, 'r') as f:
                num_frames = f['frames'].shape[0]
                
                if self.num_samples == -1:  # sample all available frames, for validation
                    i_frames = range(num_frames)
                else:
                    i_frames = self.frame_sampler(num_frames)
                
                frames = f['frames'][()][i_frames, ...]
                tforms = f['tforms'][()][i_frames, ...]
                tforms_inv = f['tforms_inv'][()][i_frames, ...]
            
            return frames, tforms, tforms_inv
        else:
            # Original single file approach
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
        if n < self.sample_range:
            # If scan has fewer frames than sample_range, use all frames
            if n < self.num_samples:
                # If scan has fewer frames than num_samples, use all frames
                return list(range(n))
            else:
                # Sample num_samples frames from available frames
                idx_frames = random.sample(range(n), self.num_samples)
                idx_frames.sort()
                return idx_frames
        else:
            n0 = random.randint(0, n-self.sample_range)  # sample the start index for the range
            idx_frames = random.sample(range(n0, n0+self.sample_range), self.num_samples)   # sample indices
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
    def read_json(data_path, jason_filename, h5_file_name, num_samples=None, options=None, split_type='train'):
        with open(data_path+'/'+jason_filename, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            
            # Get default values from train_options.py if available
            if options is not None:
                default_min_scan_len = getattr(options, 'MIN_SCAN_LEN', 108)
                default_num_samples = getattr(options, 'NUM_SAMPLES', 100)
                default_sample_range = getattr(options, 'SAMPLE_RANGE', 100)
            else:
                # Fallback defaults if no options provided
                default_min_scan_len = 108
                default_num_samples = 100
                default_sample_range = 100
            
            # Use JSON values if available, otherwise use defaults from train_options.py
            min_scan_len = obj.get('min_scan_len', default_min_scan_len)
            num_samples_val = obj.get('num_samples', default_num_samples) if num_samples is None else num_samples
            sample_range = obj.get('sample_range', default_sample_range)
            
            return SSFrameDataset(
                min_scan_len = min_scan_len,
                data_path=data_path,
                h5_file_name=h5_file_name,
                indices_in_use = [tuple(ids) for ids in obj['indices_in_use']], # convert to tuples from json string
                num_samples    = num_samples_val,
                sample_range   = sample_range,
                split_type     = split_type,
                )
   
   

   
   
        
            