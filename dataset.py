import math
import torch
from torch.utils.data import Dataset
import numpy as np

class DistDataset(Dataset):
    def __init__(self, data_path, split='train', split_ratio=0.7, seed=42):
        print(f"Loading preprocessed distribution data from {data_path}...")
        loaded = torch.load(data_path, weights_only=False)
        self._rng = np.random.RandomState(seed)

        
        all_data = []
        
        
        if 'val_data' in loaded and 'test_data' in loaded:
            print(f"Detected pre-split data (Train/Val/Test). Split: {split}")
            self.hvg_indices = loaded['hvg_indices']
            
            
            
            
            
            
            full_data = loaded['train_data'] + loaded['val_data'] + loaded['test_data']
            self.control_batches = [d for d in full_data if d['is_control']]
            
            if split == 'train':
                source_data = loaded['train_data']
            elif split == 'val':
                source_data = loaded['val_data']
            elif split == 'test':
                source_data = loaded['test_data']
            else:
                raise ValueError(f"Unknown split: {split}")
                
            
            self.perturb_data = [d for d in source_data if not d['is_control']]
            
            print(f"Selected {len(self.perturb_data)} batches for {split.upper()}")
            
        elif 'train_data' in loaded:
            
            all_data.extend(loaded['train_data'])
            all_data.extend(loaded.get('test_data', []))
            
            self.hvg_indices = loaded['hvg_indices']
            
            
            all_control = [d for d in all_data if d['is_control']]
            all_perturb = [d for d in all_data if not d['is_control']]
            
            
            unique_labels = sorted(list(set(d['label'] for d in all_perturb)))
            
            
            rng = np.random.RandomState(seed)
            rng.shuffle(unique_labels)
            
            n_train = int(len(unique_labels) * split_ratio)
            train_labels = set(unique_labels[:n_train])
            test_labels = set(unique_labels[n_train:])
            
            print(f"Total Perturbation Labels: {len(unique_labels)}")
            print(f"Split: {len(train_labels)} Train Labels, {len(test_labels)} Test Labels")
            
            
            if split == 'train':
                self.perturb_data = [d for d in all_perturb if d['label'] in train_labels]
                print(f"Selected {len(self.perturb_data)} batches for TRAINING (Labels: {list(train_labels)[:5]}...)")
            else:
                self.perturb_data = [d for d in all_perturb if d['label'] in test_labels]
                print(f"Selected {len(self.perturb_data)} batches for TESTING (Labels: {list(test_labels)[:5]}...)")
                
            
            self.control_batches = all_control
            
        else:
            
            print("WARNING: Old data format detected.")
            all_data = loaded['data']

        
        
        self.perturb_mode = loaded.get('perturb_mode', 'gene')
        
        self.truncated = loaded.get('truncated', False)

        
        if self.perturb_mode == 'drug':
            self.smiles_to_fp = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in loaded.get('smiles_to_fp', {}).items()
            }
            self.fp_dim = loaded.get('fp_dim', 1024)
            self.cell_types = loaded.get('cell_types', [])
            self.cell_type_to_idx = {ct: i for i, ct in enumerate(self.cell_types)}
            ctrl_key = 'cell_type'
        else:
            self.smiles_to_fp = None
            self.fp_dim = None
            self.cell_types = None
            self.cell_type_to_idx = None
            ctrl_key = 'cell_line'

        
        
        
        
        self.factor_loadings = loaded.get('factor_loadings')
        self.copula_ctrl_mean = loaded.get('copula_ctrl_mean')
        self.copula_ctrl_std = loaded.get('copula_ctrl_std')
        self.ctrl_psi = loaded.get('ctrl_psi')  
        self.copula_rank = int(loaded.get('copula_rank', 0)) or 0
        self.has_copula = (self.factor_loadings is not None and self.copula_rank > 0)
        if self.has_copula:
            psi_info = f", psi=({self.ctrl_psi.shape[0]},)" if self.ctrl_psi is not None else ", psi=None"
            print(f"Copula enabled: A={tuple(self.factor_loadings.shape)}, r={self.copula_rank}{psi_info}")

        
        self.cell_line_vocab = loaded.get('cell_line_vocab', {})
        self.n_cell_types = int(loaded.get('n_cell_types', 0) or 0)

        
        self.perturb_batches = {} 
        for item in self.perturb_data:
            lbl = item['label']
            if lbl not in self.perturb_batches: self.perturb_batches[lbl] = []
            self.perturb_batches[lbl].append(item)

        
        self.perturb_samples = []
        for lbl, batches in self.perturb_batches.items():
            for b in batches:
                if self.perturb_mode == 'drug':
                    
                    self.perturb_samples.append(
                        (b['params'], lbl, b.get('cell_type', 'unknown'),
                         b.get('smiles'), b.get('dose_val', 0.0), b.get('factor_var'))
                    )
                else:
                    
                    self.perturb_samples.append(
                        (b['params'], lbl, b.get('cell_line', 'unknown'), b.get('factor_var'))
                    )

        print(f"Dataset ({split}, mode={self.perturb_mode}): "
              f"{len(self.control_batches)} control batches, {len(self.perturb_samples)} perturb batches.")

        
        self.control_by_cell = {}
        for cb in self.control_batches:
            cline = cb.get(ctrl_key, 'unknown')
            if cline not in self.control_by_cell:
                self.control_by_cell[cline] = []
            self.control_by_cell[cline].append(cb)

        print(f"Control batches by {ctrl_key}: { {k: len(v) for k, v in self.control_by_cell.items()} }")

        
        
        if self.has_copula:
            fvs = [cb.get('factor_var') for cb in self.control_batches if cb.get('factor_var') is not None]
            if fvs:
                self.factor_var_fallback = torch.stack(fvs).mean(dim=0).float()
            else:
                self.factor_var_fallback = torch.ones(self.copula_rank, dtype=torch.float32)
        else:
            self.factor_var_fallback = None
        
    def __len__(self):
        
        if len(self.perturb_samples) == 0: return 0
        return len(self.perturb_samples) 
    
    def __getitem__(self, idx):
        
        
        

        real_idx = idx % len(self.perturb_samples)

        if self.perturb_mode == 'drug':
            target_params, drug_dose_name, cell_type, smiles, dose_val, target_fv = self.perturb_samples[real_idx]
            ctrl_key = cell_type
        else:
            target_params, label, cell_line, target_fv = self.perturb_samples[real_idx]
            ctrl_key = cell_line

        
        available_controls = self.control_by_cell.get(ctrl_key)

        if available_controls and len(available_controls) > 0:
            
            ci = self._rng.randint(len(available_controls))
            ctrl_batch = available_controls[ci]
            ctrl_params = ctrl_batch['params']
            ctrl_fv = ctrl_batch.get('factor_var')
        else:
            
            if len(self.control_batches) > 0:
                 ctrl_batch = self.control_batches[np.random.randint(len(self.control_batches))]
                 ctrl_params = ctrl_batch['params']
                 ctrl_fv = ctrl_batch.get('factor_var')
            else:
                 ctrl_params = torch.zeros_like(target_params)
                 ctrl_fv = None

        if self.perturb_mode == 'drug':
            drug_fp = self.smiles_to_fp.get(smiles)
            if drug_fp is None:
                
                drug_fp = torch.zeros(self.fp_dim, dtype=torch.float32)
            ct_idx = self.cell_type_to_idx.get(cell_type, 0)
            
            
            
            
            dose_log10 = math.log10(max(dose_val, 1e-6))
            out = {
                'x_ctrl': ctrl_params,
                'drug_fp': drug_fp,
                'dose': torch.tensor(dose_log10, dtype=torch.float32),
                'cell_type': torch.tensor(ct_idx, dtype=torch.long),
                'y_target': target_params,
                'drug_dose_name': drug_dose_name,
            }
        else:
            
            context_id = self.cell_line_vocab.get(ctrl_key, 0) if self.cell_line_vocab else 0
            out = {
                'x_ctrl': ctrl_params,
                'label': torch.tensor(label, dtype=torch.long),
                'y_target': target_params,
                'cell_context_id': torch.tensor(context_id, dtype=torch.long),
            }

        
        if self.has_copula:
            s_target = target_fv if target_fv is not None else self.factor_var_fallback
            s_ctrl = ctrl_fv if ctrl_fv is not None else self.factor_var_fallback
            out['s_target'] = s_target.float()
            out['s_ctrl'] = s_ctrl.float()

        return out
