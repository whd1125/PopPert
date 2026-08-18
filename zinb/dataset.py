import torch
from torch.utils.data import Dataset
import numpy as np


class ZINBDataset(Dataset):
    def __init__(self, data_path, split='train', split_ratio=0.7, seed=42):
        print(f"Loading preprocessed ZINB distribution data from {data_path}...")
        loaded = torch.load(data_path, weights_only=False)
        self._rng = np.random.RandomState(seed)

        if 'val_data' in loaded and 'test_data' in loaded:
            print(f"Detected pre-split data (Train/Val/Test). Split: {split}")
            self.hvg_indices = loaded['hvg_indices']
            full_data = loaded['train_data'] + loaded['val_data'] + loaded['test_data']
            self.control_batches = [d for d in full_data if d['is_control']]
            source_data = {'train': loaded['train_data'], 'val': loaded['val_data'],
                           'test': loaded['test_data']}[split]
            self.perturb_data = [d for d in source_data if not d['is_control']]
            print(f"Selected {len(self.perturb_data)} batches for {split.upper()}")
        elif 'train_data' in loaded:
            all_data = loaded['train_data'] + loaded.get('test_data', [])
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
            if split == 'train':
                self.perturb_data = [d for d in all_perturb if d['label'] in train_labels]
            else:
                self.perturb_data = [d for d in all_perturb if d['label'] in test_labels]
            self.control_batches = all_control
        else:
            print("WARNING: Old data format.")
            all_data = loaded['data']

        
        self.factor_loadings = loaded.get('factor_loadings')
        self.copula_ctrl_mean = loaded.get('copula_ctrl_mean')
        self.copula_ctrl_std = loaded.get('copula_ctrl_std')
        self.copula_rank = int(loaded.get('copula_rank', 0) or 0)
        self.has_copula = (self.factor_loadings is not None and self.copula_rank > 0)
        if self.has_copula:
            print(f"Copula enabled: A={tuple(self.factor_loadings.shape)}, r={self.copula_rank}")

        
        self.cell_line_vocab = loaded.get('cell_line_vocab', {})
        self.n_cell_types = int(loaded.get('n_cell_types', 0) or 0)

        self.perturb_batches = {}
        for item in self.perturb_data:
            lbl = item['label']
            if lbl not in self.perturb_batches:
                self.perturb_batches[lbl] = []
            self.perturb_batches[lbl].append(item)

        self.perturb_samples = []
        for lbl, batches in self.perturb_batches.items():
            for b in batches:
                self.perturb_samples.append(
                    (b['params'], lbl, b.get('cell_line', 'unknown'), b.get('factor_var'))
                )

        self.control_by_cell = {}
        for cb in self.control_batches:
            cline = cb.get('cell_line', 'unknown')
            if cline not in self.control_by_cell:
                self.control_by_cell[cline] = []
            self.control_by_cell[cline].append(cb)

        
        if self.has_copula:
            fvs = [cb.get('factor_var') for cb in self.control_batches
                   if cb.get('factor_var') is not None]
            self.factor_var_fallback = torch.stack(fvs).mean(dim=0).float() if fvs \
                else torch.ones(self.copula_rank, dtype=torch.float32)
        else:
            self.factor_var_fallback = None

        print(f"Dataset ({split}): {len(self.control_batches)} ctrl batches, "
              f"{len(self.perturb_samples)} perturb batches.")
        print(f"Ctrl by cell_line: { {k: len(v) for k, v in self.control_by_cell.items()} }")

    def __len__(self):
        if len(self.perturb_samples) == 0:
            return 0
        return len(self.perturb_samples)

    def __getitem__(self, idx):
        real_idx = idx % len(self.perturb_samples)
        target_params, label, cell_line, target_fv = self.perturb_samples[real_idx]

        available_controls = self.control_by_cell.get(cell_line)
        if available_controls and len(available_controls) > 0:
            ctrl_batch = available_controls[self._rng.randint(len(available_controls))]
        else:
            ctrl_batch = self.control_batches[np.random.randint(len(self.control_batches))] \
                if self.control_batches else None
        ctrl_params = ctrl_batch['params'] if ctrl_batch else torch.zeros_like(target_params)
        ctrl_fv = ctrl_batch.get('factor_var') if ctrl_batch else None

        context_id = self.cell_line_vocab.get(cell_line, 0) if self.cell_line_vocab else 0
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
