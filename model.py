import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def safe_logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1-eps)
    return torch.log(p / (1 - p))

def safe_log(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0)
    return torch.log(p)


class DrugEncoder(nn.Module):
    
    def __init__(self, fp_dim=1024, d_model=256, dropout=0.1):
        super().__init__()
        hidden = max(d_model, 512)
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, drug_fp):
        
        return self.net(drug_fp)


class DoseEncoder(nn.Module):
    
    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        hidden = max(d_model // 2, 128)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, dose):
        
        return self.net(dose.unsqueeze(-1))


class DistributionTransformer(nn.Module):
    def __init__(self, vocab_size, n_components=2, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1,
                 perturb_mode='gene', fp_dim=1024, n_cell_types=0,
                 copula_rank=0, dep_hidden=None):
        super().__init__()
        self.d_model = d_model
        self.n_components = n_components

        self.perturb_mode = perturb_mode
        self.fp_dim = fp_dim

        
        param_dim = 1 + 3 * n_components

        
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        
        
        self.shared_embedding = nn.Embedding(vocab_size, d_model)

        self.emb_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        
        if perturb_mode == 'drug':
            self.drug_encoder = DrugEncoder(fp_dim=fp_dim, d_model=d_model, dropout=dropout)
            self.dose_encoder = DoseEncoder(d_model=d_model, dropout=dropout)
            
            self.cell_type_embedding = nn.Embedding(max(n_cell_types, 1), d_model) if n_cell_types and n_cell_types > 0 else None
            
            
            
            
            
            
            self.film_drug = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),   
            )
            nn.init.zeros_(self.film_drug[-1].weight)
            nn.init.zeros_(self.film_drug[-1].bias)
        else:
            self.drug_encoder = None
            self.dose_encoder = None
            self.cell_type_embedding = None
            self.film_drug = None

        
        
        
        
        
        
        self.n_cell_types = max(int(n_cell_types or 0), 0)
        if self.n_cell_types > 0 and perturb_mode == 'gene':
            self.cell_context_embedding = nn.Embedding(self.n_cell_types, d_model)
            self.cell_context_adapter = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            nn.init.zeros_(self.cell_context_adapter[-1].weight)
            nn.init.zeros_(self.cell_context_adapter[-1].bias)
        else:
            self.cell_context_embedding = None
            self.cell_context_adapter = None

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        
        self.head_pi = nn.Linear(d_model, 1)
        self.head_w = nn.Linear(d_model, n_components)
        self.head_mu = nn.Linear(d_model, n_components)
        self.head_sigma = nn.Linear(d_model, n_components)

        for head in [self.head_pi, self.head_w, self.head_mu, self.head_sigma]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        
        
        
        
        self.copula_rank = int(copula_rank) if copula_rank else 0
        if self.copula_rank > 0:
            
            
            self.token_type_embedding = nn.Embedding(2, d_model)
            nn.init.normal_(self.token_type_embedding.weight, std=0.02)
            self.s_ctrl_encoder = nn.Sequential(
                nn.Linear(self.copula_rank, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            
            
            
            self.head_dep = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, self.copula_rank),
            )
            
            nn.init.zeros_(self.head_dep[-1].weight)
            nn.init.zeros_(self.head_dep[-1].bias)
        else:
            self.token_type_embedding = None
            self.s_ctrl_encoder = None
            self.head_dep = None

    def encode_perturbation(self, perturb_indices=None, drug_fp=None, dose=None, cell_type=None):
        
        if self.perturb_mode == 'gene':
            p_emb = self.shared_embedding(perturb_indices)
            p_emb = self.emb_adapter(p_emb)
            return p_emb
        else:
            p_emb = self.drug_encoder(drug_fp) + self.dose_encoder(dose)
            if self.cell_type_embedding is not None and cell_type is not None:
                p_emb = p_emb + self.cell_type_embedding(cell_type)
            return p_emb

    def forward(self, x_params, gene_indices, perturb_indices=None,
                drug_fp=None, dose=None, cell_type=None, s_ctrl=None,
                cell_context_id=None):
        
        B, N, D = x_params.shape
        
        
        
        x_emb = self.param_encoder(x_params)
        
        
        g_emb = self.shared_embedding(gene_indices)
        g_emb = self.emb_adapter(g_emb)
        g_emb = g_emb.unsqueeze(0).expand(B, -1, -1)
        
        
        
        
        if self.token_type_embedding is not None:
            gene_type_id = torch.zeros(B, N, dtype=torch.long, device=x_params.device)
            tgt = x_emb + g_emb + self.token_type_embedding(gene_type_id)
        else:
            tgt = x_emb + g_emb

        
        
        
        
        
        if (self.cell_context_adapter is not None
                and cell_context_id is not None
                and self.perturb_mode == 'gene'):
            ctx_emb = self.cell_context_embedding(cell_context_id)          
            ctx_bias = self.cell_context_adapter(ctx_emb)                   
            tgt = tgt + ctx_bias.unsqueeze(1)                               

        
        p_emb = self.encode_perturbation(
            perturb_indices=perturb_indices,
            drug_fp=drug_fp, dose=dose, cell_type=cell_type,
        )

        
        memory_seq = p_emb.unsqueeze(1)

        
        
        
        
        dep_added = False
        if self.copula_rank > 0 and s_ctrl is not None:
            s_log = torch.log(s_ctrl.clamp(min=1e-6))               
            s_emb = self.s_ctrl_encoder(s_log)                      
            dep_type_id = torch.ones(B, 1, dtype=torch.long, device=x_params.device)
            dep_token = s_emb.unsqueeze(1) + self.token_type_embedding(dep_type_id)  
            tgt = torch.cat([tgt, dep_token], dim=1)                
            dep_added = True

        
        
        
        output = self.transformer_decoder(tgt, memory_seq)          

        
        
        
        if dep_added:
            dep_out = output[:, -1, :]                               
            output = output[:, :N, :]                                

        
        
        
        
        
        if self.film_drug is not None:
            film_params = self.film_drug(p_emb)              
            gamma, beta = film_params.chunk(2, dim=-1)       
            output = output * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  

        K = self.n_components

        
        
        
        
        
        pi_in = x_params[..., 0]               
        w_in = x_params[..., 1:1+K]            
        mu_in = x_params[..., 1+K:1+2*K]       
        sigma_in = x_params[..., 1+2*K:1+3*K]  

            
            
        delta_pi_logit = self.head_pi(output).squeeze(-1)  
        pi_logits = safe_logit(pi_in) + delta_pi_logit
        pi = torch.sigmoid(pi_logits)

        
        delta_w_logit = self.head_w(output)  
        w_logits = safe_log(w_in) + delta_w_logit
        w = F.softmax(w_logits, dim=-1)
        w = torch.clamp(w, min=1e-8)
        w = w / w.sum(dim=-1, keepdim=True)

        
        delta_mu = self.head_mu(output)  
        mu = mu_in + delta_mu

        
        delta_log_sigma = self.head_sigma(output)
        delta_log_sigma = torch.clamp(delta_log_sigma, -5, 5)
        sigma = sigma_in * torch.exp(delta_log_sigma) + 1e-4

        out = {'pi': pi, 'w': w, 'mu': mu, 'sigma': sigma}

        
        
        
        
        
        if dep_added:
            delta_log_s = torch.clamp(self.head_dep(dep_out), -5.0, 5.0)
            out['s'] = s_ctrl * torch.exp(delta_log_s)

        return out
