import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p / (1 - p))


def safe_log(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0)
    return torch.log(p)


PARAM_DIM = 3  


class ZINBTransformer(nn.Module):
    

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=1024, dropout=0.1,
                 copula_rank=0, dep_hidden=None, n_cell_types=0):
        super().__init__()
        self.d_model = d_model
        self.copula_rank = int(copula_rank) if copula_rank else 0

        self.param_encoder = nn.Sequential(
            nn.Linear(PARAM_DIM, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.emb_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        
        self.n_cell_types = max(int(n_cell_types or 0), 0)
        if self.n_cell_types > 0:
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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head_pi = nn.Linear(d_model, 1)
        self.head_mu = nn.Linear(d_model, 1)
        self.head_theta = nn.Linear(d_model, 1)

        for head in [self.head_pi, self.head_mu, self.head_theta]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        
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

    def forward(self, x_params, gene_indices, perturb_indices,
                s_ctrl=None, cell_context_id=None):
        
        B, N, D = x_params.shape

        
        pi_in_2d = x_params[..., 0]                                    
        mu_raw = x_params[..., 1]                                      
        theta_raw = x_params[..., 2]                                   
        log_mu_in = torch.log(torch.clamp(mu_raw, min=1e-3))          
        log_theta_in = torch.log(torch.clamp(theta_raw, min=1e-3))     

        x_scaled = torch.stack([pi_in_2d, log_mu_in, log_theta_in], dim=-1)  
        x_emb = self.param_encoder(x_scaled)

        g_emb = self.shared_embedding(gene_indices)
        g_emb = self.emb_adapter(g_emb).unsqueeze(0).expand(B, -1, -1)

        tgt = x_emb + g_emb

        
        if self.token_type_embedding is not None:
            gene_type_id = torch.zeros(B, N, dtype=torch.long, device=x_params.device)
            tgt = tgt + self.token_type_embedding(gene_type_id)

        
        if (self.cell_context_adapter is not None
                and cell_context_id is not None):
            ctx_emb = self.cell_context_embedding(cell_context_id)
            ctx_bias = self.cell_context_adapter(ctx_emb).unsqueeze(1)
            tgt = tgt + ctx_bias

        p_emb = self.shared_embedding(perturb_indices)
        p_emb = self.emb_adapter(p_emb).unsqueeze(1)      

        
        dep_added = False
        if self.copula_rank > 0 and s_ctrl is not None:
            s_log = torch.log(s_ctrl.clamp(min=1e-6))               
            s_emb = self.s_ctrl_encoder(s_log)                      
            dep_type_id = torch.ones(B, 1, dtype=torch.long, device=x_params.device)
            dep_token = s_emb.unsqueeze(1) + self.token_type_embedding(dep_type_id)
            tgt = torch.cat([tgt, dep_token], dim=1)                
            dep_added = True

        output = self.transformer_decoder(tgt, p_emb)               

        
        if dep_added:
            dep_out = output[:, -1, :]                               
            output = output[:, :N, :]                                

        
        pi_ctrl = pi_in_2d
        mu_ctrl = mu_raw
        log_theta_ctrl = log_theta_in

        delta_pi = self.head_pi(output).squeeze(-1)
        pi = torch.sigmoid(safe_logit(pi_ctrl) + delta_pi)

        delta_mu = torch.clamp(self.head_mu(output).squeeze(-1), -10, 10)
        mu = mu_ctrl * torch.exp(delta_mu) + 1e-4

        delta_theta = torch.clamp(self.head_theta(output).squeeze(-1), -10, 10)
        theta = torch.exp(log_theta_ctrl + delta_theta) + 1e-4

        pi = torch.nan_to_num(pi, nan=0.5)
        mu = torch.nan_to_num(torch.clamp(mu, min=1e-4, max=1e6), nan=1.0)
        theta = torch.nan_to_num(torch.clamp(theta, min=1e-4, max=1e6), nan=1.0)

        out = {'pi': pi, 'mu': mu, 'theta': theta}

        
        
        if dep_added:
            delta_log_s = torch.clamp(self.head_dep(dep_out), -5.0, 5.0)  
            out['s'] = s_ctrl * torch.exp(delta_log_s)                     

        return out
