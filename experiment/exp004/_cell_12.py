# ProtoSSM — Prototypical State Space Model

class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective state space model.
    
    Input-dependent (selective) discretization of continuous-time SSM:
        dx/dt = Ax + Bu,  y = Cx + Du
    where A, B, C are functions of the input (selectivity).
    
    For T=12 bioacoustic windows, the sequential scan is efficient on CPU.
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Input projection: x -> (x_ssm, z_gate)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        
        # Causal conv1d for local context before SSM
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv,
            padding=d_conv - 1, groups=d_model
        )
        
        # SSM parameters
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        # A initialized as structured matrix (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D is the skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
        # B and C projections — input-dependent = selective
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        B_size, T, D = x.shape
        
        # Split into SSM path and gate
        xz = self.in_proj(x)  # (B, T, 2D)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # Causal conv1d
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Compute input-dependent SSM parameters
        dt = F.softplus(self.dt_proj(x_conv))  # (B, T, D)
        B_t = self.B_proj(x_conv)               # (B, T, N)
        C_t = self.C_proj(x_conv)               # (B, T, N)
        A = -torch.exp(self.A_log)               # (D, N), negative for stability
        
        # Sequential scan (efficient for T=12)
        y = self._selective_scan(x_conv, dt, A, B_t, C_t)
        
        # Gated output
        y = y * F.silu(z)
        return self.out_proj(y)
    
    def _selective_scan(self, x, dt, A, B, C):
        """Selective scan with input-dependent discretization.
        
        x:  (batch, T, D)
        dt: (batch, T, D) — step sizes
        A:  (D, N) — state matrix (log-space, already negated)
        B:  (batch, T, N) — input matrix
        C:  (batch, T, N) — output matrix
        """
        batch, T, D = x.shape
        N = self.d_state
        
        h = torch.zeros(batch, D, N, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(T):
            dt_t = dt[:, t, :, None]          # (batch, D, 1)
            dA = torch.exp(A[None] * dt_t)     # (batch, D, N)
            dB = dt_t * B[:, t, None, :]       # (batch, D, N)
            
            h = h * dA + x[:, t, :, None] * dB  # state update
            y_t = (h * C[:, t, None, :]).sum(-1) # output projection
            ys.append(y_t)
        
        y = torch.stack(ys, dim=1)  # (batch, T, D)
        return y + x * self.D[None, None, :]  # skip connection


class ProtoSSM(nn.Module):
    """Prototypical State Space Model for temporal bioacoustic event detection.
    
    Architecture:
    1. Linear projection of Perch embeddings (1536 -> d_model)
    2. Bidirectional Selective SSM for temporal context
    3. Prototypical cosine similarity classification
    4. Gated fusion with Perch foundation model logits
    
    Args:
        d_input: Perch embedding dimension (1536)
        d_model: Internal model dimension
        d_state: SSM state dimension
        n_ssm_layers: Number of bidirectional SSM layers
        n_classes: Number of species classes (234)
        n_windows: Windows per file (12)
        dropout: Dropout rate
    """
    
    def __init__(self, d_input=1536, d_model=128, d_state=16,
                 n_ssm_layers=2, n_classes=234, n_windows=12, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows
        
        # 1. Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. Learnable positional encoding for temporal position within file
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        
        # 3. Bidirectional SSM layers with residual connections
        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)
        
        # 4. Learnable class prototypes (initialized from data)
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        
        # 5. Per-class gated fusion with Perch logits
        #    sigmoid(alpha) blends: alpha*proto + (1-alpha)*perch
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
        
        # 6. Taxonomic auxiliary head (set after loading taxonomy)
        self.n_families = 0
        self.family_head = None
    
    def init_prototypes_from_data(self, embeddings, labels):
        """Initialize prototypes as normalized class-mean embeddings.
        
        embeddings: (N, d_input) raw Perch embeddings  
        labels: (N, n_classes) binary label matrix
        """
        with torch.no_grad():
            h = self.input_proj(embeddings)  # (N, d_model)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)
    
    def init_family_head(self, n_families, class_to_family):
        """Initialize taxonomic auxiliary head.
        
        n_families: number of unique families
        class_to_family: (n_classes,) mapping class index to family index
        """
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))
    
    def forward(self, emb, perch_logits=None):
        """
        emb: (B, T, d_input) — Perch embeddings per file
        perch_logits: (B, T, n_classes) — Perch mapped logits (optional)
        
        Returns: 
            species_logits: (B, T, n_classes)
            family_logits: (B, T, n_families) or None
            h_temporal: (B, T, d_model) — for analysis
        """
        B, T, _ = emb.shape
        
        # Project embeddings
        h = self.input_proj(emb)  # (B, T, d_model)
        h = h + self.pos_enc[:, :T, :]
        
        # Bidirectional SSM
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)                        # forward scan
            h_b = bwd(h.flip(1)).flip(1)         # backward scan
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)               # residual + layernorm
        
        h_temporal = h  # save for analysis
        
        # Prototypical cosine similarity
        h_norm = F.normalize(h, dim=-1)          # (B, T, d_model)
        p_norm = F.normalize(self.prototypes, dim=-1)  # (C, d_model)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp    # (B, T, C)
        
        # Gated fusion with Perch logits
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]  # (1, 1, C)
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim
        
        # Taxonomic auxiliary prediction
        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)  # (B, d_model) — file-level
            family_logits = self.family_head(h_pool)  # (B, n_families)
        
        return species_logits, family_logits, h_temporal
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


print("ProtoSSM architecture defined.")
print(f"Parameter count (d_model=128, 2 layers): {ProtoSSM(d_model=128, n_ssm_layers=2).count_parameters():,}")