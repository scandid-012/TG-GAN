class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, channels, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.d_model = d_model

        self.init_projection = nn.Linear(latent_dim, seq_len * d_model) # Project latent to match seq_len * d_model
        # Alternative: Project to d_model and then tile/repeat or use a learned sequence embedding

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            activation='gelu' # GELU is common in Transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Output layer to get to the desired channel dimension
        self.output_layer = nn.Linear(d_model, channels)
        self.tanh = nn.Tanh() # To output values in [-1, 1]

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        x = self.init_projection(z) # (batch_size, seq_len * d_model)
        x = x.view(x.size(0), self.seq_len, self.d_model) # Reshape to (batch_size, seq_len, d_model)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # (batch_size, seq_len, d_model)
        x = self.output_layer(x) # (batch_size, seq_len, channels)
        x = self.tanh(x) # Scale to [-1, 1]
        return x
