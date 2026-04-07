class Discriminator(nn.Module):
    def __init__(self, seq_len, channels, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(Discriminator, self).__init__()
        self.d_model = d_model
        
        # Input projection if channels != d_model, otherwise can be nn.Identity() or skipped
        self.input_projection = nn.Linear(channels, d_model) # Project input channels to d_model
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Classifier head
        # Option 1: Use the output of the first token (like [CLS] token if you add one)
        # Option 2: Average pooling over the sequence
        # Option 3: Flatten and pass through Linear layers
        self.flatten = nn.Flatten()
        # The size for the linear layer input depends on seq_len * d_model
        self.fc1 = nn.Linear(seq_len * d_model, 128) # Adjust size
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels)
        x = self.input_projection(x) # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # (batch_size, seq_len, d_model)
        
        x = self.flatten(x) # (batch_size, seq_len * d_model)
        x = self.leaky_relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x)) # Output probability (0 to 1)
        return x
