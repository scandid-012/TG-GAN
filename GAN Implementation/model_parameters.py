# Model Hyperparameters
latent_dim = 100         # Size of the latent space vector for Generator input
d_model = 64             # Feature dimension for Transformer (must be divisible by nhead)
nhead = 4                # Number of attention heads in Transformer
num_encoder_layers_g = 3 # Number of Transformer encoder layers in Generator
num_encoder_layers_d = 3 # Number of Transformer encoder layers in Discriminator
dim_feedforward = 256    # Dimension of the feedforward network in Transformer layers
dropout = 0.1            # Dropout rate

# Training Hyperparameters
lr_g = 0.0002
lr_d = 0.0002
beta1 = 0.5 # Adam optimizer beta1
beta2 = 0.999
num_epochs = 100 # Start with a smaller number to test (e.g., 50-100), GANs take time

# Initialize models (as you have them)
generator = Generator(latent_dim, sequence_length, num_eeg_channels, d_model, nhead, num_encoder_layers_g, dim_feedforward, dropout).to(device)
discriminator = Discriminator(sequence_length, num_eeg_channels, d_model, nhead, num_encoder_layers_d, dim_feedforward, dropout).to(device)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for Generator and Discriminator.")
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
# If not using DataParallel, or only 1 GPU, they are already on `device` from the initial .to(device)

# Your optimizers should be created AFTER wrapping with DataParallel
# if you're using it, as DataParallel might add a 'module.' prefix to parameter names.
# However, optimizers typically take model.parameters(), and DataParallel forwards this.
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

# Loss function
criterion = nn.BCELoss() # Binary Cross-Entropy Loss
