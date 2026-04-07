import torch
import torch.nn as nn
import os # Make sure to import os for path checking
# Assuming your Generator, Discriminator, PositionalEncoding classes are defined above
# Assuming all hyperparameters (latent_dim, d_model, etc.), device, dataloader are defined

# --- Initialize models and optimizers (as you did before) ---
generator = Generator(latent_dim, sequence_length, num_eeg_channels, d_model, nhead, num_encoder_layers_g, dim_feedforward, dropout)
discriminator = Discriminator(sequence_length, num_eeg_channels, d_model, nhead, num_encoder_layers_d, dim_feedforward, dropout)

generator.to(device)
discriminator.to(device)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

criterion = nn.BCELoss()

# --- Variables to manage resumption ---
start_epoch = 0
# **IMPORTANT**: Update this path to where your checkpoint dataset is mounted in Kaggle
# e.g., if you named your dataset "my-gan-checkpoints", it might be:
# "/kaggle/input/my-gan-checkpoints/gan_transformer_checkpoint.pth"
# Or, if you save with epoch numbers:
# "/kaggle/input/my-gan-checkpoints/gan_transformer_checkpoint_epoch_XX.pth"
# For simplicity, let's assume a single checkpoint file name for now.
checkpoint_load_path = "/kaggle/input/tg-gan-checkpoints/gan_transformer_checkpoint.pth"
# List to store loss history if you want to resume it (optional)
g_losses_log = []
d_losses_log = []


# --- Load Checkpoint ---
if os.path.exists(checkpoint_load_path):
    print(f"Loading checkpoint from {checkpoint_load_path}...")
    try:
        checkpoint = torch.load(checkpoint_load_path, map_location=device)

        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch']
        g_losses_log = checkpoint.get('g_losses', []) # Load if exists, else empty
        d_losses_log = checkpoint.get('d_losses', []) # Load if exists, else empty

        print(f"Resuming training from epoch {start_epoch + 1}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting training from scratch.")
        start_epoch = 0
        g_losses_log = []
        d_losses_log = []
else:
    print(f"No checkpoint found at '{checkpoint_load_path}'. Starting training from scratch.")
    start_epoch = 0


# --- Wrap with DataParallel if multiple GPUs are available AFTER loading state dicts ---
# This order ensures state dict keys match the base model before DataParallel adds 'module.' prefix
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for Generator and Discriminator.")
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    # Note: Optimizers were already created with base model parameters.
    # PyTorch optimizers hold references to the parameters. When models are wrapped by
    # DataParallel after optimizers are created and state_dicts loaded, this can sometimes
    # be tricky if not handled carefully. A safer alternative if issues arise is to
    # re-initialize optimizers AFTER wrapping with DataParallel if loading a checkpoint,
    # and then load their states. However, usually, loading state into base model
    # then wrapping, then loading optimizer state that refers to those base model params works.
    # For maximum safety when resuming with DataParallel and optimizers:
    # 1. Load model state into base model.
    # 2. Wrap with DataParallel.
    # 3. Create new optimizers with `dataparallel_model.parameters()`.
    # 4. Load optimizer state.
    # But let's try the simpler current approach first.

print("Starting Training Loop...")
for epoch in range(start_epoch, num_epochs): # Start from 'start_epoch'
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    epoch_d_loss_real = 0.0
    epoch_d_loss_fake = 0.0
    num_batches = 0

    for i, (real_eeg_batch,) in enumerate(dataloader): # Dataloader returns a tuple
        current_batch_size = real_eeg_batch.size(0)
        real_eeg_batch = real_eeg_batch.to(device)

        # --- Train Discriminator ---
        # Set discriminator to trainable and zero gradients
        for param in discriminator.parameters():
            param.requires_grad = True
        discriminator.zero_grad()


        # Labels for real and fake data
        real_labels = torch.ones(current_batch_size, 1, device=device)
        fake_labels = torch.zeros(current_batch_size, 1, device=device)

        # Discriminator loss on real EEG
        outputs_real = discriminator(real_eeg_batch)
        d_loss_real = criterion(outputs_real, real_labels)
        
        # Generate fake EEG
        z = torch.randn(current_batch_size, latent_dim, device=device)
        # Use .detach() when G's output is fed to D during D's training phase
        fake_eeg_batch = generator(z).detach() 

        # Discriminator loss on fake EEG
        outputs_fake = discriminator(fake_eeg_batch)
        d_loss_fake = criterion(outputs_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Accumulate discriminator losses for epoch average
        epoch_d_loss += d_loss.item()
        epoch_d_loss_real += d_loss_real.item()
        epoch_d_loss_fake += d_loss_fake.item()

        # --- Train Generator ---
        # Set discriminator to non-trainable (for G's training phase)
        # and generator to trainable, zero G's gradients
        for param in discriminator.parameters():
            param.requires_grad = False # Freeze D during G update
        generator.zero_grad()
        
        # Generate new fake EEG (G needs to track gradients through these)
        z_for_g = torch.randn(current_batch_size, latent_dim, device=device)
        fake_eeg_for_g = generator(z_for_g)
        
        # We want the discriminator to think the fake EEG is real
        outputs_fake_for_g = discriminator(fake_eeg_for_g) 
        g_loss = criterion(outputs_fake_for_g, real_labels) # G wants D to output 1 (real)
        
        g_loss.backward()
        optimizer_g.step()

        # Accumulate generator loss
        epoch_g_loss += g_loss.item()
        num_batches +=1


        if (i + 1) % 100 == 0: # Print progress e.g. every 100 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f} (Real: {d_loss_real.item():.4f}, Fake: {d_loss_fake.item():.4f}), "
                  f"G Loss: {g_loss.item():.4f}")
    
    # Log average losses for the epoch
    avg_epoch_g_loss = epoch_g_loss / num_batches
    avg_epoch_d_loss = epoch_d_loss / num_batches
    avg_epoch_d_real = epoch_d_loss_real / num_batches
    avg_epoch_d_fake = epoch_d_loss_fake / num_batches
    g_losses_log.append(avg_epoch_g_loss)
    d_losses_log.append(avg_epoch_d_loss) # You might want to log D_real and D_fake separately too

    print(f"--- Epoch [{epoch+1}/{num_epochs}] Summary ---")
    print(f"Avg D Loss: {avg_epoch_d_loss:.4f} (Avg Real: {avg_epoch_d_real:.4f}, Avg Fake: {avg_epoch_d_fake:.4f})")
    print(f"Avg G Loss: {avg_epoch_g_loss:.4f}")


    # Optionally, save generated samples or model checkpoints each epoch
    if (epoch + 1) % 10 == 0: # e.g. every 10 epochs
        with torch.no_grad():
            # Use .module if wrapped with DataParallel, else just generator
            current_generator = generator.module if isinstance(generator, nn.DataParallel) else generator
            current_generator.eval() # Set generator to evaluation mode
            fixed_noise = torch.randn(16, latent_dim, device=device) # Generate a few samples
            generated_samples = current_generator(fixed_noise).cpu().numpy()
            print(f"Generated samples at epoch {epoch+1} (first sample, first channel, first 10 points):")
            print(generated_samples[0, :10, 0])
            current_generator.train() # Set back to train mode
        
        # --- Save Checkpoint Periodically ---
        # Define a path in Kaggle's writable directory
        # You might want to save every N epochs, or the latest one
        # Using a consistent name will overwrite, which is fine for resuming the latest
        checkpoint_save_path = "/kaggle/working/gan_transformer_checkpoint.pth"
        # Or save with epoch number:
        # checkpoint_save_path = f"/kaggle/working/gan_transformer_checkpoint_epoch_{epoch+1}.pth"

        print(f"\nSaving checkpoint at epoch {epoch + 1}...")
        
        # Access .module.state_dict() if models are wrapped in DataParallel
        gen_sd_to_save = generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict()
        disc_sd_to_save = discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict()

        checkpoint = {
            'epoch': epoch + 1, # Save the number of epochs completed
            'generator_state_dict': gen_sd_to_save,
            'discriminator_state_dict': disc_sd_to_save,
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'g_losses': g_losses_log, # Save loss history
            'd_losses': d_losses_log,
        }
        torch.save(checkpoint, checkpoint_save_path)
        print(f"Checkpoint saved to {checkpoint_save_path}")


print("Training Finished.")
