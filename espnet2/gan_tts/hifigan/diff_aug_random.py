import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
"""
Assume the input is in shape B, 1, T
NOTE(yiwen) check whether the input and output are the right tensor format
choose between 'pitch_augmentation', 'mixup_gaussian_noise',  'random_time_mask', 'random_freq_mask', 'time_warp'
"""

def DiffAugment(x, policy=''): 
    """
    input: 
        - x: the final layer output of three types of discriminators, 
          high-level time domain features or hight-level frequency domain features.
            - tensor(B, C, T): from MSD, C=1
            - tensor(B, T): from MPD
            - tensor(B, C, F, T): from MFD, C=1
        - policy: the augmentation method to use
    """
    reshape_2D=False
    reshape_4D=False
    if policy:
        if len(x.shape)==2:
            reshape_2D=True
            x = x.unsqueeze(1) # B, C, T
            
        for p in policy.split(','):             
            for f in AUGMENT_FNS[p]:
                if len(x.shape)==4 and f==random_time_mask:
                    continue
                elif len(x.shape)==3 and f==random_freq_mask:
                    continue                
                # else:
                #     if p=='mixup' or p=='warp': # MFD aug in frequency domain
                #         if len(x.shape)==4: # 8, 1, 28, 12
                #             reshape_4D=True
                #             B, C, F, T = x.shape
                #             x = x.permute(0,3,1,2) # --> B, T, C, F
                #             x = x.reshape(B, -1, F) # B, C', F
                            
                if p=='mixup' or p=='warp': # MFD aug in time domain
                    if len(x.shape)==4: # 8, 1, 28, 12
                        reshape_4D=True
                        B, C, F, T = x.shape
                        x = x.reshape(B, -1, T) # B, C', T
                    
                x = f(x)
                    
                    # if reshape_4D: # MFD aug in frequency domain
                    #     x = x.reshape(B, T, C, F)
                    #     x = x.permute(0, 2, 3, 1) # B, 1, F, T
                    
                if reshape_4D: # MFD aug in time domain
                    x = x.reshape(B, C, F, T)
                    reshape_4D=False

        if reshape_2D:
            x = x.squeeze(1) 
            reshape_2D=True
        x = x.contiguous()
    return x


def pitch_shift(batch_waveform_segment, pitch_factor):
    """Applies pitch shift by resampling and interpolating for a batch of waveform segments."""
    B, _, segment_length = batch_waveform_segment.shape
    
    # print(f'debug2 -- batch_waveform_segment.shape{batch_waveform_segment.shape}')

    new_length = int(segment_length / pitch_factor) # shorter than the original seg, raise the pitch
    
    # --> B, 1, segT
    pitch_shifted_segment = F.interpolate(batch_waveform_segment, size=new_length, mode='linear', align_corners=False)
    
    # Pad to the original segment length along the time dimension
    padded_segment = F.pad(pitch_shifted_segment, (0, segment_length - new_length), mode='constant') # only pad the last dim, at back
    
    return padded_segment

def pitch_augmentation(batch_waveform, pitch_factor=1.3, segment_ratio=0.1):
    """
    Apply pitch shift augmentation to a random segment of each sample in the batch.
    
    Parameters:
    - batch_waveform (Tensor): Input audio batch of shape (B, 1, T).
    - pitch_factor (float): Factor by which to shift the pitch (e.g., 1.3 for a 30% increase).
    - segment_ratio (float): Ratio of the waveform to select for pitch shifting.
    
    Returns:
    - augmented_batch_waveform (Tensor): Audio batch after pitch shift augmentation.
    """

    B, _, T = batch_waveform.shape
    segment_length = int(T * segment_ratio)
    
    if segment_length / pitch_factor <= 1:
        return batch_waveform # NOTE(yiwen) if too short, don't do this augmentation
    
    start_idx = np.random.randint(0, T - segment_length)  # Random start index for the segment
    
    # print(f'debug1 -- start_idx {start_idx}')
    # Select the same random segment from each sample in the batch
    batch_waveform_segment = batch_waveform[:, :, start_idx:start_idx + segment_length]
   
    # Apply pitch shift to the selected segment
    shifted_segment = pitch_shift(batch_waveform_segment, pitch_factor)
    
    # Replace the segment in the original waveform with the pitch-shifted segment
 
    augmented_batch_waveform = torch.cat([
        batch_waveform[:, :, :start_idx],
        shifted_segment,
        batch_waveform[:, :, start_idx+segment_length:]
    ], dim=2)
    
    # print(f'debug3 -- augmented_batch_waveform.shape {augmented_batch_waveform.shape}')
    
    return augmented_batch_waveform


def add_gaussian_noise(waveform, noise_factor=0.001):
    """Add Gaussian noise to the waveform."""
    noise = torch.randn_like(waveform) * noise_factor
    noisy_waveform = waveform + noise
    return noisy_waveform


def mixup_gaussian_noise(batch_waveform, noise_factor=0.005, segment_ratio=0.5):
    """
    Apply random Gaussian noise to waveform.
    
    Parameters:
    waveform (torch.Tensor): Input waveform tensor of shape (n_samples,)
    noise_factor (float): gaussian noise factor
    segment_ratio (float): time ratio to add gaussian noise
    
    Returns:
    torch.Tensor: Waveform with random Gaussian noise applied
    """
    # print(f'debug -- batch_waveform.shape {batch_waveform.shape}')
    B, _, T = batch_waveform.shape
    segment_length = int(T * segment_ratio)
    start_idx = np.random.randint(0, T - segment_length)
    batch_waveform_segment = batch_waveform[:, :, start_idx:start_idx + segment_length]
    
    # Add Gaussian Noise
    gaussian_noisy_segment = add_gaussian_noise(batch_waveform_segment, noise_factor=noise_factor)
    padded_segment = F.pad(gaussian_noisy_segment, (0, segment_length - gaussian_noisy_segment.shape[-1]), mode='constant')

    gaussian_noisy_waveform = torch.cat([
        batch_waveform[:, :, :start_idx],
        padded_segment,
        batch_waveform[:, :, start_idx+segment_length:]
    ], dim=2)
    
    return gaussian_noisy_waveform



def random_time_mask(batch_waveform, ratio=0.1, mask_intensity=0):
    """
    :param batch_waveform: (B, 1, T)
    """
    assert batch_waveform.dim() == 3, "Input tensor must have 3 dimensions (B, 1, T)."
    B, _, T = batch_waveform.shape
    cutout_size = int(ratio * T)

    start_idx = np.random.randint(0, T - cutout_size)
    mask = torch.ones((B, 1, T), dtype=batch_waveform.dtype, device=batch_waveform.device)

    mask[:, :, start_idx:start_idx + cutout_size] = mask_intensity
    masked_waveform = batch_waveform * mask

    return masked_waveform



def random_freq_mask(x, ratio=0.1):
    """
    Apply a differentiable random cutout mask to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, 1, F, T).
        ratio (float): Ratio of the cutout region to the spectrogram feature map size.

    Returns:
        torch.Tensor: Masked tensor with the same shape as input.
    """
    assert x.dim() == 4, "Input tensor must have 4 dimensions (B, 1, F, T)."
    B, _, F, T = x.shape
    # print(f'debug -- x {x.shape}')
    # Calculate cutout size
    cutout_size = (int(F * ratio + 0.5), int(T * ratio + 0.5))
    
    # Random offsets for the cutout center
    offset_f = torch.randint(0, F + (1 - cutout_size[0] % 2), size=[B, 1, 1], device=x.device)
    offset_t = torch.randint(0, T + (1 - cutout_size[1] % 2), size=[B, 1, 1], device=x.device)

    # Create meshgrid for batch, frequency, and time
    grid_batch, grid_f, grid_t = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    
    # Adjust grid coordinates with the offsets
    grid_f = torch.clamp(grid_f + offset_f - cutout_size[0] // 2, min=0, max=F - 1)
    grid_t = torch.clamp(grid_t + offset_t - cutout_size[1] // 2, min=0, max=T - 1)

    # Create the mask
    mask = torch.ones((B, F, T), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_f, grid_t] = 0  # Set the cutout region to 0

    # Apply the mask to the input
    masked_x = x * mask.unsqueeze(1)  # unsqueeze to match the channel dimension
    # print(f'debug -- masked_x {masked_x.shape}')
    return masked_x



def time_warp(batch_waveform, max_time_warp=80):
    """
    Apply time warp on a batch of spectrograms with the same warp transformation.
    
    Parameters:
    batch_spectrogram (torch.Tensor): Batch spectrogram of shape (B, 1*F, T).
    max_time_warp (int): Maximum time frames to warp.
    
    Returns:
    torch.Tensor: Time-warped batch spectrogram of the same shape as input.
    """
    B, _, T = batch_waveform.shape  # Extract batch, time, and frequency dimensions

    # Check if warping is possible
    if T <= max_time_warp * 2:
        return batch_waveform  # No warping if time dimension is too small
    
    # Select a random center for the warp
    center = np.random.randint(max_time_warp, T - max_time_warp)
    
    # Determine warp distance
    window = max_time_warp
    
    center = np.random.randint(window, T - window)
    warped = np.random.randint(center - window, center + window) + 1  # Random warp

    # Split the spectrogram into left and right parts based on the center
    left = batch_waveform[:,:,:center]
    right = batch_waveform[:,:,center:]

    # Apply time warping: resize left and right parts using bilinear interpolation
    left_resized = F.interpolate(left, size=warped, mode='bilinear', align_corners=False)
    right_resized = F.interpolate(right, size=T - warped, mode='bilinear', align_corners=False)

    # Concatenate the left and right parts after warping
    warped_batch = torch.cat([left_resized, right_resized], dim=-1)
        
    return warped_batch



AUGMENT_FNS = {
    'mask': [random_time_mask, random_freq_mask],
    'warp': [pitch_augmentation],
    'mixup': [mixup_gaussian_noise],
}