import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cal_uncertainty(singing, singing_hat):
    scores = torch.sqrt((singing - singing_hat) ** 2)  

    return scores

def find_valid_intervals(threshold=0.5, ratio=0.75, pred_uncertainty=None, singing=None, singing_hat=None, from_gt=True):
    # the intervals will be different in each element in the batch
    if from_gt:
        B, T = singing.shape
        uncertainty_scores = cal_uncertainty(singing, singing_hat)
    else:
        B, T = pred_uncertainty.shape
        uncertainty_scores = pred_uncertainty
    min_window_size = T // 10 

    uncertainty_scores.detach()
    valid_intervals = [] 

    
    for b in range(B):
        scores = uncertainty_scores[b]
        valid_mask = scores > threshold  

        # sliding window
        current_start = None
        intervals = []  
        for start in range(T - min_window_size + 1): 
            end = start + min_window_size
            window = valid_mask[start:end] 

            if window.float().mean() >= ratio:
                intervals.append((start, end))

        if len(intervals) == 0:
            # rand_start = np.random.randint(T-T//10+1)
            # valid_intervals.append(np.array([rand_start,rand_start+T//10]))  
            valid_intervals.append(np.array([-1,-1]))   
        else: # if there is valid interval, choose one
            rand_interval = np.random.randint(len(intervals))
            valid_intervals.append(intervals[rand_interval])

    return np.array(valid_intervals), uncertainty_scores


def sample_interval(threshold=3e-5, ratio=0.75, pred_uncertainty=None, singing=None, singing_hat=None, from_gt=True):
    '''
    Args:
        threshold: gap lower bound, >threshold high uncertainty.
        ratio: lower bound of valid frames in one segment.
    '''
    if from_gt:
        T = singing.shape[1] # total frames in one singing segment
        valid_intervals, uncertainty_scores = find_valid_intervals(threshold=threshold, ratio=ratio, singing=singing, singing_hat=singing_hat, from_gt=from_gt)
    else:
        T = pred_uncertainty.shape[1]
        valid_intervals, uncertainty_scores = find_valid_intervals(threshold=threshold, ratio=ratio, pred_uncertainty=pred_uncertainty, from_gt=from_gt)
    valid_intervals = valid_intervals / T # to ratio

    valid_intervals = torch.tensor(valid_intervals)
    
    return valid_intervals, uncertainty_scores # [B * (start, end)]


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, T=10240):
        super().__init__()
        self.embed_dim = embed_dim
        self.T = T
        self.register_buffer("positional_encoding", self._get_sinusoidal_encoding())

    def _get_sinusoidal_encoding(self):
        position = torch.arange(self.T).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / self.embed_dim))
        
        pe = torch.zeros(self.T, self.embed_dim)  # [T, embed_dim]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, T, embed_dim] to match batch shape

    def forward(self, x):
        return x + self.positional_encoding[:, :x.shape[1], :].to(x.device)


class UncertaintyTransformer(nn.Module): # the uncertainty might be temporal dependent
    def __init__(self, input_dim=1, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super(UncertaintyTransformer, self).__init__()
        self.embed_dim = embed_dim
        
        self.T = 10240

        self.preprocess = nn.Linear(self.T, self.T//20)
        # Input embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Position encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.T, embed_dim))  
        nn.init.uniform_(self.positional_encoding, -0.1, 0.1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, input_dim)
        self.postprocess = nn.Linear(self.T//20, self.T)

        self.uncertainty_criterion = nn.MSELoss()
        self.uncertainty_loss_weight = 1.0
    
    def forward(self, x):

        x_n = self.preprocess(x).unsqueeze(-1) # downsample T, 1D signal B, T, 1
        B, T, _ = x_n.shape

        # Input embedding
        x_embed = self.input_proj(x_n)  # [B, T, D]
        
        # Add positional encoding
        pos_embed = self.positional_encoding[:, :T, :]  # [1, T, D]
        x_embed = x_embed + pos_embed
        
        # Transformer expects input shape [T, B, D]
        x_embed = x_embed.transpose(0, 1)  # [T, B, D]

        # Transformer Encoder
        x_transformed = self.transformer_encoder(x_embed)  # [T, B, D]
        
        # Back to original dimension
        x_transformed = x_transformed.transpose(0, 1)  # [B, T, D]
        output = self.output_proj(x_transformed)  # [B, T, 1]

        x_out = self.postprocess(output.squeeze(-1)) # [B, T]

        return x_out



if __name__=='__main__':
    # Example usage
    B, T = 32, 10240
    singing = torch.randn(B, T)  # ori signal (singing)
    singing_hat = torch.randn(B, T)  # gen signal (singing_hat)

    valid_intervals, uncertainty_scores = sample_interval(threshold=0.5, ratio=0.75, singing=singing, singing_hat=singing_hat, from_gt=True)

    # Initialize model
    model = UncertaintyTransformer(input_dim=1, embed_dim=64, num_heads=4, num_layers=3)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward pass
    predicted = model(singing)
    target = singing - singing_hat
    loss = model.uncertainty_criterion(predicted, target) # TODO(yiwen) add uncertainty predictor loss to the whole loss with weight

    loss *= model.uncertainty_loss_weight
    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")
