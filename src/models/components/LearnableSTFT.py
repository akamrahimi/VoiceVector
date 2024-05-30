import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableSTFTFunction(nn.Module):
    def __init__(self, kernel_size, num_kernels):
        super(LearnableSTFTFunction, self).__init__()
        self.kernel_size = kernel_size
        self.learnable_weights = nn.Parameter(torch.randn(num_kernels, kernel_size))

    def forward(self, x):
      x_padded = F.pad(x, (0, self.kernel_size - 1))
      x_windows = x_padded.unfold(-1, self.kernel_size, 1)
      x_weighted = torch.einsum('ijklm,lm->ijkml', x_windows.unsqueeze(-2), self.learnable_weights)
      return x_weighted


class CustomCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_kernels):
        super(CustomCNNLayer, self).__init__()
        self.learnable_stft = LearnableSTFTFunction(kernel_size, num_kernels)
        self.conv1d = nn.Conv1d(in_channels * kernel_size * num_kernels, out_channels, 1)

    def forward(self, x):
        b, c, s = x.shape
        x = self.learnable_stft(x)

        x = x.reshape(b, -1, s)  # Combine the STFT functions and input channels
        x = F.relu(self.conv1d(x))
        return x