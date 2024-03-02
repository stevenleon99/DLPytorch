import torch
import torch.nn as nn

# Example input with batch size of 2, 2 features
input = torch.randn(3, 3)
print(input)

# Applying LayerNorm to normalize the 2 features
layer_norm = nn.LayerNorm(normalized_shape=[3], eps=1e-5, elementwise_affine=True)

# elementwise_affine: 
# This normalization can unintentionally limit the representational power of the 
# neural network because it forces the outputs to adhere to this standardized 
# distribution, potentially removing useful information embedded in the scale and 
# shift of the data.
# By introducing learnable parameters for 
# scaling (gamma) and shifting (beta) after normalization, 
# the affine transformation allows the network to recover 
# any representation that might have been lost during the normalization process. 

# Normalized output
output = layer_norm(input)
print(output)