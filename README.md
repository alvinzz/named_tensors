# Named Tensors
## A PyTorch Library for Dimension-Checking
Have ever run into the following PyTorch message?
```
RuntimeError: The size of tensor a (3) must match the size of tensor b (12) at non-singleton dimension 4
```
If you have, and then spent frustrating minutes digging through the code to
figure out exactly where the `3` and `12` came from, then this library is for you!

Instead of guessing what the `3` and `12` represent, you can give your tensor
dimensions names with the **NamedTensor** library!

For example, this is an implementation of (batched) video self-attention, follwing "Non-local Neural Networks" (Wang et. al., CVPR 2018):
```
import torch
from named_tensor import NamedTensor

[B, T, H, W, IM_CH] = [2, 3, 4, 5, 6]
[KQ_CH, V_CH] = [7, 8]

image = NamedTensor(
    torch.rand(B, T, H, W, IM_CH),
    dims=['B', 'T', 'H', 'W', 'IM_CH'],
)

K_matrix = NamedTensor(
    torch.rand(IM_CH, KQ_CH),
    dims=['IM_CH', 'KQ_CH'],
)
Q_matrix = NamedTensor(
    torch.rand(IM_CH, KQ_CH),
    dims=['IM_CH', 'KQ_CH'],
)
V_matrix = NamedTensor(
    torch.rand(IM_CH, KQ_CH),
    dims=['IM_CH', 'V_CH'],
)

Q = image.einsum(Q_matrix, reduce_dims=['IM_CH'])
Q = Q.join_dims(['T', 'H', 'W'], 'THW_query')
K = image.einsum(K_matrix, reduce_dims=['IM_CH'])
K = K.join_dims(['T', 'H', 'W'], 'THW_key/value')

# Each query is expressed as a normalized sum of keys.
query_key_products = Q.einsum(K, reduce_dims=['KQ_CH'])
# Not numerically stable, but this simple version of soft-max is shown for clarity
exp = query_key_products.apply_elemwise(torch.exp)
exp_sum = exp.reduce(torch.sum, ['THW_query'])
query_key_components = exp / exp_sum

# Each query is converted to a value.
V = image.einsum(V_matrix, reduce_dims=['IM_CH'])
V = V.join_dims(['T', 'H', 'W'], 'THW_key/value')
self_attention_values = query_key_components.einsum(V, reduce_dims=['THW_key/value'])

self_attention_image = self_attention_values.split_dim('THW_query', ['T', 'H', 'W'], [T, H, W])
print(self_attention_image.shape)
```

Special thanks to Matician for allowing me to expose this library, and to my internal reviewers Navneet Dalal, Rowan Dempster, and Oishi Banerjee.

Features coming soon (once I find that branch...): gather, scatter, fancy indexing
Potential features: convolutions