**problem:** the random data will lead to gradient vanish in some of them (approaching edge) when process through activation function like sigmoid. So the data need to be normalized called batch normalized

another view is that: when all the data is normalized, the weight associate with each other give similar gradient on the learning. Given that no matter how initilization the training, the gradient will be the same

- image normalization by \
`transforms.normalize()`\
- batch normalizatopm
    - calculate by the mean and dev of each channel (RGB)\
    `nn.BatchNorm1d()`\

