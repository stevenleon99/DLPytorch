the number of parameter
 - equal to the connection number between each pair of neurons
 - Param([784, 256, 256, 256, 256, 10]) = 784*256 + 256^2*3 + 256*10 = 399872
 - 399872 * 4byte = 1.5M
 - for the convolution layer, calculate the subsampling layer to the FC (full connection) and the following + kernel's parameter [3,1,3,3] 3 kernels, 1 inputs, 3 h, 3 w

kernel
 - each kernel can extract different feature of the img
 - each kernel can be analogy as a view to the image
 - [3, 28, 28] RGB channel, h, w -> [6, 3, 28, 28] number of kernel, number of previous first number, h, w

ResNet
 - introduce shortcut/skip
 - the mdoel can decide whether short cut the layers to achieve better performance