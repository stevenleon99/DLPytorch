sigmoid /logistic function
 - f(x) = 1/(1+exp(-x))
 - derivative of f(x) = f(x)*(1-f(x))

tanh
 - the gradient is larger than sigmoid

ReLU
 - f(x) = 0 x< 0 =x x>0
 - the gradient is easy to calculate
 - sigmoid will cause gradient vanishing

leaky ReLU
 - a small slop when x < 0, avoid gradient vanishing when x < 0

