underfitting
 - the estimated model cannot fully represent the ground truth
 - the estimated model is not enough complex
 - train and test accuracy are bad

overfitting
 - the estimated model is too enough
 - try to fitting the ground truth feature and also the noise
 - train accuracy is good but test accuracy is not good
 - it is also called the generalization performance

avoid overfitting
 - more data
 - constraint model complexity by shallow network, regularization
 - dropout
 - data argumentation
 - early stopping by validation set

regularization / weight decay
 - ${\lambda * \sum{|\theta_{i}|}}$
 - theta can be the weight, so the model will limit the magnitude of the coefficient of the weight to a lower value
 - like a power 7 polynomial, it will constrain the weight for high power to near zero
 - do not use it when the model is not overfitting 

 momentum
 - a plus of graident of param
 - the direction of plus is last gradient of param
 - like a momentum of history
 - 


 learning rate decay
 - 