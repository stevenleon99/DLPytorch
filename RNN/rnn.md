the sentence can also be embedded
 - [word number, b, word vector] wn: number of words in the sentence, b: number of batch, wv: the dimension of the word
 - `nn.embedding(2,5) # 2 words in vocab, 5 dimensional embedded` 
 - word2vec or GloVe is the ready to use encoded look up table


folded model
 - ${x_t@w_{xh} + h_t@w_{hh}}$ xt is current input, wxh is the shared weight for the input, ht is the context initilized at starting point by [0 ....], whh is the shared weight for that
 - ${x_t@w_{xh} + h_t@w_{hh}}$ [3(batch), 100(dimension of a word)] @ [20(hidden layer), 100(feature length)]^T + [batch, hidden layer] @ [hidden length, hidden length]^T

 terminology
  - feature: the dimension for a word
  - number of features in the hidden state/memory shape h, how many dimension for h
  

gradient explosive and gradient vanishing
 - for the term ${\dfrac{\partial hi}{\partial hx}}$ is a cumulative mutiply term for 1RNN layer, if there are a lot of layers, and suppose W_hh > 1, the result will be infinite; while W_hh < 1 the result will be close to 0
 - gradient explosion: by diving the norm, so that we can keep the direction of gradient by decrease the magnitude (norm clipping algorithm)
 - gradient vanishing: LSTM (have a gating mechanism that controls the flow of information. These gates (input, output, and forget gates) can learn which data in a sequence is important to keep or throw away.)
