Ctrl Shft + V for preview

Cross entropy loss
 - use together with softmax
 - can use for bi-classify or multi-classify
 - entropy = - ${\sum_{i}(P(i)logP(i))}$
 - can be derive to KL divergence, the difference of two distribution
 - for hot hot encoding, the H(p, q) = $D_{KL}(p|q)$
 - binary classification: H(P, Q) = - p(cat)log(Q(cat))-(1-P(cat))log(1-Q(cat)); 1-P(cat) = P(dog); P is prob by truth Q is prob by pred
 - multi classification: - ${\sum_{i}(P(i)logP(i))}$, H(P, Q) smaller and the pred getting closer to truth
 - in comparison with sigmoid which is easy to lead to gradient vanish, but the cross entropy by the sigmoid can lead to larger graident

Hinge loss
 - used in SVM
 - 
 