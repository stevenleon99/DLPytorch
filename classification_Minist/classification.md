Cant directly used maximize accuracy
 - setting a threshold like 0.5, the weight lift the accuracy 0.4 -> 0.45, the overall accuracy is still 3/5
 - 0.45 < 0.5, so dy = 0, and graident of dw =0
 - or another extreme case is that accuracy from 0.499 to 0.5, but overall accuracy become 4/5 from 3/5, the 0.2/0.001 gradient is explosive

