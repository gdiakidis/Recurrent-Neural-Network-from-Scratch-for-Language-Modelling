# Recurrent-Neural-Network-for-Language-Modelling
We have implemented an RNN which we have trained by using backpropagation through time. Then we use this RNN to generate new sentences.

One exciting property of RNNs is that they can be used as sentence generators to statistically
generate new (unseen) sentences. Since the model effectively has learned a probability distribution over words Wn+1 for a given sequence [w1, ··· wn], we can generate new sequences. We can then apply the RNN forward and sample a new word wt+1 from the distribution
yˆ(t) at each time step. Then we feed this word in as input at the next step, and repeat until the model emits an end token.


## References
Jiang Guo. Backpropagation Through Time. Unpubl. ms., Harbin Institute of Technology,
2013.

Tomas Mikolov, Martin Karafiat, Lukas Burget, Jan Cernock ´ y, and Sanjeev Khudan- `
pur. Recurrent neural network based language model. In INTERSPEECH, volume 2,
page 3, 2010.
