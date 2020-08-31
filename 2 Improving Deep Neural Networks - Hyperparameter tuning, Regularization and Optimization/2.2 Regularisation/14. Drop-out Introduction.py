'''
3 - Dropout

Finally, dropout is a widely used regularization technique that is specific to deep learning.
It randomly shuts down some neurons in each iteration.
Watch these two videos to see what this means!


At each iteration, you shut down (= set to zero) each neuron of a layer with probability (1 − keep_prob) or keep it with probability keep_prob(50% here).
The dropped neurons don't contribute to the training in both the forward and backward propagations of the iteration.


1st layer: we shut down on average 40% of the neurons. 3rd layer: we shut down on average 20% of the neurons.

When you shut some neurons down, you actually modify your model.
The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons.
With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.
'''
