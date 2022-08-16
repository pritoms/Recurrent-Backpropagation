# Recurrent Backpropagation Networks

## Introduction

This paper discusses a new backpropagation algorithm for training recurrent neural networks (RNNs). This new algorithm is called Real-Time Recurrent Learning (RTRL) and was developed by Jürgen Schmidhuber. Schmidhuber claims that this new algorithm fixes some of the problems of the existing backpropagation through time (BPTT) algorithm. The key differences between these two algorithms are discussed in the next section.


## Context

Before going into the context of the paper, we will first set the stage for the discussion to follow.

### Neural Network Basics

Neural networks are typically discussed with regard to their structure. It is important to remember that neural networks are made up of individual neurons and the connections between them. The simplest neural network is a single neuron whose input is connected to a bias unit, which gives a constant input of 1. These input units are then connected to the neuron output unit. The activation function of this single neuron is the sigmoid function. The sigmoid function asymptotically approaches 0 and 1 as the input values approach -infinity and infinity, respectively. The graph of the sigmoid function is shown below:

![Sigmoid Function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

For this single neuron, the output is computed as follows:

$$f(x) = \frac{1}{1 + e^{-x}}$$

where 

$$x = w_1x_1 + w_2x_2 + w_3x_3 + \ldots + w_nx_n$$


### Recurrent Neural Networks (RNN)

The most commonly discussed RNNs at this time are LSTMs and GRUs. These are typically used in applications like image classification, speech recognition, and machine translation. RNNs have a special structure that allows them to process sequences of data. This is important for any problem that has an ordering to its data. For example, a sentence has an ordering to its words and this ordering cannot be ignored for a computer to understand the sentence. A traditional neural network does not have this property and cannot process the sentence in any meaningful way. An RNN can be used to solve this problem.

The key to RNNs is that they have a loop in their structure. Any input is passed through a hidden layer and the output of this hidden layer is also passed into the next iteration of the hidden layer. The output of the RNN is determined by the output of the last iteration of the hidden layer. This structure is shown below:

![RNN](rnn-unrolled.png)

The cell state and hidden state are both considered hidden states, but they can be computed by different means. The cell state is computed by a simple summation of weighted inputs, whereas the hidden state is computed using a nonlinear activation function. These two states are combined using a gating mechanism, which is described below.

### Gated Recurrent Units (GRU)

GRUs were introduced by Cho et. al in 2014 and are an improved version of LSTM. GRUs solve the problem that LSTM have of having too many parameters to be trained efficiently. GRUs also have a simpler structure than LSTMs and can be trained more easily. GRUs do not have an output gate, which is one of the main differences between GRUs and LSTMs.

GRUs have two gates: the update gate and the reset gate. The update gate is used to decide how much of the new information should be retained in the cell state. The reset gate is used to decide how much of the old cell state should be kept. These gates are computed using sigmoid activation functions and are shown below:

$$z_t = \sigma(W_zx_t + U_zh_{t-1} + b_z)$$
$$r_t = \sigma(W_rx_t + U_rh_{t-1} + b_r)$$

Once these gates are computed, they are used to determine the new cell state.

$$\hat{h} = tanh(Wx_t + r_t \odot Uh_{t-1})$$
$$h_t = (1 - z_t) \odot \hat{h} + z_t \odot h_{t-1}$$

The new cell state is then used to compute the output of the RNN.

### Long Short Term Memory Units (LSTM)

LSTMs were introduced by Hochreiter and Schmidhuber in 1997 and have been the most common RNN structure since their introduction. LSTMs are a special kind of RNN that are capable of learning long-term dependencies. This is important for a computer to understand text, video, or audio. LSTMs have been proven to be much better at these problems than traditional RNNs due to their ability to learn long-term dependencies. LSTMs are also more accurate than GRUs for these problems.

LSTMs have three gates: the forget gate, the input gate, and the output gate. The forget gate is used to determine how much of the previous cell state will be used in the next cell state. It is computed using the sigmoid activation function. The input gate is used to determine how much of the new information will be added to the cell state. It is also computed using the sigmoid activation function. The output gate is used to determine how much of the cell state will be used as the output of the LSTM. It is computed using the sigmoid activation function.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\hat{C}_t = tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t * C_{t-1} + i_t * \hat{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * tanh(C_t)$$

The cell state is then used to compute the output of the LSTM.

### Backpropagation Through Time (BPTT)

Backpropagation Through Time (BPTT) is an algorithm that can be used to train an RNN. BPTT is a special version of backpropagation that allows for training of RNNs. The key to BPTT is that it unrolls the RNN and treats the unrolled RNN as a single neural network. Because RNNs are cyclic, they cannot be trained with typical backpropagation algorithms, but BPTT can be used to overcome this problem. The unrolled RNN is shown below:

![Unrolled RNN](https://github.com/htbarnes/ml-and-dl-papers/raw/master/recurrent_backpropagation_networks/rnn-unrolled.png)

The BPTT algorithm is then used to train the unrolled RNN. We will discuss the BPTT algorithm in more detail in the next section.


## Paper Discussion

This paper discusses a new backpropagation algorithm for training recurrent neural networks (RNNs). This new algorithm is called Real-Time Recurrent Learning (RTRL) and was developed by Jürgen Schmidhuber. Schmidhuber claims that this new algorithm fixes some of the problems of the existing backpropagation through time (BPTT) algorithm. The key differences between these two algorithms are discussed in the next section.

### Key Differences

There are two main differences between RTRL and BPTT. The first difference is that RTRL computes the gradient for all time steps at once, whereas BPTT computes the gradient for each time step separately. The second difference is that RTRL uses truncated backpropagation with a special approach to truncating the gradient computation. This truncated backpropagation is used to reduce the memory requirements for RTRL.

BPTT works by computing the gradient for each time step separately and then summing these gradients together to get the gradient for the entire sequence. For example, if there are 10 time steps, then the gradients for each of these time steps are computed separately and then summed together. RTRL takes a different approach and computes the gradient for all 10 time steps at once. This leads to a much simpler computation because the RTRL algorithm only has to compute the gradient once instead of 10 times. This also leads to better results because it is an exact gradient computation instead of an approximation of the exact gradient.

One problem with RTRL is that it requires a lot of memory to store the gradient for the entire sequence. This problem is solved by truncated BPTT. Truncated BPTT works by limiting the number of time steps for which the gradient is computed. This is shown below:

![RTRL](https://github.com/htbarnes/ml-and-dl-papers/raw/master/recurrent_backpropagation_networks/rtrl.png)

The above figure shows that the gradient is computed for all time steps up to t and then the rest of the time steps are ignored. This allows for a much lower memory requirement for RTRL. The truncation of the gradient computation also leads to a better approximation of the exact gradient than BPTT. This is due to the fact that the gradient is still computed for all of the relevant time steps, but it is not computed for all of the irrelevant time steps.

Another advantage of RTRL is that it can be used to train higher order RNNs. Higher order RNNs are RNNs that have more than one hidden layer. BPTT cannot be used to train higher order RNNs, but RTRL can be used. This allows for more complex RNNs to be trained, which leads to better performance.

### Algorithm

The RTRL algorithm is shown below:

$$\begin{aligned}
& \frac{\partial E}{\partial W_{kj}^{(i)}} = \sum_{t=1}^T \sum_{l=1}^L \frac{\partial E}{\partial x_{kl}^{(i)}(t)} \frac{\partial x_{kl}^{(i)}(t)}{\partial W_{kj}^{(i)}} \\ 
& \frac{\partial E}{\partial b_k^{(i)}} = \sum_{t=1}^T \sum_{l=1}^L \frac{\partial E}{\partial x_{kl}^{(i)}(t)} \frac{\partial x_{kl}^{(i)}(t)}{\partial b_k^{(i)}} \\
& \frac{\partial E}{\partial x_{kl}^{(i)}(t)} = \sum_{j=1}^J \frac{\partial E}{\partial x_{kj}^{(i-1)}(t)} \frac{\partial x_{kj}^{(i-1)}(t)}{\partial x_{kl}^{(i)}(t)} + \frac{\partial E}{\partial x_{kl}^{(i)}(t+1)} \frac{\partial x_{kl}^{(i)}(t+1)}{\partial x_{kl}^{(i)}(t)} \\
& \frac{\partial E}{\partial x_{kj}^{(0)}(t)} = \sum_{l=1}^L \frac{\partial E}{\partial x_{kl}^{(1)}(t)} \frac{\partial x_{kl}^{(1)}(t)}{\partial x_{kj}^{(0)}(t)} \\
& \frac{\partial E}{\partial x_{kl}^{(i)}(t+1)} = \sum_{j=1}^J \frac{\partial E}{\partial x_{kj}^{(i)}(t+1)} \frac{\partial x_{kj}^{(i)}(t+1)}{\partial x_{kl}^{(i)}(t+1)}
\end{aligned}$$

This algorithm is used to compute the gradient for all of the weights and biases in the network. This algorithm is computed at each time step and then the gradients are summed over all time steps. This algorithm is also used to compute the gradient for higher order RNNs.

### Results

The results of this paper are summarized below:

- **RTRL is more efficient than BPTT.** RTRL is more efficient than BPTT because it computes the gradient for all time steps at once instead of computing the gradient for each time step separately. This allows for RTRL to train an RNN with a much smaller memory requirement.

- **RTRL is more accurate than BPTT.** RTRL is more accurate than BPTT because it computes the exact gradient instead of an approximation of the exact gradient. This leads to better performance for the RNN being trained by RTRL.

- **RTRL can be used to train higher order RNNs.** RTRL can be used to train higher order RNNs because it uses a different approach for computing the gradient. Higher order RNNs are RNNs that have more than one hidden layer. This leads to a more complex network that can perform better than a traditional RNN.

## Conclusion

Overall, I think this paper is a good introduction to the RTRL algorithm for training recurrent neural networks. The paper explains the problems with BPTT and how RTRL solves these problems. The paper also explains how RTRL can be used to train higher order RNNs. I think that this paper is a good example of how new algorithms can be developed to solve problems with existing algorithms. I also think that this paper is a good example of how new algorithms can be used to improve the performance of machine learning systems.
