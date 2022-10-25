# Solving the Bose Hubbard Model using Machine Learning

This is an implementation of a variational quantum monte carlo technique to solve the Bose Hubbard Model (and possibly its extensions) as explained in [arXiv:1707.09723](https://arxiv.org/abs/1707.09723). 

![](./ANN.svg)

It utilizes a feed-forward neural network as a variational ansatz for the wavefunction of the system. The ground state wavefunction can be determined by training the network parameters to minimize the expectation value of the hamiltonian (energy). This method can serve as an alternative to exact diagonalization as it reduces the memory required to specify the wavefunction by exploiting any structure in the co-efficients to store it more efficiently (in the form of a neural network).

## Basic usage

### Setting up the neural network:

```julia
num_bosons, num_lattice_sites = 5, 5
num_hidden_layers = 10

# chain together dense layers to construct a feed-forward neural network
network = Chain(Dense(num_lattice_sites, num_hidden_layers, tanh), Dense(num_hidden_layers, 2)) 

# obtain wavefunction coefficients from the network
psi(n) = exp(sum(network(n) .* [1, 1im]))

# enumerate occupation basis set
basis = generate_basis(num_bosons, num_lattice_sites)
```
Alternatively, you can use the helper function `main_init` like so:

```julia
num_bosons, num_lattice_sites, num_hidden_layers = 5, 5, 10
psi, network, basis = main_init(num_bosons, num_lattice_sites, num_hidden_layers)
```

### Calculate gradient of the wavefunction w.r.t weights
```julia
tmp = @diff psi(rand(num_lattice_sites))
deriv = grad.([tmp], params(network))
```
This is performed using the convenient macro provided by `AutoGrad.jl`. You will likely never require this; it is just to document the implementation inside the MC loop for calculating the energy gradient.

### Calculate the expectation value of any observable (by monte carlo sampling):

```julia
expectationMC(psi, op, L, N, basis, network, rtol, atol, window_size; callback, kwargs...)
```

Currently, there is no way to sample the basis set without enumerating it completely first (forcing the requirement of the `basis` argument which is a list of all basis elements). The `generate_basis(num_bosons, num_lattice_sites)` function can be used for this purpose.

To determine the convergence of the MC average, we track the last `window_size` values of the sampling. `rtol` and `atol` are required to specify the tolerance of statistical errors in the result.

The operator `op` must be a function that takes in a vector (`state`, an element of the basis set), num_bosons, `N` and num_lattice_sites, `L`. It must return a collection of tuples `(state_final, coefficient)` obtained by the application of the operator on the state. Any parameters required by the operator are passed through the keyword arguments, `kwargs`. 

#### Hamiltonian
$$H = -t\sum_{\langle i, j, \rangle} a_i^{\dagger}a_j + \frac{U}{2}\sum_i n_i (n_i - 1) - \mu \sum_i n_i$$
`hamiltonian(state, L, N; t, U, mu)`

#### Hopping parameter
$$\text{Hop(i, j)} = a_i^{\dagger}a_j$$
`hop(state, L, N; i, j)`

#### Number operator
$$Number(f, i) = f(a_i^{\dagger}a_i) = f(n_i)$$
`num(state, i, f = identity)`

### Calculate energy gradient

```julia
Ow_energy(psi, L, N, basis, network, rtol, atol, window_size; callback, kwargs...)
```

Same arguments as discussed in the previous section. However, `kwargs` must specifically contain the parameters required to construct the hamiltonian (since it calculates the *energy* gradient).

### Training the network

```julia
n_iter = 10
train!(psi, N, L, basis, network; callback = progress_bar_init(n_iter, N, L, basis), n_iter = n_iter, gamma = 0.05, t = 0.01, mu = 0.5, U = 1)
```

This method implements a basic gradient descent update for minimizing the energy of the system. `gamma` is the learning rate and `n_iter` is the number of gradient updates to be performed. This function mutates the network and wavefunction provided as input.

### Callback functions

All the MC loops have the provision of calling a callback function every `n` iterations (`n=100` currently). This can be useful if you want to track some other quantity or perform an action periodically while the MC procedure is running. The function must take an input tuple `(expectations, n_iter)` which specifies the state of the simulation at that moment. `expectations` carries the last `window_size` results of the MC sampling and `n_iter` is the iteration number at the moment the callback is invoked.

A `debug_plot_init` function is provided out of the box that plots the MC values in real-time (using `Makie.jl`) as the loop is running which can be useful for debugging the code. Additionally, a `progress_bar_init` is also provided to print a progress bar (using `ProgressMeter.jl`) to the terminal, along with the energy of the wavefunction after each iteration of the training loop.