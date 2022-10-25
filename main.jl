using AutoGrad, StatsBase, LinearAlgebra, DataStructures, ProgressMeter, GLMakie

# Construct neural network structure 
begin
	struct Dense
		w # Weights
		b # Bias
		f # Activation
		Dense(i::Int, o::Int, f = identity) = new(Param(randn(o, i)), Param(randn(o)), f)
	end

	Base.size(d::Dense) = length(d.w) + length(d.b)
	(d::Dense)(x) = d.f.(d.w * x .+ d.b)

	struct Chain
		layers
		Chain(args...) = new(args)
	end

	Base.size(c::Chain) = reduce((p, l) -> p + size(l), c.layers, init = 0)
	(c::Chain)(inp) = reduce((x, f) -> f(x), c.layers, init = inp)

	# To extract parameters
	params(d::Dense) = [d.w, d.b]
	params(c::Chain) = reduce((p, l) -> vcat(p, params(l)), c.layers, init = [])

	# To flatten parameters
	params_list(d::Dense) = vcat(vec(d.w), d.b)
	params_list(c::Chain) = reduce((p, l) -> vcat(p, params_list(l)), c.layers, init = Float64[])

	# To un-flatten parameters
	function reconstruct_params(d::Dense, params_list)
		[reshape(params_list[1:prod(length(d.w))], size(d.w)), params_list[(end - prod(length(d.b)) + 1):end]]
	end

	function reconstruct_params(c::Chain, params_list)
		params = []
		inds = pushfirst!(cumsum([length(l.w) + length(l.b) for l in c.layers]), 0)
		for (i, layer) in enumerate(c.layers)
			params = vcat(params, reconstruct_params(layer, params_list[(inds[i] + 1):inds[i+1]]))
		end

		return params
	end
end;

# Enumerate occupation basis set for N bosons on L lattice sites
begin
	hilbert_dim(k, n) = binomial(k + n - 1, n - 1)
	
	function generate_basis(N, L)
	    if L > 1
	        basis = zeros(Int16, (hilbert_dim(N, L), L))
	        j = 1
	    
	        for n in 0:N
	            d = hilbert_dim(n, L - 1)
	            basis[j:(j + d - 1), 1] .= (N - n)
	            basis[j:(j + d - 1), 2:end] = generate_basis(n, L - 1)
	            j += d
	        end
	        
	    else
	        basis = [N]
	    end
	
	    return basis
	end

	function rms(arr)
		avg = mean(arr)
		return sqrt(sum((arr .- avg).^2)/(length(arr) - 1))
	end
end

# Operator definitions; they must return a (collection of) tuples (state_final, coefficient) after operating on a state.
begin
	# aᵢ† aⱼ with clamps on i > N and j < 0		
	function hop(state, L, N; i, j)
		state_final = copy(state)
		state_final[i] += 1
		state_final[j] -= 1
		coeff = (state[i] >= N || state[j] == 0) ? 0. : sqrt((state[i] + 1) * state[j])

		return [(state_final, coeff)]
	end

	# f(n̂) - arbitrary functions of number operator
	function num(state, i, f = identity)
		return [(copy(state), f(state[i]))]
	end
	
	function hamiltonian(state, L, N; t, U, mu)
		res = Dict{Vector{Int16}, Float64}()
		for i in 1:L
			j = mod1(i, L)

			state_final, coeff = hop(state, L, N; i = i, j = j)[1]
			res[state_final] = get(res, state_final, 0.) - t * coeff
			
			state_final, coeff = hop(state, L, N; i = j, j = i)[1]
			res[state_final] = get(res, state_final, 0.) - t * coeff

			state_final, coeff = num(state, i, (n) -> n * (n - 1))[1]
			res[state_final] = get(res, state_final, 0.) + 0.5 * U * coeff

			state_final, coeff = num(state, i)[1]
			res[state_final] = get(res, state_final, 0.) - mu * coeff
		end

		return res
	end
end

begin
	# uniformly sample from enumerated basis set 
	sample_state(basis) = basis[rand(1:size(basis)[1]), :]
	
	# Monte Carlo expectation for arbitrary operator. Pass operator dependant parameters in kwargs.
	function expectationMC(psi, op, L, N, basis, network, rtol = 1e-3, atol = 1e-6, window_size = 1000; callback = identity, kwargs...)

		res = 0. + 0im # result (summing over contributions of the MC chain, but not averaged yet)

		state = sample_state(basis) # initial sample
		expectations = CircularBuffer{ComplexF64}(window_size) # keeps track of history of MC results for last 'window_size' sweeps
		n_iter = 0
	
		while(true)
			n_iter += 1

			# propose new state
			new_state = sample_state(basis)
			
			# accept based on weights 
			if (rand() < abs.(psi(new_state)/psi(state)) ^ 2)
				state = new_state
			end

			# calculate contribution of new state to the result; op * |state> = coeff * |state_op>
			for (state_op, coeff) in op(state, L, N; kwargs...)
				res += coeff * psi(state_op)/psi(state)
			end
	
			# push the result to hist (divided by n_iter to calculate Monte Carlo average)
			push!(expectations, res/n_iter)
			
			# terminate MC when the result has converged within a window
			if(isfull(expectations) && (rms(abs.(expectations)) < (atol + rtol * mean(abs.(expectations))))) break end

			# callback function to perform any task periodically (e.g. plotting)
			if n_iter % 100 == 0 callback([expectations, n_iter]) end
		end

		return mean(expectations)
	end
end

# Debug plotting callback function 
function debug_plot_init()
	fig = Figure(); display(fig)
	ax = Axis(fig[1,1])
	hist_plot = Observable([Point2f0(0, 0)])
	hist_mean = Observable([0.])
	ax.xlabel = "Number of iterations"
	ax.ylabel = "MC average"

	hlines!(ax, hist_mean; linewidth = 2.5, color = :black, linestyle = :dash)
	lines!(ax, hist_plot; linewidth = 4, color = :purple)

	function debug_plot(state)
		expectations, n_iter = state
		# update time series
		push!(hist_plot[], Point2f0(n_iter, abs(expectations[end])))
		if (length(hist_plot[]) > 500)
			popfirst!(hist_plot[])
		end
		hist_plot[] = hist_plot[]

		# update mean value
		hist_mean[] = [abs(mean(expectations))]

		# update axis lims
		ylims!(ax, minimum(getindex.(hist_plot[], 2)), maximum(getindex.(hist_plot[], 2)))
		xlims!(ax, minimum(getindex.(hist_plot[], 1)), maximum(getindex.(hist_plot[], 1)))
		sleep(0.00001)
	end

	return debug_plot
end

## Other expectations needed to compute energy gradient

function Ow_energy(psi, L, N, basis, network, rtol = 1e-3, atol = 1e-6, window_size = 1000; callback = identity, kwargs...)
	
	res = zeros(ComplexF64, size(network))
	state = sample_state(basis)
	expectations = CircularBuffer{Vector{ComplexF64}}(window_size)
	n_iter = 0

	tmp_res = copy(res)

	while(true)
		tmp_res = zeros(ComplexF64, size(network))
		n_iter += 1
		new_state = sample_state(basis)
		
		if (rand() < abs.(psi(new_state)/psi(state)) ^ 2)
			state = new_state
		end

		for (state_op, coeff) in hamiltonian(state, L, N; kwargs...)
			tmp_res .+= coeff * psi(state_op)/psi(state)
		end
		
		deriv = @diff psi(state)
		Ow = vcat(vec.(grad.([deriv], params(network)))...) ./ psi(state)
		tmp_res .*= conj.(Ow)

		res .+= tmp_res

		push!(expectations, res ./ n_iter)

		if(isfull(expectations) && (rms(norm.(expectations)) < (atol .+ rtol * mean(norm.(expectations))))) break end

		if n_iter % 100 == 0 callback([norm.(expectations), n_iter]) end
	end

	return [mean([expectations[i][j] for i in 1:length(expectations)]) for j in 1:length(expectations[1])]
end

function Ow(psi, basis, network, rtol = 1e-3, atol = 1e-6, window_size = 1000; callback = identity)
	res = zeros(ComplexF64, size(network))
	state = sample_state(basis)
	expectations = CircularBuffer{Vector{ComplexF64}}(window_size)
	n_iter = 0

	while(true)
		n_iter += 1

		new_state = sample_state(basis)
		
		if (rand() < abs.(psi(new_state)/psi(state)) ^ 2)
			state = new_state
		end

		deriv = @diff psi(state)
		Ow = vcat(vec.(grad.([deriv], params(network)))...) ./ psi(state)
		
		res .+= conj.(Ow)
		push!(expectations, (res) ./ n_iter)
		
		if(isfull(expectations) && (rms(norm.(expectations)) < (atol .+ rtol * mean(norm.(expectations))))) break end
	end

	return [mean([expectations[i][j] for i in 1:length(expectations)]) for j in 1:length(expectations[1])]
end

function energy_grad(psi, L, N, basis, network, rtol = 1e-3, atol = 1e-6, window_size = 1000; kwargs...)
	2 .* real.(Ow_energy(psi, L, N, basis, network, rtol, atol, window_size; kwargs...) .- (Ow(psi, basis, network, rtol, atol, window_size) .* expectationMC(psi, hamiltonian, L, N, basis, network, rtol, atol, window_size; kwargs...)))
end

# Mock network + wavefunction for testing
begin
	N, L = 5, 5
	Nh = 10
	u = Chain(Dense(L, Nh, tanh), Dense(Nh, 2))
	psi(n) = exp(sum(u(n) .* [1, 1im]))

	basis = generate_basis(5, 5)
	() # to suppress terminal output 

	# for some expectation values run: (remove callback arg if you dont need real-time plotting)
	# expectationMC(psi, hamiltonian, 5, 5, basis, u, 1e-3, 1e-5, 1000; callback = debug_plot_init(), t = 0.01, mu = 0.5, U = 1)
	# or 
	# expectationMC(psi, hop, 5, 5, basis, u, 1e-3, 1e-5, 1000; callback = debug_plot_init(),i = 3, j = 4)
	# or
	# Ow(psi, basis, u, 1e-3, 1e-5, 1000; callback = debug_plot_init())
	# or 
	# Ow_energy(psi, 5, 5, basis, u, 1e-3, 1e-5, 1000; callback = debug_plot_init(), t = 0.01, mu = 0.5, U = 1)
end;

function gradient_descent(psi, L, N, basis, network, rtol = 1e-3, atol = 1e-6, window_size = 1000; callback = nothing, show_progress = true, gamma = 0.05, n_iter = 10, kwargs...)
    w = params_list(network)

    for i in 1:n_iter
        w .-= gamma * energy_grad(psi, L, N, basis, network, rtol, atol, window_size; kwargs...)

		if !isnothing(callback) callback([psi, network, i]; kwargs...) end
	end

	for (old_param, new_param) in zip(params(network), reconstruct_params(network, w))
		old_param .= new_param
	end

    return psi
end

function progress_bar_init(n_iter)
	p = Progress(n_iter; showspeed=true)

	function progress_bar(state; kwargs...)
		psi, u, i = state
		energy = expectationMC(psi, hamiltonian, N, l, basis, u, 1e-3, 1e-5, 1000; kwargs...)
		ProgressMeter.next!(p; showvalues = [(:iter, i), (:energy, energy)])
	end
end