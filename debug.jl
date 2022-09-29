### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 2eb50920-380d-11ed-1bf5-291d89f61e84
using AutoGrad, GLMakie, StatsBase, LinearAlgebra, GeometryTypes, DataStructures

# ╔═╡ f4ce722c-49d7-41a2-813b-6a3eaebc3d4b
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

# ╔═╡ 6b83f78b-13db-4d2c-a79f-98d1319fb320
# begin
# 	y = @diff ψ(rand(M))
# 	grad.([y], params(u))
# end

# ╔═╡ 32f46fee-6bd0-4cbb-b4d5-f6b7e38214f5
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
end

# ╔═╡ 52e117e7-b8af-46f7-9fda-1d2e52e316d6
begin
	N, M = 5, 5
	Nh = 10
	u = Chain(Dense(M, Nh, tanh), Dense(Nh, 2))
	ψ(n) = exp(sum(u(n) .* [1, 1im]))

	basis = generate_basis(5, 5)
end;

# ╔═╡ 8c9f6a1c-7a04-4c2b-a2b6-fc2722702432
begin
	function hop(state, i, j, N)
		state_final = copy(state)
		state_final[i] += 1
		state_final[j] -= 1
		coeff = (state[i] > N || state[j] == 0) ? 0. : sqrt((state[i] + 1) * state[j])

		return coeff, state_final
	end

	function num(state, i, f = identity)
		return f(state[i]), copy(state)
	end
	
	function hamiltonian(state, t, U, mu, L, N)
		res = Dict{Vector{Int16}, Float64}()
		for i in 1:L
			j = mod1(i, L)

			coeff, state_final = hop(state, i, j, N)	
			res[state_final] = get(res, state_final, 0.) - t * coeff
			
			coeff, state_final = hop(state, j, i, N)	
			res[state_final] = get(res, state_final, 0.) - t * coeff

			coeff, state_final = num(state, i, (n) -> n * (n - 1))
			res[state_final] = get(res, state_final, 0.) + 0.5 * U * coeff

			coeff, state_final = num(state, i)
			res[state_final] = get(res, state_final, 0.) - mu * coeff
		end

		return res
	end

	sample_state(basis) = basis[rand(1:size(basis)[1]), :]
	
	function expectationMC(ψ, op, basis, network, rtol = 1e-3)
		val, val_old = 0., 0.
		state = sample_state(basis)
	
		while(true)
			new_state = sample_state(basis)
			
			if (rand() < abs.(ψ(new_state)/ψ(state)) ^ 2)
				state = new_state
				
				for (state_final, coeff) in op(state)
					val += coeff * ψ(state_final)
				end
		
				val /= ψ(state)
				if ((val - val_old) < rtol * val_old) break end
				val_old = val
			end
		end

		return val
	end

	function Ow_energy(ψ, basis, network, rtol = 1e-3)
		
		val = zeros(ComplexF64, size(network))
		val_old = zeros(ComplexF64, size(network))
			
		state = sample_state(basis)
	
		while(true)
			new_state = sample_state(basis)
			
			if (rand() < abs.(ψ(new_state)/ψ(state)) ^ 2)
				state = new_state
				
				for (state_final, coeff) in hamiltonian(state, 0.15, 1., 1.5, 5, 5)
					val .+= coeff * ψ(state_final)
				end
				
				deriv = @diff ψ(state)
				Ow = vcat(vec.(grad.([deriv], params(network)))...) ./ ψ(state)
					
				val .*= conj.(Ow)
		
				if (all(abs.(val .- val_old) .< rtol .* abs.(val_old))) break end
				val_old .= val
			end
		end

		return val	
	end

	function Ow(ψ, basis, network, rtol = 1e-3)
		val = zeros(ComplexF64, size(network))
		
		state = sample_state(basis)
	

        fig = Figure(); display(fig)
        ax = Axis(fig[1,1])
        hist = Observable([Point2f0(0, norm(val))])
		hist2 = CircularBuffer{Float64}(200)
        lines!(ax, hist; linewidth = 4, color = :purple)

        n_iter = 0

		while(true)
			new_state = sample_state(basis)
			
			if (rand() < abs.(ψ(new_state)/ψ(state)) ^ 2)
				state = new_state
				
				deriv = @diff ψ(state)
				Ow = vcat(vec.(grad.([deriv], params(network)))...) ./ ψ(state)
				
				val .+= conj.(Ow)
				push!(hist2, norm(val))
				if (n_iter >= 200 && (abs(mean(hist2[1:100]) - mean(hist2[101:200])) < rtol * mean(hist2[1:100]))) break end
                n_iter += 1
                
                if n_iter % 100 == 0
                    push!(hist[], Point2f0(n_iter, norm(val)))
                    hist[] = hist[]
                    ylims!(ax, 0, maximum(getindex.(hist[], 2)))
                    xlims!(ax, 0, n_iter)
                    sleep(0.00001)
                end
			end
		end

		return hist
	end

	function energy_grad(ψ, basis, network, rtol = 1e-3)
		2 .* real.(Ow_energy(ψ, basis, network, rtol) .- Ow(ψ, basis, network, rtol) .* expectationMC(ψ, hamiltonian, basis, network, rtol))
	end
end

# ╔═╡ f0dc3e36-5f2c-4259-bdd0-fca41bdad42f
# Ow_energy(ψ, basis, u, 1e-3)
