### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ 1a4bd574-b4d3-488e-a697-26e8e77b76de
using AutoGrad, StatsBase, LinearAlgebra, DataStructures, GLMakie, GeometryTypes
using AbstractPlotting, AbstractPlotting.MakieLayout

# ╔═╡ aad5e746-7bf7-45d9-a9cc-f18787c4ca78
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

# ╔═╡ 21a461b9-89ed-4abc-9ad9-b73d31f09d4b
# begin
# 	y = @diff ψ(rand(M))
# 	grad.([y], params(u))
# end

# ╔═╡ 79af6eac-481b-4237-9a47-45c617afe042
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

# ╔═╡ 9f242029-4785-4fad-8d9c-d20ef64ff1e1
begin
	N, M = 5, 5
	Nh = 10
	u = Chain(Dense(M, Nh, tanh), Dense(Nh, 2))
	ψ(n) = exp(sum(u(n) .* [1, 1im]))

	basis = generate_basis(5, 5)
end;

# ╔═╡ f83e6652-90c7-4510-bcca-8fd6f2c51404
begin
	function hop(state, L, N; i, j)
		state_final = copy(state)
		state_final[i] += 1
		state_final[j] -= 1
		coeff = (state[i] >= N || state[j] == 0) ? 0. : sqrt((state[i] + 1) * state[j])

		return coeff, state_final
	end

	function num(state, i, f = identity)
		return f(state[i]), copy(state)
	end
	
	function hamiltonian(state, L, N; t, U, mu)
		res = Dict{Vector{Int16}, Float64}()
		for i in 1:L
			j = mod1(i, L)

			coeff, state_final = hop(state, L, N; i = i, j = j)	
			res[state_final] = get(res, state_final, 0.) - t * coeff
			
			coeff, state_final = hop(state, L, N; i = j, j = i)	
			res[state_final] = get(res, state_final, 0.) - t * coeff

			coeff, state_final = num(state, i, (n) -> n * (n - 1))
			res[state_final] = get(res, state_final, 0.) + 0.5 * U * coeff

			coeff, state_final = num(state, i)
			res[state_final] = get(res, state_final, 0.) - mu * coeff
		end

		return res
	end

	sample_state(basis) = basis[rand(1:size(basis)[1]), :]
	
	function expectationMC(ψ, op, L, N, basis, network, atol = 1e-3, window_size = 1000; kwargs...)
		res = 0. + 0im
		state = sample_state(basis)
		hist = CircularBuffer{ComplexF64}(window_size)
	
		###
		scene, layout = layoutscene(resolution = (1200, 900))
		ax = layout[1, 1] = LAxis(scene)
		sl1 = layout[2, 1] = LSlider(scene, range = 0:0.01:0.1, startvalue = 0.1)

		fig = Figure(); display(fig)
        ax = Axis(fig[1,1])
        hist2 = Observable([Point2f0(0, abs(res))])
		lines!(ax, hist2; linewidth = 4, color = :purple)
		n_iter = 0
		# debug = Observable("0")
		# text!(campixel(ax.scene), debug, space = :data)
		###
	
		while(true)
			new_state = sample_state(basis)
			
			if (rand() < abs.(ψ(new_state)/ψ(state)) ^ 2)
				state = new_state
			end

			for (state_final, coeff) in op(state, L, N; kwargs...)
				res += coeff * ψ(state_final)
			end
	
			res /= ψ(state)

			push!(hist, res/(n_iter + 1))
			
			if(isfull(hist) && (rms(abs.(hist))/mean(abs.(hist)) < atol)) break end


			n_iter += 1
			### 
			if n_iter % 100 == 0
				push!(hist2[], Point2f0(n_iter, abs(hist[end])))
				hist2[] = hist2[]
				ylims!(ax, 0, ymax)
				xlims!(ax, 0, n_iter)
				sleep(0.00001)
			end
			###
		end

		return mean(hist), n_iter
	end

	function Ow_energy(ψ, L, N, basis, network, atol = 1e-3, window_size = 1000; kwargs...)
		
		res = zeros(ComplexF64, size(network))
		state = sample_state(basis)
		hist = CircularBuffer{Vector{ComplexF64}}(window_size)
		
		while(true)
			new_state = sample_state(basis)
			
			if (rand() < abs.(ψ(new_state)/ψ(state)) ^ 2)
				state = new_state
			end

			for (state_final, coeff) in hamiltonian(state, L, N; kwargs...)
				res .+= coeff * ψ(state_final)
			end
			
			deriv = @diff ψ(state)
			Ow = vcat(vec.(grad.([deriv], params(network)))...) ./ ψ(state)
				
			res .*= conj.(Ow)
			push!(hist, (res)./(length(hist) + 1))

			if(isfull(hist) && rms(norm.(hist)) < atol) break end
		end

		return [mean([hist[i][j] for i in 1:length(hist)]) for j in 1:length(hist[1])]
	end

	function Ow(ψ, basis, network, atol = 1e-3, window_size = 1000)
		res = zeros(ComplexF64, size(network))
		state = sample_state(basis)
		hist = CircularBuffer{Vector{ComplexF64}}(window_size)

		while(true)
			new_state = sample_state(basis)
			
			if (rand() < abs.(ψ(new_state)/ψ(state)) ^ 2)
				state = new_state
			end

			deriv = @diff ψ(state)
			Ow = vcat(vec.(grad.([deriv], params(network)))...) ./ ψ(state)
			
			res .+= conj.(Ow)
			push!(hist, (res)./(length(hist) + 1))
			
			if(isfull(hist) && rms(norm.(hist)) < atol) break end

    	end

		return [mean([hist[i][j] for i in 1:length(hist)]) for j in 1:length(hist[1])]
	end

	function energy_grad(ψ, L, N, basis, network, rtol = 1e-3; kwargs...)
		2 .* real.(Ow_energy(ψ, L, N, basis, network, rtol; kwargs...) .- Ow(ψ, basis, network, rtol) .* expectationMC(ψ, hamiltonian, L, N, basis, network, rtol); kwargs...)
	end
end

# ╔═╡ 4bbc4155-f6e3-4daa-8eb2-a234cdb8c0b4
# hist = energy_grad(ψ, 5, 5, basis, u, 1e-3; t = 0.5, mu = 1.5, U = 1.)

# ╔═╡ df166181-e284-41b2-844b-c30c1694d6b3
# ╠═╡ disabled = true
#=╠═╡
Ow(ψ, basis, u, 1e-2)
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AutoGrad = "6710c13c-97f1-543f-91c5-74e8f7d95b35"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AutoGrad = "~1.2.5"
DataStructures = "~0.18.13"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "533c239cced097407f6d07a5752bafd0035beb22"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AutoGrad]]
deps = ["Libdl", "LinearAlgebra", "SpecialFunctions", "Statistics", "TimerOutputs"]
git-tree-sha1 = "0f97c20e513d18e9c0d8f58e1b6eb0e288249832"
uuid = "6710c13c-97f1-543f-91c5-74e8f7d95b35"
version = "1.2.5"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.DataAPI]]
git-tree-sha1 = "1106fa7e1256b402a86a8e7b15c00c85036fef49"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.11.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "9dfcb767e17b0849d6aaf85997c98a5aea292513"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.21"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═1a4bd574-b4d3-488e-a697-26e8e77b76de
# ╠═aad5e746-7bf7-45d9-a9cc-f18787c4ca78
# ╠═21a461b9-89ed-4abc-9ad9-b73d31f09d4b
# ╠═79af6eac-481b-4237-9a47-45c617afe042
# ╠═9f242029-4785-4fad-8d9c-d20ef64ff1e1
# ╠═f83e6652-90c7-4510-bcca-8fd6f2c51404
# ╠═4bbc4155-f6e3-4daa-8eb2-a234cdb8c0b4
# ╠═df166181-e284-41b2-844b-c30c1694d6b3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
