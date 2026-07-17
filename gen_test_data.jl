begin
	using StringDistances
	using LinearAlgebra
	using Random
	using Yao
	using Ket
	using Chain
	using Serialization
	using RandomMatrices
	using NPZ	
end

include("utils.jl")

const cuda_available = false
const cuda_available = is_package_installed("CUDA")
if cuda_available 
	using CUDA
	# CUDA.allowscalar(true)
end

begin		
	ZZMax = matblock(exp(-im*pi/4)*Diagonal([1,im,im,1]))
	⊗ = kron 
	cx = cnot
	LD = Levenshtein() # function that computes edit distance
end

function angular2_design(nbit, nlayer) # returns a circuit 
	# total params = 2*nbit*nlayer

	@assert nbit%2 == 0

    circuit = chain(nbit)
	ent_chain = [
		chain(put((i, i+1)=>ZZMax) for i in 1:2:nbit-1),
		chain(put((i+1,(i+1)%nbit+1)=>ZZMax) for i in 1:2:nbit-1),
    ]

	push!(circuit, chain(put(i=>Rx(0.0)*Ry(0.0)) for i in 1:nbit))
	# push!(circuit, chain(put(i=>Ry(0.0)) for i in 1:nbit))
	# push!(circuit, chain(put(i=>Rx(0.0)) for i in 1:nbit))
	
	for k in 1:nlayer-1
		push!(circuit, ent_chain[(k-1)%2+1]) # add 2-qubit layer
		push!(circuit, chain(put(i=>H) for i in 1:nbit))
		push!(circuit, chain(put(i=>Rx(0.0)*Ry(0.0)) for i in 1:nbit))
		# push!(circuit, chain(put(i=>Ry(0.0)) for i in 1:nbit))
		# push!(circuit, chain(put(i=>Rx(0.0)) for i in 1:nbit))
    end

    return circuit
end

function angular2_cost(nbit, nlayer) # Quantinuum cost of angular2_design (numerator)
	N1q = nbit*(nlayer+1)
	N2q = nbit*nlayer÷2
	Nm = 2 # or 2*bit? 
	num = N1q + 10N2q + 5Nm # numerator
	return num

	# angular2_cost(...)/5000*100+5 if 100 shots
end

# angular2_design(4, 4) |> vizcircuit

begin
	struct RopeEncoder 
		# Total output state has 2s+q+1 qubits. Does not produce a circuit.
		
		q::Int64 # target number of qubits per each s-mer
		s::Int64 # size of s-mers to consider
		N::Int64 # baseline angle denominator, should coincide with k-mer length k
		d::Vector{ComplexF64} # exponents
		
		function RopeEncoder(;q=q,s=s,N=N)
			d = [exp(2pi*im/N*i) for i=1:2^q]
			return new(q,s,N,d)
		end
	end
	
	function (self::RopeEncoder)(dna) 
		# returns normalized state. dna = [4,2,3,1...]
		
		@assert length(dna)==self.N # N should match k-mer length
		# embeds = [zeros(ComplexF64, 2^self.q) for i = 1:4^self.s]

		nthr = nthreads()
		p = [self.N/nthr*i for i=1:nthr-1] .|> floor .|> Int
		# rs = [1:p[1], p[1]+1:p[2], p[2]+1:p[3], p[3]+1:self.N-self.s+1]
		insert!(p,1,0)
		push!(p, self.N-self.s+1)
		rs = [p[i]+1:p[i+1] for i=1:nthr]		

		# embeds_parts = Array{Any}(undef, nthr) 
		embeds_parts = [[zeros(ComplexF64, 2^self.q) for i = 1:4^self.s] for j=1:nthr]
		@threads for i=1:nthr
			embeds_parts[i] = self(dna, rs[i])
		end
		embeds = sum(embeds_parts)

		# embeds = self(dna, 1:self.N-self.s+1)
	
		embed = vcat(embeds...) # concatenate
		embed = vcat(real.(embed), imag.(embed))
		embed /= norm(embed)
		return embed
		# return embeds
	end

	function (self::RopeEncoder)(dna, range::UnitRange{Int}) 
		# returns normalized state. dna = [4,2,3,1...]
		
		@assert length(dna)==self.N # N should match k-mer length
		embeds = [zeros(ComplexF64, 2^self.q) for i = 1:4^self.s]
		# s4 = 4^(self.s-1)
		shft = 2*(self.s-1)
		
		ind = reduce((part, digit) -> 4*part+digit-1, dna[range.start-1+self.s:-1:range.start]; init=0) # convert sequence of base-4 digits to a number	
		di = [exp((range.start-1)*2pi*im/self.N*i) for i=1:2^self.q]
		# di = [exp(BigFloat(range.start-1)*2pi*im/self.N*i) for i=1:2^self.q]
		for i = range.start+self.s:range.stop+self.s-1
			di .*= self.d
			@inbounds embeds[ind+1] .+= di 
			@inbounds ind = ((dna[i]-1) << shft) + (ind >> 2) 
		end
		begin
			di .*= self.d
			embeds[ind+1] += di
		end
	
		# embed = vcat(embeds...) # concatenate
		# embed = vcat(real.(embed), imag.(embed))
		# embed /= norm(embed)
		# return embed
		return embeds
	end
end

begin
	struct Angular2Encoder
		# produces encoding circuit given a state 
		
		n::Int64 # number of parameters (size of the input state)
		q::Int64 # target number of qubits 
		scale::Float64 # scaling factor (for the input state)
		circuit::ChainBlock{2} # circuit structure with params placeholders
		
		function Angular2Encoder(;n=n, q=q, scale=scale)
			# nlayer = n/q/2 |> ceil |> Int
			nlayer = n/q/2 |> floor |> Int
			ntail = n - 2*q*nlayer
			circuit = angular2_design(q, nlayer)			
			push!(circuit, chain(put(i=>Rx(0.0)*Ry(0.0)) for i in 1:ntail÷2))
			return new(n, q, scale, circuit)
		end
	end
	
	function (self::Angular2Encoder)(params)
		@assert length(params)==self.n

		# params_fixed = vcat(params*self.scale, zeros(nparameters(self.circuit)-self.n))
		# m = nparameters(self.circuit)
		# params_fixed = params[1:m]*self.scale # discard tail params (todo: make a better method)
		params_fixed = params*self.scale 
		
		circuit = dispatch(self.circuit, params_fixed) # copy of a circuit with dispatched params
		return circuit
	end
end

function generate_kmer_pairs(; k=1000, n=100, buf=1, max_err=0.3, sparsity=1, rng=Random.default_rng()) # uses Channel for sync
	# out_ch = Channel{Int}(n)
	out_ch = Channel{Int}(buf)
	# kmer_pairs = Array{Any}(undef, n)
	kmer_pairs = Array{Union{Tuple{Vector{UInt8}, Vector{UInt8}},Nothing}}(undef, n)	
	# kmer_pairs = Array{Tuple{Vector{UInt8}, Vector{UInt8}}}(undef, n)
	# kmer_pairs = Array{UInt8, undef, n, 2, k}	
	# kmer_pairs = Array{UInt8, undef, 1, 2, k}	

	seeds = [rand(rng, Int64) for i=1:n] # generate a seed for each mutation for reproducibility
	gen_task = Threads.@spawn begin 
		for i=1:n 			
			err_i = max_err*i^sparsity/n^sparsity
			# kmer = rand(rng, 1:4, k)
			kmer= rand(rng, [UInt8(l) for l=1:4], k)			 
			@time mut = mutate(kmer, err_i, rng=Xoshiro(seeds[i]))			
			kmer_pairs[i] = (kmer, mut)
			@info "↑ Mutation $i is done"
			# kmer_pairs[i,1,:] = kmer
			# kmer_pairs[i,2,:] = mut

			push!(out_ch, i)	
			# @show "gen kmer_pair", i
			# put!(out_ch, i)	
		end 

		close(out_ch)
	end

	# Threads.@spawn begin 
	# 	for hm in out_ch 
	# 		@show hm 
	# 	end
	# end

	# println(fetch(gen_task)) # to see errors 
	return out_ch, kmer_pairs
end

function compute_rope_encodings(kmers_ch::Channel, kmer_pairs::Array; encoder::RopeEncoder=nothing, oblivious=false) # uses Channel for sync
	n = length(kmer_pairs)
	out_ch = Channel{Int}(n)
	ropes = Array{Any}(undef, n)

	gen_task = Threads.@spawn begin 
		for i in kmers_ch 
			# @show i, 2
			kmer, mut = kmer_pairs[i]
			# kmer, mut = kmer_pairs[i,1,:], kmer_pairs[i,2,:]
			@time begin 
				ekmer_task = Threads.@spawn encoder(kmer)
				emut_task = Threads.@spawn encoder(mut)
				ekmer = fetch(ekmer_task)
				emut = fetch(emut_task)
				# ekmer = encoder(kmer)
				# emut = encoder(mut)
				ropes[i] = (ekmer, emut)
			end
			@info "↑ Ropes computation $i is done"

			if oblivious # we don't keep very long kmers
				kmer_pairs[i] = nothing
				# GC.gc()
			end

			# @show "gen rope", i
			push!(out_ch, i)			
		end 

		close(out_ch)
	end

	# println(fetch(gen_task)) # to see errors 
	return out_ch, ropes
end

function compute_angular_encodings(ropes_ch::Channel, ropes::Array; encoder::Angular2Encoder=nothing) # uses Channels for sync 
	n = length(ropes)
	out_ch = Channel{Int}(n)
	angulars = Array{Any}(undef, n)

	gen_task = Threads.@spawn begin 
		for i in ropes_ch 
			# @show i, 3
			ekmer, emut = ropes[i]
			akmer = encoder(ekmer)
			amut = encoder(emut)
			angulars[i] = (akmer, amut)

			# @show "gen angular", i
			push!(out_ch, i)		
		end

		close(out_ch)
	end

	# println(fetch(gen_task)) # to see errors 
	return out_ch, angulars
end

function max_angular_encoder_fidelity(rope1, rope2; scale=2/1.2) 
	n = length(rope1)
	# @assert n%2==0

	fids = Array{Any}(undef, n÷2)

	zs = zero_state(1)
	for i=1:n÷2
		# sand = sandwich(zs, Rx((rope1[i+n÷2]-rope2[i+n÷2])*scale)*Ry((rope1[i]-rope2[i])*scale), zs)
		sand = sandwich(zs, Rx((rope1[2i-1]-rope2[2i-1])*scale)*Ry((rope1[2i]-rope2[2i])*scale), zs)
		fids[i] = sand |> abs2
	end

	fid = prod(fids)
	return fid
end

function main(; n=100, buf=1, N=4000, q=1, s=5, q2=20, scale=2.4, err=0.4, seed=3, sparsity=1.7, save_results=false, oblivious=false)
	q1 = 2s+q+1
	rng = Xoshiro(seed)	
	kmers_ch, kmers = generate_kmer_pairs(k=N, n=n, buf=buf, max_err=err, sparsity=sparsity, rng=rng)
	# @show "s1"
	ropes_ch, ropes = compute_rope_encodings(kmers_ch, kmers, encoder=RopeEncoder(q=q,s=s,N=N), oblivious=oblivious)
	# @show "s2"
	angulars_ch, angulars = compute_angular_encodings(ropes_ch, ropes, encoder=Angular2Encoder(n=2^q1,q=q2,scale=scale))	
	# @show "s3"
	zs = zero_state(q2) 
	if cuda_available 
		zs = zs |> cu
	end

	ropes_export = zeros(Float64, n, 2, 2^(2s+q+1)) # export to python
	points_export = zeros(Float64, n, 4)
	
	if !oblivious
		kmers_export = zeros(Int8, n, 2, N) 
	end

	points = Array{Any}(undef, n)
	tasks = Array{Any}(undef, n)
	for i in angulars_ch
		tasks[i] = @spawn begin 
			@time begin 	
				if oblivious
					ld = err*i^sparsity/n^sparsity # use error rate instead of LD 
				else
					ld = LD(kmers[i][1], kmers[i][2])
				end 			
				fid1 = ropes[i][1]'*ropes[i][2] |> abs2
				fid2 = sandwich(zs, angulars[i][1]'*angulars[i][2], zs) |> abs2
				fid3 = max_angular_encoder_fidelity(ropes[i][1], ropes[i][2], scale=scale)
				points[i] = (ld, fid1, fid2, fid3)
			end
			@info "↑ Point $i computation is done"

			if save_results
				ropes_export[i,1,:] = ropes[i][1]
				ropes_export[i,2,:] = ropes[i][2]
				points_export[i, :] .= points[i]
				if !oblivious
					kmers_export[i,1,:] = kmers[i][1]
					kmers_export[i,2,:] = kmers[i][2]
				end
			end
		end
	end
	
	[wait(task) for task in tasks]

	if save_results
		save(points, "points.jlb")		
		# npzwrite("points.npy", points)
		npzwrite("ropes.npy", ropes_export)		
		npzwrite("points.npy", points_export)

		if !oblivious
			npzwrite("kmers.npy", kmers_export)
		end
	end

	return points
end

function main_ropes_only(; n=100, buf=1, N=4000, q=1, s=5, err=0.4, seed=3, sparsity=1.7, save_results=false, oblivious=false)
	q1 = 2s+q+1
	rng = Xoshiro(seed)	
	kmers_ch, kmers = generate_kmer_pairs(k=N, n=n, buf=buf, max_err=err, sparsity=sparsity, rng=rng)
	ropes_ch, ropes = compute_rope_encodings(kmers_ch, kmers, encoder=RopeEncoder(q=q,s=s,N=N), oblivious=oblivious)
	# angulars_ch, angulars = compute_angular_encodings(ropes_ch, ropes, encoder=Angular2Encoder(n=2^q1,q=q2,scale=scale))	
	# zs = zero_state(q2) 
	# if cuda_available 
	# 	zs = zs |> cu
	# end

	ropes_export = zeros(Float64, n, 2, 2^(2s+q+1)) # export to python
	# points_export = zeros(Float64, n, 4)
	points_export = zeros(Float64, n, 2)
	
	if !oblivious
		kmers_export = zeros(Int8, n, 2, N) 
	end

	points = Array{Any}(undef, n)
	tasks = Array{Any}(undef, n)
	# for i in angulars_ch
	for i in ropes_ch
		tasks[i] = @spawn begin 
			@time begin 	
				if oblivious
					ld = err*i^sparsity/n^sparsity # use error rate instead of LD 
				else
					ld = LD(kmers[i][1], kmers[i][2])
				end 			
				fid1 = ropes[i][1]'*ropes[i][2] |> abs2
				# fid2 = sandwich(zs, angulars[i][1]'*angulars[i][2], zs) |> abs2
				# fid3 = max_angular_encoder_fidelity(ropes[i][1], ropes[i][2], scale=scale)
				# points[i] = (ld, fid1, fid2, fid3)
				points[i] = (ld, fid1)
			end
			@info "↑ Point $i computation is done"

			if save_results
				ropes_export[i,1,:] = ropes[i][1]
				ropes_export[i,2,:] = ropes[i][2]
				points_export[i, :] .= points[i]
				if !oblivious
					kmers_export[i,1,:] = kmers[i][1]
					kmers_export[i,2,:] = kmers[i][2]
				end
			end
		end
	end
	
	[wait(task) for task in tasks]

	if save_results
		save(points, "points.jlb")		
		npzwrite("ropes.npy", ropes_export)		
		npzwrite("points.npy", points_export)

		if !oblivious
			npzwrite("kmers.npy", kmers_export)
		end
	end

	return points
end

function from_ropes(;q2=10, scale=2.4, save_results=false)
	ropes_export = npzread("ropes.npy")
	points_old = npzread("points.npy")
	n,_,d = size(ropes_export)
	# @show n,d

	ropes_ch = Channel{Int}(n)
	ropes = Array{Any}(undef, n)
	
	angulars = Array{Any}(undef, n)

	gen_task = Threads.@spawn begin 
		for i =1:n
			# @show i
			ropes[i] = (ropes_export[i,1,:], ropes_export[i,2,:])
			push!(ropes_ch, i)		
		end

		close(ropes_ch)
	end

	angulars_ch, angulars = compute_angular_encodings(ropes_ch, ropes, encoder=Angular2Encoder(n=d,q=q2,scale=scale))	
	# @show "s3"
	zs = zero_state(q2) 
	if cuda_available 
		zs = zs |> cu
	end

	points_export = zeros(Float64, n, 4) 

	points = Array{Any}(undef, n)
	for i in angulars_ch
		fid1 = ropes[i][1]'*ropes[i][2] |> abs2
		fid2 = sandwich(zs, angulars[i][1]'*angulars[i][2], zs) |> abs2
		fid3 = max_angular_encoder_fidelity(ropes[i][1], ropes[i][2],scale=scale)
		points[i] = (i, fid1, fid2, fid3)
		@show i

		if save_results
			points_export[i, :] .= points[i]
			points_export[i, 1] = points_old[i, 1] # use old value of ld / err rate
		end
	end
	
	if save_results
		save(points, "points2.jlb")
		npzwrite("points2.npy", points_export)
	end

	return points
end
