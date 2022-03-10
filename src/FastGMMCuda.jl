export initGMM, fitGMM

using CUDA
using ParallelKMeans

# NOTE: 
#  - Julia is column major by default => X ∈ R^{d×n}
#  - ω are the weights of the data points
function initGMM(X::AbstractArray{T}, ω::AbstractArray{T}, K::Int) where {T<:Real}
    dims = size(X)
    @assert length(dims) == 2

    d = dims[1]
    n = dims[2]

    @assert n > 0
    @assert d > 0

    @assert K > 0

    gmm = GMM{T}(d, K)

    # start by a round of k-means to quickly generate good guesses for the cluster centroids
    # (and potentially the number of components)
    kmeans_results = kmeans(Yinyang(), X, K; weights=ω)

    # seed the centroids
    copyto!(gmm.μ, kmeans_results.centers)

    gmm
end
 
function fitGMM(in_X::AbstractArray{T}, in_ω::AbstractArray{T}, K; max_iterations=100, tolerance=1e-8) where {T<:Real}
    gmm = initGMM(in_X, in_ω, K)

    X = CuArray(in_X)
    ω = CuArray(in_ω)

    n = size(X)[2]
  
    iteration = 0

    # todo: make this not necessarily dependent on CUDA
    # NOTE: this matrix can easily get HUGE
    log_γ = CUDA.zeros(n, K)

    last_log_likelihood = typemin(T)

    while iteration < max_iterations
        # 1.) E-Step

		# we use logarithms in the following
		log_π = transpose(log.(gmm.π))

        # NOTE: what we would like to do is compute all of those in parallel, but this could have a huge memory footprint
        # TODO: spawn in parallel! wait for completion (see https://cuda.juliagpu.org/stable/usage/multitasking/), pin memory
        for k in 1:K
            # compute log(p(X|μ, Σ))
            #
            # since p(X|μ,Σ) = 1/sqrt((2π)^d|Σ|) * exp(-1/2 (x-μ)^t Σ^{-1} (x-μ))
            #   => log(p(X|μ,Σ)) = -0.5 * (d * log(2π) + log(|Σ|)) - 0.5 log((x-μ)^t Σ^{-1} (x-μ))
            # and since right now, we only allow diagonal covariances, Σ^{-1} is the diagonal matrix with
            # 1/σ_i on the diagonal
            #
            # since each point occurs ω_i times, we can multiply the terms in the sum by ω_i
            #
            # TODO: once Distributions.jl is GPU friendly, we should replace this entirely
            log_γ[:, k] = get_normalizer(gmm.σ[:, k], gmm.d) .- 0.5 * sum(1 ./ gmm.σ[:, k] .* transpose(ω) .* (X .- gmm.μ[:, k]).^2; dims=1)
        end

        # for each point, find the maximum log probability from log(π_k) + log(p(X_i|μ_k, Σ_k)) - log(p(X_i)) = log(π_k) + log_γ[:, k]
        max_log_p_k_per_point = maximum(log_π .+ log_γ, dims=2)

        # this seems necessary for numerical stability
        log_p_x = log.(sum(exp.(log_γ .+ log_π .- max_log_p_k_per_point), dims=2)) .+ max_log_p_k_per_point

        log_γ = log_γ .- log_p_x

        # at this point, log_γ[i, k] = p(X_i | μ_k, Σ_k) / p(X_i)
        current_log_likelihood = sum(log_p_x)

        if abs(current_log_likelihood - last_log_likelihood) < tolerance || current_log_likelihood < last_log_likelihood
            break
        end

        last_log_likelihood = current_log_likelihood

        # 2.) M-Step
        max_log_γ_per_x = maximum(log_γ, dims=1)
        log_Γ = max_log_γ_per_x .+ log.(sum(exp.(log_γ .- max_log_γ_per_x), dims=1))

		# TODO: check whether we should a loop to distribute that over K loops
		gmm.μ = ((transpose(ω) .* X) * exp.(log_γ)) ./ exp.(log_Γ)

		# TODO: use streams and distribute?
		# NOTE: this here runs into allowscalar - problems
		gmm.σ = reduce(hcat, [sum(transpose(exp.(log_γ[:, k]) .* ω) .* (X .- gmm.μ[:, k]).^2; dims=2) / exp(log_Γ[k]) for k in 1:K])

		log_π_log_Γ = log_π .+ log_Γ
		max_log_π_log_Γ	= maximum(log_π_log_Γ)

		gmm.π = transpose(exp.(log_π .+ log_Γ .- max_log_π_log_Γ .- log(sum(exp.(log_π_log_Γ .- max_log_π_log_Γ)))))

		# TODO: handle this correctly!
		gmm.π = clamp.(gmm.π, 1e-8, 1.0f0-1e-8)

		iteration += 1
	end

	gmm
end