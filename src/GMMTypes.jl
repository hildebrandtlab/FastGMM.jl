export GMM, get_normalizer

mutable struct GMM{T<:Real}
    # dimension
    d::Int

    # number of components
    K::Int

    # weight, mean, covariance
    # NOTES: 
    #   - π is scalar while μ and σ are d-dimensional
    #   - right now, we only support diagonal σ

    π::AbstractArray{T}
    μ::AbstractArray{T}
    σ::AbstractArray{T}

    function GMM{T}(d::Int, K::Int) where T<:Real
        π = CUDA.fill(1.0f0 / K, K)
        μ = CuArray{T}(undef, (d, K))
        σ = CUDA.ones(d, K) # start with unit variances

        new(d, K, π, μ, σ)
    end
end

function Base.show(io::IO, gmm::GMM{T}) where {T<:Real}
    println(io, "$(gmm.d)-dimensional GMM{$T} with $(gmm.K) components:\n")
    println(io, "\tπ: $(gmm.π)")
    println(io, "\tμ: $(gmm.μ)")
    println(io, "\tσ: $(gmm.σ)")
end
   
function get_normalizer(σ::AbstractArray{T}, d::Int) where {T<:Real}
    # the normalizer of the d-dimensional normal distribution is sqrt((2π)^d |Σ|)
    # since we will always use log-transformed quantities, we thus need
    # log(1/(((2π)^d |Σ|)^(1/2))) = log(((2π)^d |Σ|)^(-1/2)) = -1/2 * (log(2π)*d + log(|Σ|))
    
    # right now, we only allow diagonal covariances => |Σ| = Π_{i=1}^d σ_i => log(|Σ|) = Σ_{i=1}^d log(σ_i)
    return -0.5f0 .* ((log(2.0f0 * pi) * d) .+ sum(log.(σ); dims=1))
end