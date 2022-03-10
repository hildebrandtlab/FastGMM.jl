using FastGMM
using Distributions
using StatsBase

d = 2
n = 1000000

# start very simple
m = MvNormal([10, -10], [5, 0.2])

X = Float32.(rand(m, n))
ω = Float32.(repeat([1], n))

gmm = fitGMM(X, ω, 1)




K = 4

# build a model to sample from
m = MixtureModel(MvNormal, [
    ([0, 0], [1, 1]),
    ([5, 5], [0.2, 0.4]),
    ([10, 0], [0.8, 0.1]),
    ([1, 24], [0.2, 2])],
    [0.4, 0.2, 0.1, 0.3])

X = Float32.(rand(m, n))
ω = Float32.(repeat([1], n))

gmm = fitGMM(X, ω, K)
