import Pkg; Pkg.add("Flux")
import Pkg; Pkg.add("Zygote")
import Pkg; Pkg.add("Random")
using Flux, Zygote, Random, Printf

Random.seed!(123)
ν = Float32(0.01 / π)
n_input, n_hidden, n_output = 2, 20, 1
N_f, N_i, N_b = 2000, 100, 100

# Collocation points (t,x) in interior
t_f = Float32.(rand(N_f) .* 1.0)
x_f = Float32.(rand(N_f) .* 2.0 .- 1.0)

# Initial condition points (t=0, x)
t_i = zeros(Float32, N_i)
x_i = Float32.(rand(N_i) .* 2.0 .- 1.0)

# Boundary points (t, x=-1) and (t, x=1)
t_b = [rand(Float32, N_b); rand(Float32, N_b)]
x_b = [fill(Float32(-1.0), N_b); fill(Float32(1.0), N_b)]

model = Chain(
    Dense(n_input, n_hidden, tanh),
    Dense(n_hidden, n_hidden, tanh),
    Dense(n_hidden, n_hidden, tanh),
    Dense(n_hidden, n_output)
) |> f32

function compute_loss()
    # PDE residual loss
    mse_pde = 0.0f0
    for i in 1:N_f
        t, x = t_f[i], x_f[i]
        u_val, (∂u_∂t, ∂u_∂x) = pullback(t, x) do t, x
            model([t, x])[1]
        end
        ∂²u_∂x² = derivative(x -> ∂u_∂x(x), x)
        residual = ∂u_∂t + u_val * ∂u_∂x - ν * ∂²u_∂x²
        mse_pde += residual^2
    end
    mse_pde /= N_f

    # Initial condition loss
    mse_ic = sum(abs2, model([t_i[i]; x_i[i]][1] + sin(π * x_i[i])) for i in 1:N_i) / N_i

    # Boundary condition loss
    mse_bc = sum(abs2, model([t_b[i]; x_b[i]][1]) for i in 1:2N_b) / (2N_b)

    return mse_pde + mse_ic + mse_bc
end

# Helper functions for derivatives
pullback(f, x) = Zygote.pullback(f, x)
derivative(f, x) = first(Zygote.gradient(f, x))

opt = ADAM(0.001f0)
parameters = Flux.params(model)
epochs = 1000

for epoch in 1:epochs
    loss_val, back = Flux.withgradient(compute_loss, parameters)
    Flux.update!(opt, parameters, back)
    @printf("Epoch: %4d, Loss: %.4e\n", epoch, loss_val)
end