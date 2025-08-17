# Add Pkgs
# import Pkg; Pkg.add("Optim")
# import Pkg; Pkg.add("BenchmarkTools")
# import Pkg; Pkg.add("FileIO")
# import Pkg; Pkg.add("JLD2")

# Load necessary packages
using Interpolations, LinearAlgebra, Optim
using Plots
using FileIO, JLD2
using BenchmarkTools

# Define the model parameters and computation parameters

function compparam(; maxiter=1000, itertol=1e-8, opttol=1e-8)
    
    (; maxiter, itertol, opttol)
end

function modelparam(; A = 1, beta = 0.8, alpha = 0.5, xgrid=range(1e-8, 0.4, length=1000), useV=false)
    if useV                                                                 # Load initial value function from file if it exists
        savevars = FileIO.load("cakeeating/cakeeating.jld2", "savevars")
        V0vec = savevars.V0vec
    else
        V0vec = zeros(length(xgrid))
    end

    (; A, beta, alpha, xgrid, V0vec, kpvec= zeros(length(xgrid)), Vvec= zeros(length(xgrid)))
end

# Function to evaluate the value function at a given capital and state

function valfun(modelparam; kp, x, itp)
    (; A, beta, alpha) = modelparam
    
    c = A * (x^alpha) - kp
    return c <= 0 ? Inf : -(log(c) + beta * itp(kp))

end

# Function to compute the Bellman equation

function Bellman(modelparam, compparam; itp)
    (; opttol) = compparam
    (; A, beta, alpha, xgrid, V0vec, kpvec, Vvec) = modelparam

    for j in eachindex(xgrid)
        x = xgrid[j]
        res = Optim.optimize(kp->valfun(modelparam; kp, x, itp), opttol, A*x^alpha)
        kpvec[j] = Optim.minimizer(res)
        Vvec[j] = -Optim.minimum(res)
    end


    return (; V0vec, kpvec, Vvec, itp)

end

# Function to perform Value Function Iteration (VFI)

function VFI(modelparam, compparam)
    (; maxiter, itertol) = compparam
    (; xgrid, V0vec, kpvec, Vvec) = modelparam

    for iter in 1:maxiter

        # Interpolate the value function
        itpraw = interpolate(V0vec, BSpline(Cubic()))
        itp    = scale(itpraw, xgrid)

        (; V0vec, kpvec, Vvec, itp) = Bellman(modelparam, compparam; itp)
        
        err = maximum(abs.(Vvec - V0vec))
        V0vec .= Vvec  # Update the value function for the next iteration

        if err < itertol
            println("Converged after $iter iterations with error $err")
            break
        end
        
    end

    # Save the results to a file
    savevars = (; V0vec, kpvec)
    FileIO.save("cakeeating/cakeeating.jld2", "savevars", savevars)

    return Vvec, kpvec, xgrid
end


# Function to perform Howard's method for Value Function Iteration (VFI)

function HowardVFI(modelparam, compparam)
    (; maxiter, itertol) = compparam
    (; xgrid, V0vec, kpvec, Vvec) = modelparam

    for iter in 1:maxiter

        # Interpolate the value function
        itpraw = interpolate(V0vec, BSpline(Cubic()))
        itp    = scale(itpraw, xgrid)

        # Howard's method: update the value function
        if mod(iter, 10) == 1
            (; V0vec, kpvec, Vvec, itp) = Bellman(modelparam, compparam; itp)
        else
            for j in eachindex(xgrid)
                x = xgrid[j]
                kp = kpvec[j]
                Vvec[j] = -valfun(modelparam; kp, x, itp)
            end
        end
        
        err = maximum(abs.(Vvec - V0vec))
        V0vec .= Vvec  # Update the value function for the next iteration

        if err < itertol
            println("Converged after $iter iterations with error $err")
            break
        end
        
    end

    # Save the results to a file
    savevars = (; V0vec, kpvec)
    FileIO.save("cakeeating/cakeeating.jld2", "savevars", savevars)

    return Vvec, kpvec, xgrid
end


# Function to plot the results of Value Function Iteration

function plotVFI(modelparam, compparam; saveplot=true)
    # Vvec, kpvec, xgrid = VFI(modelparam, compparam)             # Perform VFI
    Vvec, kpvec, xgrid = HowardVFI(modelparam, compparam)       # Perform Howard's VFI
    
    p1 = plot(xgrid[100:600], Vvec[100:600], label="Value Function (V)", xlabel="Capital (k)", ylabel="Value (V)", linestyle=:dash, linewidth=2, title="Value Function Iteration")
    p2 = plot(xgrid[100:600], kpvec[100:600], label="Optimal Policy (k')", xlabel="Capital (k)", ylabel="Saving (K')", linestyle=:dash, linewidth=2)

    vfifig = plot(p1, p2, layout=(2, 1), size=(1000, 800), legend=:bottomright)

    if saveplot
        savefig(vfifig, "cakeeating/cakeeating_vfi.pdf")
    end
end


# Run the Value Function Iteration and plot the results

plotVFI(modelparam(; useV=false), compparam())