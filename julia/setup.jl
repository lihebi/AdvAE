using Pkg

# activate current project
Pkg.activate(pwd())

# install missing packages
Pkg.instantiate()

# current status
Pkg.status()

# restore to default
Pkg.activate()

# managing projects
Pkg.update()
# precompile command is availabe at PKG repl, the usage here is not very straightforward
# Pkg.precompile

LOAD_PATH
DEPOT_PATH
ENV["JULIA_LOAD_PATH"]

Pkg.add("BSON")
Pkg.add("Images")
Pkg.add("MLDatasets")
Pkg.add("TensorBoardLogger")
Pkg.add("LoggingExtras")
Pkg.add("ProgressMeter")

# FIXME Flux is downgraded to 0.8
Pkg.add("Flux")
Pkg.add(PackageSpec(name="Flux", version="0.9"))

# Pkg.add(PackageSpec(name="Flux", rev="master"))
#
# fix has_cudnn instead of libcudnn
# Pkg.develop("Flux")

Pkg.add("CuArrays")
# Pkg.add("Metalhead")

# used packages
# Pkg.add(PackageSpec(url="https://github.com/jaypmorgan/Adversarial.jl.git"))

# Pkg.add("Plots")

# Pkg.add("FixedPointNumbers")
# Pkg.add("Distances")
# Pkg.add("Distributions")
# Pkg.add("FileIO")

# FIXME Augmentor not on Julia 1.0
# Pkg.add("Augmentor")
# Pkg.add(PackageSpec(url="https://github.com/Evizero/Augmentor.jl"))


# CUDA support
#
# Pkg.add("CUDAapi")
# Pkg.add("NNlib")
# Pkg.add("CUDAnative")
# Pkg.add("CUDAdrv")

# For cutting edge Zygote, I'm not sure whether these masters are needed
#
# Pkg.add(PackageSpec(name="CUDAnative", rev="master"))
# Pkg.add(PackageSpec(name="CUDAdrv", rev="master"))
# Pkg.add(PackageSpec(name="CuArrays", rev="master"))

# Zygote is unstable
#
# Pkg.add("ZygoteRules")
# Pkg.add("Zygote")
# only the master export pullback
# Pkg.add(PackageSpec(name="ZygoteRules", rev="master"))
# Pkg.add(PackageSpec(name="Zygote", rev="master"))


CuArrays.allowscalar(false)
CuArrays.allowscalar(true)
