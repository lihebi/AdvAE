using Pkg

# activate current project
Pkg.activate(pwd())

# install missing packages
Pkg.instantiate()

# current status
Pkg.status()

# restore to default
Pkg.activate()


# used packages
Pkg.add(PackageSpec(url="https://github.com/jaypmorgan/Adversarial.jl.git"))

Pkg.add("MLDatasets")
Pkg.add("TensorBoardLogger")
Pkg.add("LoggingExtras")
Pkg.add("Plots")
Pkg.add("ProgressMeter")


# CUDA support
#
# Pkg.add("CUDAapi")
# Pkg.add("NNlib")
# Pkg.add("CUDAnative")
# Pkg.add("CUDAdrv")
# Pkg.add("CuArrays")

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

Pkg.add("Flux")
# Pkg.add(PackageSpec(name="Flux", rev="master"))
#
# fix has_cudnn instead of libcudnn
# Pkg.develop("Flux")


CuArrays.allowscalar(false)
CuArrays.allowscalar(true)
