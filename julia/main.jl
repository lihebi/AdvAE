# This is intended as entry for running experiment

# package management

using Pkg; Pkg.activate("."); Pkg.add("ProgressMeter")
# using Zygote causes many errors, not using for now
using Pkg; Pkg.activate("."); Pkg.rm("Zygote")
