using Statistics
using Distributions
import Distributions: logpdf

include("adv.jl")


function dense_AE()
    # FIXME why leakyrelu
    encoder = Chain(x -> reshape(x, :, size(x, 4)),
                    Dense(28 * 28, 32, relu)) |> gpu
    decoder = Chain(Dense(32, 28 * 28),
                    x -> reshape(x, 28, 28, 1, :),
                    # FIXME use clamp?
                    # reshape(clamp.(x, 0, 1), 28, 28)
                    x -> σ.(x)) |> gpu
    Chain(encoder, decoder)
end

"""From https://discourse.julialang.org/t/upsampling-in-flux-jl/25919/3
"""
function upsample(x)
    ratio = (2, 2, 1, 1)
    (h, w, c, n) = size(x)
    y = similar(x, (1, ratio[1], 1, ratio[2], 1, 1))
    fill!(y, 1)
    z = reshape(x, (h, 1, w, 1, c, n))  .* y
    reshape(permutedims(z, (2,1,4,3,5,6)), size(x) .* ratio)
end

function CNN_AE()
    # FIXME padding='same'?
    encoder = Chain(Conv((3,3), 1=>16, pad=(1,1), relu, init=my_glorot_uniform),
                    # BatchNorm(16)
                    MaxPool((2,2)))
    decoder = Chain(Conv((3,3), 16=>16, pad=(1,1), relu, init=my_glorot_uniform),
                    # UpSampling((2,2)),
                    upsample,
                    Conv((3,3), 16=>1, pad=(1,1)),
                    x -> σ.(x))
    Chain(encoder, decoder) |> gpu
end

function CNN2_AE()
    encoder = Chain(Conv((3,3), 1=>16, pad=(1,1), relu, init=my_glorot_uniform),
                    MaxPool((2,2)),
                    Conv((3,3), 16=>8, pad=(1,1), relu, init=my_glorot_uniform),
                    MaxPool((2,2)))
    decoder = Chain(Conv((3,3), 8=>8, pad=(1,1), relu, init=my_glorot_uniform),
                    upsample,
                    Conv((3,3), 8=>16, pad=(1,1), relu, init=my_glorot_uniform),
                    upsample,
                    Conv((3,3), 16=>1, pad=(1,1), init=my_glorot_uniform),
                    x -> σ.(x))
    Chain(encoder, decoder) |> gpu
end

mymse(ŷ, y) = sum((ŷ .- y).*(ŷ .- y)) * 1 // length(y)

function aetrain!(model, opt, ds;
                  train_steps=ds.nbatch, print_steps=50)
    ps=Flux.params(model)

    loss_metric = MeanMetric()
    step = 0

    @info "Training for $train_steps steps, printing even $print_steps steps .."
    @showprogress 0.1 "Training..." for step in 1:train_steps
        x, _ = next_batch!(ds) |> gpu
        gs = Flux.Tracker.gradient(ps) do
            logits = model(x)
            loss = my_mse(logits, x)
            add!(loss_metric, loss.data)
            loss
        end
        Flux.Tracker.update!(opt, ps, gs)

        if step % print_steps == 0
            println()
            @show get!(loss_metric)
        end
    end
end

function evaluate_AE(ae, ds; cnn=nothing)
    xx,yy = next_batch!(ds) |> gpu

    @info "Sampling BEFORE images .."
    sample_and_view(xx, yy, cnn)
    @info "Sampling AFTER images .."
    # NOTE: I need .data, currently handled in sample_and_view
    rec = ae(xx)
    sample_and_view(rec, yy, cnn)
end

function advae_train!(ae, cnn, opt, attack_fn, ds;
                      train_steps=ds.nbatch, print_steps=50)
    ps=Flux.params(ae)

    m_cleanloss = MeanMetric()
    m_cleanacc = MeanMetric()
    m_advloss = MeanMetric()
    m_advacc = MeanMetric()

    model = Chain(ae, cnn)

    @info "Training for $train_steps steps, printing every $print_steps steps .."
    @showprogress 0.1 "Training..." for step in 1:train_steps
        x, y = next_batch!(ds) |> gpu
        # this computation won't affect model parameter gradients
        x_adv = attack_fn(model, x, y)

        gs = Flux.Tracker.gradient(ps) do
            # FIXME this will slow down the model twice
            adv_logits = model(x_adv)
            adv_loss = my_xent(adv_logits, y)

            clean_logits = model(x)
            clean_loss = my_xent(clean_logits, y)

            # DEBUG
            rec_loss = mymse(ae(x), x)
            # rec_loss = mymse(rec, x_adv)

            add!(m_advloss, adv_loss.data)
            add!(m_advacc, accuracy_with_logits(adv_logits.data, y))
            add!(m_cleanloss, clean_loss.data)
            add!(m_cleanacc, accuracy_with_logits(clean_logits.data, y))

            l = adv_loss + rec_loss
            # l = clean_loss
            # l = adv_loss + clean_loss
        end
        # FIXME clean up cnn parameter gradients?
        Flux.Tracker.update!(opt, ps, gs)

        if step % print_steps == 0
            println()
            @show get!(m_cleanloss)
            @show get!(m_cleanacc)
            @show get!(m_advloss)
            @show get!(m_advacc)
        end
    end
end

function test_AE()
    ds, test_ds = load_MNIST_ds(batch_size=50);
    x, y = next_batch!(ds) |> gpu;

    cnn = get_Madry_model()[1:end-1]
    cnn(x)

    opt = ADAM(1e-3)

    # TODO schedule lr simply by setting field of opt, and maintain the
    # state. FIXME But it seems that the state is maintained by a dict with key
    # Param(x). I'm not sure if this aproach can resume the state.
    #
    # opt.eta = 1e-4

    train!(cnn, opt, ds)


    # ae = dense_AE()
    ae = CNN_AE()
    # ae = CNN2_AE()

    # FIXME opt states?
    opt = ADAM(1e-4)
    # FIXME Flux.mse(logits, x)
    # FIXME sigmoid mse?
    aetrain!(ae, opt, ds)

    evaluate_AE(ae, test_ds, cnn=cnn)

    # FIXME performance overhead
    @epochs 2 advae_train!(ae, cnn, opt, attack_PGD_k(40), ds, print_steps=20)

    evaluate(cnn, test_ds)
    evaluate(cnn, test_ds, attack_fn=attack_PGD_k(40))

    evaluate(Chain(ae, cnn), test_ds)
    evaluate(Chain(ae, cnn), test_ds, attack_fn=attack_FGSM)
    evaluate(Chain(ae, cnn), test_ds, attack_fn=attack_PGD_k(40))
end
