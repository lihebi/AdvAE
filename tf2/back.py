def get_wrapperd_adv_model(model):
    class Dummy(cleverhans.model.Model):
        def __init__(self):
            self.model = model
            # FIXME are these shared layers?
            self.presoftmax = Sequential(model.layers[:-1])
        def predict(self, x):
            return self.presoftmax(x)
        def fprop(self, x, **kwargs):
            logits = self.predict(x)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}

    mm = Dummy()
    x = tf.keras.Input(shape=(28,28,1))
    y = tf.keras.Input(shape=(10,))
    adv = PGD(mm, x)
    adv = tf.stop_gradient(adv)
    y = model(adv)
    return tf.keras.Model(inputs, y)



def custom_advtrain_old():
    # ...

    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    train_advacc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    acc_fn = tf.keras.metrics.categorical_accuracy

    for epoch in range(3):
        clear_tqdm()
        for i, (x, y) in enumerate(tqdm(ds, total=tqdm_total)):
            train_step(model, params, opt, x, y, train_loss, train_acc)

            with train_summary_writer.as_default():
                # FIXME i + epoch * tqdm_total
                tf.summary.scalar('loss', train_loss.result(), step=i)
                tf.summary.scalar('accuracy', train_acc.result(), step=i)

            train_acc_metric.update_state(y, nat_logits)
            train_advacc_metric.update_state(y, adv_logits)
            if i % 20 == 0:
                print('')
                print('step {}, loss: {:.5f}'.format(i, loss))
                nat_acc = train_acc_metric.result()
                adv_acc = train_advacc_metric.result()
                print('nat acc: {:.5f}, adv acc: {:.5f}'.format(nat_acc, adv_acc))
                # reset here because I don't want the train loss to be delayed so much
                train_acc_metric.reset_states()
                train_advacc_metric.reset_states()
                # I also want to monitor the validation accuracy
                v_logits = model(vx)
                valid_nat_acc = acc_fn(v_logits, vy)
                adv_vx = attack_fn(model, vx, vy)
                v_logits = model(adv_vx)
                valid_adv_acc = acc_fn(v_logits, vy)
                print('valid nat acc: {:.5f}, valid adv acc: {:.5f}'
                      .format(valid_nat_acc.numpy().mean(), valid_adv_acc.numpy().mean()))

        # Display metrics at the end of each epoch.
        nat_acc = train_acc_metric.result()
        adv_acc = train_advacc_metric.result()
        print('nat acc: {:.5f}, adv acc: {:.5f}'.format(nat_acc, adv_acc))
        train_acc_metric.reset_states()
        train_advacc_metric.reset_states()

def exp_all():
    mixing_fns = {
        # TODO linear. This function is essentially nat^2
        'nat': lambda nat, adv: nat,

        # simply mixing
        'f1': lambda nat, adv: 1,

        'f0': lambda nat, adv: 0,

        # FIXME (σ(nat) - 0.5) * 2
        # 'σ(nat)': lambda nat, adv: tf.math.sigmoid(nat),
        # relative value
        # FIXME will this be slow to compute gradient?
        # TODO σ(nat) * natloss + σ(adv) * advloss?
        # FIXME divide by zero
        # FIXME this should not be stable
        # 'σ(nat:adv)': lambda nat, adv: tf.math.sigmoid(nat / adv)
        # TODO functions other than σ?
        # TODO use training iterations? This should be a bad idea.
    }

    lrs = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1]
    # lrs = [1e-2, 1e-1, 5e-3]
    # lrs = [5e-3, 3e-3]

    # default config
    config = get_default_config()

    for fname in mixing_fns:
        for lr in reversed(lrs):
            # DEBUG using time in the logname
            # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # config['logname'] = '{}-{}-{}'.format(fname, lr, current_time)

            config['logname'] = '{}-{}'.format(fname, lr)

            print('=== Exp:', config['logname'])
            config['mixing_fn'] = mixing_fns[fname]
            config['lr'] = lr
            exp_train(config)
