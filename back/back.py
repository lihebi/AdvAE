def keras_main():
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    train_x
    train_y
    val_x
    test_x
    model = get_mnist_model()
    model.summary()
    model.fit([train_x, train_y],
              batch_size=BATCH_SIZE,
              validation_split=0.1,
              # validation_data=((val_x, val_y)),
              nb_epoch=NUM_EPOCHS,
              # callbacks=[es],
              shuffle=True)
    # train_model(model, train_x, train_y, val_x, val_y)
    loss, acc = model.evaluate((test_x, test_y))
    print('loss: {:.5f}, acc: {:.5f}'.format(loss, acc))


def get_mnist_model():
    img = keras.layers.Input(shape=(28,28,1,), dtype='float32')
    label = keras.layers.Input(shape=(10,), dtype='float32')
    
    x = keras.layers.Conv2D(32, 3)(img)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(32, 3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D((2,2))(x)
    
    x = keras.layers.Conv2D(64, 3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(64, 3)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D((2,2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(200)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(200)(x)
    x = keras.layers.Activation('relu')(x)

    logits = keras.layers.Dense(10)(x)

    
    # x = keras.layers.Softmax()(x)

    
    ce_loss = tf.losses.softmax_cross_entropy(label, logits)
    
    # vi_loss = tf.reduce_mean(tf.gradients(logits, img))
    # define many templates
    # mast the loss through the templates
    # loss = ce_loss + vi_loss
    loss = ce_loss

    model = keras.models.Model([img, label], logits)
    model.add_loss(loss)
    model.compile(optimizer='rmsprop', metrics=['accuracy'])
    return model


def train_model(model, train_x, train_y, val_x, val_y):
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # def fn(correct, predicted):
    #     # keras.losses.categorical_crossentropy
    #     return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
    #                                                    logits=predicted)
    # model.add_loss(tf.nn.softmax_cross_entropy_with_logits)
    # model.add_loss(fn)
    def fn(y, yhat):
        return tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat)
        # return keras.losses.categorical_crossentropy(y, yhat)
    model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])
    # model.compile(optimizer=sgd, metrics=['accuracy'])
    # es = keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                    min_delta=0, patience=3,
    #                                    verbose=0, mode='auto')
    model.fit([train_x, train_y],
              batch_size=BATCH_SIZE,
              # validation_data=(val_x, val_y),
              nb_epoch=NUM_EPOCHS,
              # callbacks=[es],
              shuffle=True)

    # if file_name != None:
    #     model.save(file_name)

    return model
def my_fast_PGD(sess, model, x, y):
    "DEPRECATED"
    epsilon = 0.3
    k = 40
    a = 0.01
    
    # loss = model.cross_entropy()
    # grad = tf.gradients(loss, model.x)[0]
    
    # FGSM, PGD, JSMA, CW
    # without random
    # 0.97, 0.97, 0.82, 0.87
    # using random, the performance improved a bit:
    # 0.99, 0.98, 0.87, 0.93
    #
    # adv_x = np.copy(x)
    adv_x = x + np.random.uniform(-epsilon, epsilon, x.shape)
    
    for i in range(k):
        g = sess.run(model.ce_grad, feed_dict={model.x: adv_x, model.y: y})
        adv_x += a * np.sign(g)
        adv_x = np.clip(adv_x, x - epsilon, x + epsilon)
        # adv_x = np.clip(adv_x, 0., 1.)
        adv_x = np.clip(adv_x, CLIP_MIN, CLIP_MAX)
    return adv_x

self.ce_grad = tf.gradients(self.ce_loss, self.x)[0]
def train_adv(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        loss = model.cross_entropy()
        train_step = tf.train.AdamOptimizer(0.001).minimize(model.ce_loss)

        # my_adv_training does not contain initialization of variable
        init = tf.global_variables_initializer()
        sess.run(init)

        my_adv_training(sess, model, loss, train_step=train_step, batch_size=128, num_epochs=50)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)
    def setup_loss(self):
        rec = self.AE(self.x)
        self.rec_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=rec, labels=self.x_ref))

        high = self.CNN(self.AE(self.x))
        high_ref = self.CNN(self.x_ref)
        self.high_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=high, labels=high_ref))
        
        self.logits = self.FC(self.CNN(self.AE(self.x)))
        self.ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y))

        clean_rec = self.AE(self.x_ref)
        self.clean_rec_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=clean_rec, labels=self.x_ref))
        
        clean_high = self.CNN(self.AE(self.x_ref))
        self.clean_high_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=clean_high, labels=high_ref))
        
        clean_logits = self.FC(self.CNN(self.AE(self.x_ref)))
        self.clean_ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=clean_logits, labels=self.y))

        self.preds = tf.argmax(self.logits, axis=1)
        # self.probs = tf.nn.softmax(self.logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))

        # I also want to add the Gaussian noise training objective
        # into the unified loss, such that the resulting AE will still
        # acts as a denoiser
        noise_factor = 0.5
        noisy_x = self.x_ref + noise_factor * tf.random.normal(shape=tf.shape(self.x_ref), mean=0., stddev=1.)
        noisy_x = tf.clip_by_value(noisy_x, CLIP_MIN, CLIP_MAX)
        noisy_rec = self.AE(noisy_x)
        self.noisy_rec_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=noisy_rec, labels=self.x_ref))
        noisy_logits = self.FC(self.CNN(self.AE(noisy_x)))
        self.noisy_ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=noisy_logits, labels=self.y))

        noisy_high = self.CNN(self.AE(noisy_x))
        self.noisy_high_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=noisy_high, labels=high_ref))

        # TODO monitoring each of these losses and adjust weights
        self.metrics = [
            # adv data
            self.rec_loss, self.ce_loss, self.high_loss,
            # clean data
            self.clean_rec_loss, self.clean_ce_loss, self.clean_high_loss,
            # noisy data
            self.noisy_rec_loss, self.noisy_ce_loss, self.noisy_high_loss]
        self.metric_names = [
            # adv data
            'rec_loss', 'ce_loss', 'high_loss',
            # clean data
            'clean_rec_loss', 'clean_ce_loss', 'clean_high_loss',
            # noisy data
            'noisy_rec_loss', 'noisy_ce_loss', 'noisy_high_loss']
        # DEBUG adjust here for different loss terms and weights
        self.unified_adv_loss = (self.ce_loss
                                 + self.noisy_ce_loss
                                 # + self.high_loss
        )
        self.unified_adv_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.unified_adv_loss, var_list=self.AE_vars)
        
def plot_two_scale():
                # plot loss
                fig, ax = plt.subplots()
                losses, accs = np.transpose(plot_data_loss_acc)

                color = 'tab:red'
                ax.set_ylabel('loss', color=color)
                ax.tick_params(axis='y', labelcolor=color)
                ax.plot(losses, label='loss', color=color)
                
                axnew = ax.twinx()
                color = 'tab:blue'
                axnew.set_ylabel('acc', color=color)
                axnew.tick_params(axis='y', labelcolor=color)
                axnew.plot(accs, label='acc', color=color)
                
                legend = ax.legend()
                legend = axnew.legend()
                plt.savefig('training-process-acc.png')
                plt.close(fig)
        # different types of adv
        # PGD_adv_x = self.myPGD(self.x)
        # PGD_adv_rec = self.AE(self.myPGD(self.x))
        # PGD_postadv = self.myPGD(rec)

        # FGSM_adv_x = self.myFGSM(self.x)
        # FGSM_adv_rec = self.AE(self.myFGSM(self.x))
        # FGSM_postadv = self.myFGSM(rec)

        # JSMA_adv_x = self.myJSMA(self.x)
        # JSMA_adv_rec = self.AE(self.myJSMA(self.x))
        # JSMA_postadv = self.myJSMA(rec)

        # if run_CW:
        #     CW_adv_x = self.myCW(self.x)
        #     CW_adv_rec = self.AE(self.myCW(self.x))
        #     CW_postadv = self.myCW(rec)
    def test_AE(self, sess, clean_x):
        # Testing denoiser
        noise_factor = 0.5
        noisy_x = clean_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_x.shape)
        noisy_x = np.clip(noisy_x, CLIP_MIN, CLIP_MAX)

        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        
        rec = self.AE(inputs)
        model = keras.models.Model(inputs, rec)

        rec = sess.run(rec, feed_dict={inputs: noisy_x})
        print('generating png ..')
        to_view = np.concatenate((noisy_x[:5], rec[:5]), 0)
        grid_show_image(to_view, 5, 2, 'AE_out.png')
        print('PNG generatetd to AE_out.png')

    def test_Entire(self, sess, clean_x, clean_y):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        labels = keras.layers.Input(shape=(10,), dtype='float32')
        
        logits = self.FC(self.CNN(self.AE(inputs)))
        preds = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            preds, tf.argmax(labels, 1)), dtype=tf.float32))
        acc = sess.run(accuracy, feed_dict={inputs: clean_x, labels: clean_y})
        print('clean accuracy: {}'.format(acc))
        
        # I'm adding testing for accuracy of noisy input
        noise_factor = 0.5
        noisy_x = clean_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_x.shape)
        noisy_x = np.clip(noisy_x, CLIP_MIN, CLIP_MAX)
        acc = sess.run(accuracy, feed_dict={inputs: noisy_x, labels: clean_y})
        print('noisy x accuracy: {}'.format(acc))

    def test_Adv_Denoiser(self, sess, clean_x, clean_y):
        """Visualize the denoised image against adversarial examples."""
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        labels = keras.layers.Input(shape=(10,), dtype='float32')

        adv_x = self.myPGD(inputs)
        rec = self.AE(adv_x)
        logits = self.FC(self.CNN(rec))
        preds = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            preds, tf.argmax(labels, 1)), dtype=tf.float32))
        
        adv_x_concrete, denoised_x, acc = sess.run([adv_x, rec, accuracy],
                                                   feed_dict={inputs: clean_x, labels: clean_y})

        print('accuracy: {}'.format(acc))
        print('generating png ..')
        to_view = np.concatenate((adv_x_concrete[:5], denoised_x[:5]), 0)
        grid_show_image(to_view, 5, 2, 'AdvAE_out.png')
        print('PNG generatetd to AdvAE_out.png')
    def test_CNN(self, sess, clean_x, clean_y):
        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        labels = keras.layers.Input(shape=(10,), dtype='float32')
        
        logits = self.FC(self.CNN(inputs))
        preds = tf.argmax(logits, axis=1)
        probs = tf.nn.softmax(logits)
        # self.acc = tf.reduce_mean(
        #     tf.to_float(tf.equal(tf.argmax(self.logits, axis=1),
        #                          tf.argmax(self.label, axis=1))))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            preds, tf.argmax(labels, 1)), dtype=tf.float32))
        acc = sess.run(accuracy, feed_dict={inputs: clean_x, labels: clean_y})
        print('accuracy: {}'.format(acc))
        
    # test whether CNN part is functioning
    model.test_CNN(sess, test_x, test_y)
    # clean image and plot image
    model.test_AE(sess, test_x)
    # test clean image and noisy image, the final accuracy
    model.test_Entire(sess, test_x, test_y)
    # FIXME remove
    model.test_Adv_Denoiser(sess, test_x[:10], test_y[:10])
    
    acc = sess.run(model.accuracy, feed_dict={model.x: test_x, model.y: test_y})
    print('Model accuracy on clean data: {}'.format(acc))
def my_FGSM(sess, model):
    # FGSM attack
    fgsm_params = {
        'eps': 0.3,
        'clip_min': CLIP_MIN,
        'clip_max': CLIP_MAX
    }
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(model.x, **fgsm_params)
    return adv_x

def my_PGD(sess, model):
    pgd = ProjectedGradientDescent(model, sess=sess)
    pgd_params = {'eps': 0.3,
                  'nb_iter': 40,
                  'eps_iter': 0.01,
                  'clip_min': CLIP_MIN,
                  'clip_max': CLIP_MAX}
    adv_x = pgd.generate(model.x, **pgd_params)
    return adv_x

def my_JSMA(sess, model):
    jsma = SaliencyMapMethod(model, sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': CLIP_MIN, 'clip_max': CLIP_MAX,
                   'y_target': None}
    return jsma.generate(model.x, **jsma_params)
    

def my_CW(sess, model):
    """When feeding, remember to put target as y."""
    # CW attack
    cw = CarliniWagnerL2(model, sess)
    cw_params = {'binary_search_steps': 1,
                 # 'y_target': model.y,
                 'y': model.y,
                 'max_iterations': 10000,
                 'learning_rate': 0.2,
                 # setting to 100 instead of 128, because the test_x
                 # has 10000, and the last 16 left over will cause
                 # exception
                 'batch_size': 100,
                 'initial_const': 10,
                 'clip_min': CLIP_MIN,
                 'clip_max': CLIP_MAX}
    adv_x = cw.generate(model.x, **cw_params)
    return adv_x

def my_CW_targeted(sess, model):
    """When feeding, remember to put target as y."""
    # CW attack
    cw = CarliniWagnerL2(model, sess)
    cw_params = {'binary_search_steps': 1,
                 'y_target': model.y,
                 # 'y': model.y,
                 'max_iterations': 10000,
                 'learning_rate': 0.2,
                 # setting to 100 instead of 128, because the test_x
                 # has 10000, and the last 16 left over will cause
                 # exception
                 'batch_size': 10,
                 'initial_const': 10,
                 'clip_min': CLIP_MIN,
                 'clip_max': CLIP_MAX}
    adv_x = cw.generate(model.x, **cw_params)
    return adv_x

class CNNModel(cleverhans.model.Model):
    def __init__(self):
        self.setup_model()
    # abstract interfaces
    def x_shape(self):
        assert(False)
        return None
    def y_shape(self):
        assert(False)
        return None
    def rebuild_model(self):
        assert(False)
        return None
    # general methods
    def setup_model(self):
        shape = self.x_shape()
        # DEBUG I'm trying the input layer vs. placeholders
        self.x = keras.layers.Input(shape=self.x_shape(), dtype='float32')
        self.y = tf.placeholder(shape=self.y_shape(), dtype='float32')
        self.logits = self.rebuild_model(self.x)
        self.preds = tf.argmax(self.logits, axis=1)
        self.probs = tf.nn.softmax(self.logits)
        # self.acc = tf.reduce_mean(
        #     tf.to_float(tf.equal(tf.argmax(self.logits, axis=1),
        #                          tf.argmax(self.label, axis=1))))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))
        # FIXME num_classes
        # self.num_classes = np.product(self.y_shape[1:])
    def cross_entropy_with(self, y):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=y))
    def logps(self):
        """Symbolic TF variable returning an Nx1 vector of log-probabilities"""
        return self.logits - tf.reduce_logsumexp(self.logits, 1, keep_dims=True)
    def predict(self, x):
        return keras.models.Model(self.x, self.logits)(x)
    # cleverhans
    def fprop(self, x, **kwargs):
        logits = self.predict(x)
        return {self.O_LOGITS: logits,
                self.O_PROBS: tf.nn.softmax(logits=logits)}


    @staticmethod
    def compute_accuracy(sess, x, y, preds, x_test, y_test):
        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1)),
            dtype=tf.float32))
        return sess.run(acc, feed_dict={x: x_test, y: y_test})

    # loss functions. Using different loss function would result in
    # different model to compare
    def certainty_sensitivity(self):
        """Symbolic TF variable returning the gradients (wrt each input
        component) of the sum of log probabilities, also interpretable
        as the sensitivity of the model's certainty to changes in `X`
        or the model's score function

        """
        crossent_w_1 = self.cross_entropy_with(tf.ones_like(self.y) / self.num_classes)
        return tf.gradients(crossent_w_1, self.x)[0]
    def l1_certainty_sensitivity(self):
        """Cache the L1 loss of that product"""
        return tf.reduce_mean(tf.abs(self.certainty_sensitivity()))
    def l2_certainty_sensitivity(self):
        """Cache the L2 loss of that product"""
        return tf.nn.l2_loss(self.certainty_sensitivity())
    def cross_entropy(self):
        """Symbolic TF variable returning information distance between the
        model's predictions and the true labels y

        """
        return self.cross_entropy_with(self.y)
    def cross_entropy_grads(self):
        """Symbolic TF variable returning the input gradients of the cross
        entropy.  Note that if you pass in y=(1^{NxK})/K, this returns
        the same value as certainty_sensitivity.

        """
        # return tf.gradients(self.cross_entropy(), self.x)[0]
        # FIXME why [0]
        return tf.gradients(self.cross_entropy(), self.x)

    def l1_double_backprop(self):
        """L1 loss of the sensitivity of the cross entropy"""
        return tf.reduce_sum(tf.abs(self.cross_entropy_grads()))

    def l2_double_backprop(self):
        """L2 loss of the sensitivity of the cross entropy"""
        # tf.sqrt(tf.reduce_mean(tf.square(grads)))
        return tf.nn.l2_loss(self.cross_entropy_grads())
    def gradients_group_lasso(self):
        grads = tf.abs(self.cross_entropy_grads())
        return tf.reduce_mean(tf.sqrt(
            keras.layers.AvgPool3D(4, 4, 'valid')(tf.square(grads))
            + 1e-10))
    
class MNIST_CNN(CNNModel):
    def x_shape(self):
        # return [None, 28*28]
        return (28,28,1,)
    def y_shape(self):
        return [None, 10]
        # return (10,)
    def rebuild_model(self, xin):
        # FIXME may not need reshape layer
        x = keras.layers.Reshape([28, 28, 1])(xin)
        x = keras.layers.Conv2D(32, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(32, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)

        x = keras.layers.Conv2D(64, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, 3)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D((2,2))(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(200)(x)
        x = keras.layers.Activation('relu')(x)
        logits = keras.layers.Dense(10)(x)
        return logits
    
# model = AdvAEModel()
        optimizer = keras.optimizers.Adam(0.001)
        # 'adadelta'
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[loss, acc])
        with sess.as_default():
            es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=10, verbose=0,
                                               mode='auto')
            mc = keras.callbacks.ModelCheckpoint('best_model.ckpt',
                                                 monitor='val_loss', mode='auto', verbose=0,
                                                 save_weights_only=True,
                                                 save_best_only=True)
            model.fit(clean_x, clean_y, epochs=100, validation_split=0.1,
                      # verbose=2,
                      callbacks=[es, mc])
            model.load_weights('best_model.ckpt')
    def train_CNN_old(self, sess, train_x, train_y):
        """Train CNN part of the graph."""
        # TODO I'm going to compute a hash of current CNN
        # architecture. If the file already there, I'm just going to
        # load it.
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        logits = self.FC(self.CNN(inputs))
        model = keras.models.Model(inputs, logits)
        def loss(y_true, y_pred):
            return tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=y_pred, labels=y_true))
        optimizer = keras.optimizers.Adam(0.001)
        # 'rmsprop'
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])
        with sess.as_default():
            es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=10, verbose=0,
                                               mode='auto')
            mc = keras.callbacks.ModelCheckpoint('best_model.ckpt',
                                                 monitor='val_loss', mode='auto', verbose=0,
                                                 save_weights_only=True,
                                                 save_best_only=True)
            model.fit(clean_x, clean_y, epochs=100, validation_split=0.1,
                      # verbose=2,
                      callbacks=[es, mc])
            model.load_weights('best_model.ckpt')
    def train_AE_simple(self, sess, clean_x):
        noise_factor = 0.1
        noisy_x = clean_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_x.shape)
        noisy_x = np.clip(noisy_x, CLIP_MIN, CLIP_MAX)
        self.AE.compile('adam', 'binary_crossentropy')
        with sess.as_default():
            self.AE.fit(noisy_x, clean_x, verbose=2)
        
    def train_AE_old(self, sess, clean_x):
        noise_factor = 0.1
        noisy_x = clean_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_x.shape)
        noisy_x = np.clip(noisy_x, CLIP_MIN, CLIP_MAX)

        inputs = keras.layers.Input(shape=(28,28,1,), dtype='float32')
        rec = self.AE(inputs)
        model = keras.models.Model(inputs, rec)
        optimizer = keras.optimizers.RMSprop(0.001)
        # 'adadelta'
        model.compile(optimizer=optimizer,
                      # loss='binary_crossentropy'
                      loss=keras.losses.mean_squared_error
        )
        with sess.as_default():
            es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=5, verbose=0,
                                               mode='auto')
            mc = keras.callbacks.ModelCheckpoint('best_model.ckpt',
                                                 monitor='val_loss', mode='auto', verbose=0,
                                                 save_weights_only=True,
                                                 save_best_only=True)
            model.fit(noisy_x, clean_x, epochs=100, validation_split=0.1, verbose=2, callbacks=[es, mc])
            model.load_weights('best_model.ckpt')
    # def setup_AE(self):
    #     """From noise_x to x."""
    #     inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
    #     x = keras.layers.Flatten()(inputs)
    #     x = keras.layers.Dense(100, activation='relu')(x)
    #     x = keras.layers.Dense(20, activation='relu')(x)
    #     x = keras.layers.Dense(100, activation='relu')(x)
    #     x = keras.layers.Dense(28*28, activation='sigmoid')(x)
    #     decoded = keras.layers.Reshape([28,28,1])(x)
    #     self.AE = keras.models.Model(inputs, decoded)
        
        
    def save_AE(self, sess, path):
        """Save AE weights to checkpoint."""
        saver = tf.train.Saver(var_list=self.AE_vars)
        saver.save(sess, path)


    def load_AE(self, sess, path):
        saver = tf.train.Saver(var_list=self.AE_vars)
        saver.restore(sess, path)
    def save_CNN(self, sess, path):
        """Save AE weights to checkpoint."""
        saver = tf.train.Saver(var_list=self.CNN_vars + self.FC_vars)
        saver.save(sess, path)
            
    def load_CNN(self, sess, path):
        saver = tf.train.Saver(var_list=self.CNN_vars + self.FC_vars)
        saver.restore(sess, path)
class AE2_Model(AEModel):
    def setup_AE(self):
        
        inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        # denoising
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = keras.layers.Conv2D(32, (3, 3), padding='same')(x)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (7, 7, 32)
        # (?, 8, 8, 32) for cifar10

        x = keras.layers.Conv2D(32, (3, 3), padding='same')(encoded)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(x)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        
        # here the fisrt channel is 3 for CIFAR model
        x = keras.layers.Conv2D(3, (3, 3), padding='same')(x)
        # x = keras.layers.BatchNormalization()(x)
        
        self.AE1 = keras.models.Model(inputs, x)
        
        decoded = keras.layers.Activation('sigmoid')(x)
        self.AE = keras.models.Model(inputs, decoded)
def test_model_against_attack(sess, model, attack_func, inputs, labels, targets, prefix=''):
    """
    testing against several attacks. Return the atatck accuracy and L2 distortion.
    """
    # generate victims
    adv_x = attack_func(sess, model)
    # adv_x = my_CW(sess, model)
    adv_preds = model.predict(adv_x)
    if targets is None:
        adv_x_concrete = sess.run(adv_x, feed_dict={model.x: inputs, model.y: labels})
        adv_preds_concrete = sess.run(adv_preds, feed_dict={model.x: inputs, model.y: labels})
        
        correct_indices = np.equal(np.argmax(adv_preds_concrete, 1), np.argmax(labels, 1))
        incorrect_indices = np.not_equal(np.argmax(adv_preds_concrete, 1), np.argmax(labels, 1))
        
        incorrect_x = inputs[incorrect_indices]
        incorrect_adv_x = adv_x_concrete[incorrect_indices]
        incorrect_y = labels[incorrect_indices]
        incorrect_pred = adv_preds_concrete[incorrect_indices]
        
        # FIXME seems the tf acc is different from np calculated acc
        adv_acc = model.compute_accuracy(sess, model.x, model.y, adv_preds, inputs, labels)
    else:
        # FIXME incorrect instances
        adv_x_concrete = sess.run(adv_x, feed_dict={model.x: inputs, model.y: targets})
        adv_preds_concrete = sess.run(adv_preds, feed_dict={model.x: inputs, model.y: targets})
        adv_acc = model.compute_accuracy(sess, model.x, model.y, model.logits, adv_x_concrete, labels)

    l2 = np.mean(mynorm(inputs, adv_x_concrete, 2))
    if attack_func == my_CW:
        # CW l2 is a bit problematic, because in case it fails, it
        # will output the same value, and thus 0 l2. In general, I
        # need to calculate L2 only for successful attacks.
        l2old = l2
        l2 = np.mean(mynorm(inputs[incorrect_indices], adv_x_concrete[incorrect_indices], 2))
        print('CW, calculate l2 only for incorrect examples {} (old l2: {}).'.format(l2, l2old))

    filename = 'attack-result-{}.pdf'.format(prefix)
    print('Writing adv examples to {} ..'.format(filename))
    # correct: row1: clean, row2: adv
    images = np.concatenate((inputs[correct_indices][:5],
                             inputs[incorrect_indices][:5],
                             adv_x_concrete[correct_indices][:5],
                             adv_x_concrete[incorrect_indices][:5]))
    titles = np.concatenate((np.argmax(labels[correct_indices][:5], 1),
                             np.argmax(labels[incorrect_indices][:5], 1),
                             np.argmax(adv_preds_concrete[correct_indices][:5], 1),
                             np.argmax(adv_preds_concrete[incorrect_indices][:5], 1)))
    grid_show_image(images, int(len(images)/2), 2, filename=filename, titles=titles)
    
    # print('True label: {}'.format(np.argmax(labels, 1)))
    # print('Predicted label: {}'.format(np.argmax(adv_preds_concrete, 1)))
    # print('Adv input accuracy: {}'.format(adv_acc))
    # print("L2: {}".format(l2))
    if targets is not None:
        target_acc = model.compute_accuracy(sess, model.x, model.y, model.logits, adv_x_concrete, targets)
        print('Targeted label: {}'.format(np.argmax(targets, 1)))
        print('Targeted accuracy: {}'.format(target_acc))
    return adv_acc, l2

def test_against_attacks(sess, model):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
     # all 10000 inputs
    inputs, labels = test_x, test_y
    # random 100 inputs
    indices = random.sample(range(test_x.shape[0]), 100)
    inputs = test_x[indices]
    labels = test_y[indices]

    print('testing FGSM ..')
    acc, l2 = test_model_against_attack(sess, model, my_FGSM, inputs, labels, None, prefix='FGSM')
    print('{} acc: {}, l2: {}'.format('FGSM', acc, l2))
    
    print('testing PGD ..')
    acc, l2 = test_model_against_attack(sess, model, my_PGD, inputs, labels, None, prefix='PGD')
    print('{} acc: {}, l2: {}'.format('PGD', acc, l2))
    
    print('testing JSMA ..')
    acc, l2 = test_model_against_attack(sess, model, my_JSMA, inputs, labels, None, prefix='JSMA')
    print('{} acc: {}, l2: {}'.format('JSMA', acc, l2))
    
    print('testing CW ..')
    acc, l2 = test_model_against_attack(sess, model, my_CW, inputs, labels, None, prefix='CW')
    print('{} acc: {}, l2: {}'.format('CW', acc, l2))

        # to_run['baseline_pred'] = tf.argmax(baseline_logits, axis=1)
        # to_run['baseline_acc'] = baseline_acc

        # to_run['obliadv_pred'] = tf.argmax(obliadv_logits, axis=1)
        # to_run['obliadv_acc'] = obliadv_acc

        # to_run['whiteadv_pred'] = tf.argmax(whiteadv_logits, axis=1)
        # to_run['whiteadv_acc'] = whiteadv_accuracy

        # to_run['postadv_pred'] = tf.argmax(postadv_logits, axis=1)
        # to_run['postadv_acc'] = postadv_accuracy

        # print('baseline_acc: {}, obliadv_acc: {}, whiteadv_acc: {}, postadv_acc: {}'
        #       .format(res['obliadv'][2], res['obliadv_rec'][2],
        #               res['whiteadv'][2], res['postadv'][2]))
        
        # images += list(res['adv_x'][:10])
        # titles += ['adv_x'] + ['']*9

    # clses = [
    #     # default model
    #     # ONE
    #     B2_Model,
    #     A2_Model,
    #     # A1_Model,
    #     # A12_Model,
    #     # TWO
    #     N0_A2_Model,
    #     C0_A2_Model,
    #     # C2_A2_Model,
    #     # THREE
    #     C0_N0_A2_Model,
    #     # C2_N0_A2_Model
    # ]
    # for cls in clses:
    #     train_denoising(cls, train_x, train_y, prefix='mnist', advprefix='mnist-' + cls.name())
    
    # train_denoising(FCAdvAEModel, 'saved_models', 'fccnn', 'FCAdvAE')
    
    # train_denoising(PostNoisy_Adv_Model, 'saved_models', 'aecnn', 'PostNoisy_Adv')
        # target = self.x
        # output = noisy_x

        # tmp = tf.clip_by_value(output, tf.epsilon(), 1.0 - tf.epsilon())
        # tmp = -target * tf.log(tmp) - (1.0 - target) * tf.log(1.0 - tmp)
        # loss = tf.reduce_mean(tmp, 1)
        
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        
        # loss = tf.reduce_mean(keras.backend.binary_crossentropy(self.AE(noisy_x), self.x), 1)
        
    def train_CNN_keras(self, sess, train_x, train_y, test_x, test_y):
        with sess.as_default():
            # FIXME maybe the performance is caused by the data size?
            orig_test_x = test_x
            orig_test_y = test_y
            (train_x, train_y), (test_x, test_y) = validation_split(train_x, train_y)
            
            # outputs = Dense(num_classes,
            #                 activation='softmax',
            #                 kernel_initializer='he_normal')(y)

            # model = self.model()
            # keras.losses.categorical_crossentropy

            def loss(y_true, y_pred):
                return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)


            
            # preds = keras.layers.Dense(10,
            #                 activation='softmax',
            #                 kernel_initializer='he_normal')(keras.layers.Flatten()(self.CNN(self.x)))
            # truemodel = keras.models.Model(self.x, preds)
            
            model = self.model
            # model = truemodel

            model.compile(
                # loss='categorical_crossentropy',
                loss=loss,
                optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                metrics=['accuracy'])
            # model.summary()
            lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

            # lr_reducer = keras.callbacks.ReduceLROnPlateau(
            #     factor=np.sqrt(0.1),
            #     cooldown=0,
            #     patience=5,
            #     min_lr=0.5e-6)

            # callbacks = [lr_reducer, lr_scheduler]
            callbacks = [lr_scheduler]
            
            batch_size = 32  # orig paper trained all networks with batch_size=128
            epochs = 200
            

            datagen = self.get_generator()
            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(train_x)

            # Fit the model on the batches generated by datagen.flow().
            self.model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
                                     validation_data=(test_x, test_y),
                                     # validation_split=0.9,
                                     epochs=epochs, verbose=1, workers=4,
                                     steps_per_epoch=math.ceil(train_x.shape[0] / batch_size),
                                     callbacks=callbacks)
            # model.fit(train_x, train_y,
            #           batch_size=batch_size,
            #           epochs=epochs,
            #           validation_data=(test_x, test_y),
            #           # validation_split=0.9,
            #           shuffle=True,
            #           callbacks=callbacks)
            # Score trained model.
            # scores = self.model.evaluate(test_x, test_y, verbose=1)
            scores = self.model.evaluate(orig_test_x, orig_test_y, verbose=1)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
    def get_generator(self):
            # This will do preprocessing and realtime data augmentation:
            datagen = keras.preprocessing.image.ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
            return datagen
    def setup_trainstep_test(self):
        self.CNN_train_step = tf.train.AdamOptimizer(0.001).minimize(
            self.CNN_loss, var_list=self.CNN_vars+self.FC_vars)
class KerasAugment():
    def __init__(self, train_x):
        """train_x is the training data you are going to use. The generator
will fit on these data first."""
        # This will do preprocessing and realtime data augmentation:
        self.datagen = keras.preprocessing.image.ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        self.datagen.fit(train_x)

    def augment_batch(self, sess, batch_x):
        return self.datagen.flow(batch_x, batch_size=1)

def __test():
    sys.path.append('/home/hebi/github/reading/')
    from cifar10_challenge import model as madry_resnet
    madrynet = madry_resnet.Model(mode='train')
    
    madrynet.x_input
    keras.layers.Lambda()

    sys.path.append('/home/hebi/github/reading/tensorflow-models')
    from official.resnet.keras.resnet_cifar_model import resnet56
    # the problem of using this model is that, the softmax is
    # applied. I have to modify the function to get a pre-softmax
    # tensor.

   

def __test():
    from tensorflow.python.tools import inspect_checkpoint as chkp
    chkp.print_tensors_in_checkpoint_file('saved_model/AE.ckpt', tensor_name='', all_tensors=True)

def __test():
    train_double_backprop("saved_model/double-backprop.ckpt")
    train_group_lasso("saved_model/group-lasso.ckpt")
    
    sess, model = restore_model("saved_model/double-backprop.ckpt")
    sess, model = restore_model("saved_model/group-lasso.ckpt")
    sess, model = restore_model("saved_model/ce.ckpt")
    sess, model = restore_model("saved_model/adv.ckpt")
    # sess, model = restore_model("saved_model/adv2.ckpt")

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    
    # testing model accuracy on clean data
    model.compute_accuracy(sess, model.x, model.y, model.logits, test_x, test_y)
    
    # 10 targeted inputs for demonstration
    inputs, labels, targets = generate_victim(test_x, test_y)
    # all 10000 inputs
    inputs, labels = test_x, test_y
    # random 100 inputs
    indices = random.sample(range(test_x.shape[0]), 100)
    inputs = test_x[indices]
    labels = test_y[indices]
    
    test_model_against_attack(sess, model, my_FGSM, inputs, labels, None, prefix='FGSM')
    test_model_against_attack(sess, model, my_PGD, inputs, labels, None, prefix='PGD')
    test_model_against_attack(sess, model, my_JSMA, inputs, labels, None, prefix='JSMA')
    # test_model_against_attack(sess, model, my_CW_targeted, inputs, labels, targets, prefix='CW_targeted')
    test_model_against_attack(sess, model, my_CW, inputs, labels, None, prefix='CW')
    
def my_distillation():
    """1. train a normal model, save the weights
    
    2. teacher model: load the weights, train the model with ce loss
    function but replace: logits=predicted/train_temp(10)
    
    3. predicted = teacher.predict(data.train_data)
       y = sess.run(tf.nn.softmax(predicted/train_temp))
       data.train_labels = y
    
    4. Use the updated data to train the student. The student should
    load the original weights. Still use the loss function with temp.

    https://github.com/carlini/nn_robust_attacks

    """
    pass


class TestA():
    def __init__(self, x):
        self.a = 1
        self.x = x
        self.foo()
        self.bar(x)
    def foo(self):
        self.b = 2
    def bar(self, xx):
        self.xx = xx
class TestB(TestA):
    def foo(self):
        self.b = 3
class TestC(TestA):
    def __init__(self, x):
        super().__init__(x)
    def foo(self):
        self.b = 4

def __test():
    TestA(1).b
    TestB(2).a
    TestB(2).b
    TestB(8).x
    TestB(8).xx
    TestC(4).a
    TestC(5).b
    TestC(3).x
def restore_model(path):
    tf.reset_default_graph()
    sess = tf.Session()
    model = MNIST_CNN()
    saver = tf.train.Saver()
    saver.restore(sess, path)
    print("Model restored.")
    return sess, model

    
def __test():
    sess = tf.Session()
    model = MNIST_CNN()
    loss = model.cross_entropy()
    my_adv_training(sess, model, loss)
    
def train(sess, model, loss):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_mnist_data()
    
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    best_loss = math.inf
    best_epoch = 0
    patience = 5
    print_interval = 500

    init = tf.global_variables_initializer()
    sess.run(init)
    
    saver = tf.train.Saver(max_to_keep=10)
    for i in range(NUM_EPOCHS):
        shuffle_idx = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_idx)
        nbatch = train_x.shape[0] // BATCH_SIZE
        
        for j in range(nbatch):
            start = j * BATCH_SIZE
            end = (j+1) * BATCH_SIZE
            batch_x = train_x[shuffle_idx[start:end]]
            batch_y = train_y[shuffle_idx[start:end]]
            _, l, a = sess.run([train_step, loss, model.accuracy],
                               feed_dict={model.x: batch_x,
                                          model.y: batch_y})
        print('EPOCH ends, calculating total loss')
        l, a = sess.run([loss, model.accuracy],
                        feed_dict={model.x: val_x,
                                   model.y: val_y})
        print('EPOCH {}: loss: {:.5f}, acc: {:.5f}' .format(i, l, a,))
        save_path = saver.save(sess, 'tmp/epoch-{}'.format(i))
        if best_loss < l:
            patience -= 1
            if patience <= 0:
                # restore the best model
                saver.restore(sess, 'tmp/epoch-{}'.format(best_epoch))
                print('Early stopping .., best loss: {}'.format(best_loss))
                # verify if the best model is restored
                l, a = sess.run([loss, model.accuracy],
                                feed_dict={model.x: val_x,
                                           model.y: val_y})
                assert l == best_loss
                break
        else:
            best_loss = l
            best_epoch = i
            patience = 5

        
def train_double_backprop(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        c=100
        # TODO verify the results in their paper
        loss = model.cross_entropy() + c * model.l2_double_backprop()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)


def train_group_lasso(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        # group lasso
        c=100
        loss = model.cross_entropy() + c * model.gradients_group_lasso()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

def train_ce(path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST_CNN()
        loss = model.cross_entropy()
        train(sess, model, loss)
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)

    def train_CNN_keras(self, sess, train_x, train_y):
        """FIXME WRN seems to work on keras preprocessor very well."""
        # generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
        #                                                          width_shift_range=5./32,
        #                                                          height_shift_range=5./32,)
        # model.summary()
        # plot_model(model, "WRN-16-8.png", show_shapes=False)
        (train_x, train_y), (val_x, val_y) = validation_split(train_x, train_y)

        # inputs = keras.layers.Input(shape=self.xshape(), dtype='float32')
        outputs = keras.layers.Activation('softmax')(self.model(self.x))
        model = keras.models.Model(self.x, outputs)
        with sess.as_default():
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
            batch_size = 100
            # model.fit_generator(generator.flow(train_x, train_y, batch_size=batch_size),
            #                     steps_per_epoch=len(train_x) // batch_size, epochs=100,
            #                    validation_data=(val_x, val_y),
            #                    validation_steps=val_x.shape[0] // batch_size,)
            keras.backend.set_learning_phase(1)
            model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=2)
            keras.backend.set_learning_phase(0)
            print(model.evaluate(val_x, val_y))
    def train_AE_keras(self, sess, train_x, train_y):
        noise_factor = 0.1
        noisy_x = train_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_x.shape)
        noisy_x = np.clip(noisy_x, CLIP_MIN, CLIP_MAX)

        inputs = keras.layers.Input(shape=self.shape, dtype='float32')
        rec = self.AE(inputs)
        model = keras.models.Model(inputs, rec)
        
        optimizer = keras.optimizers.RMSprop(0.001)
        # 'adadelta'
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      # loss=keras.losses.mean_squared_error
                      # metrics=[self.noisy_accuracy]
        )
        with sess.as_default():
            model.fit(noisy_x, train_x, epochs=100, validation_split=0.1, verbose=2)

        self.x = keras.layers.Input(shape=self.cnn_model.xshape(), dtype='float32')
        self.y = keras.layers.Input(shape=self.cnn_model.yshape(), dtype='float32')

        noisy_x = my_add_noise(self.x, noise_factor=0.5)

        # print(self.AE1(noisy_x).shape)
        # print(self.x.shape)

        ##############################
        ## noisy
        
        # noisy, xent, pixel, (0.36)
        # loss = my_sigmoid_xent(self.AE1(noisy_x), self.x)
        
        # noisy, l2, pixel, 0.3
        # loss = tf.reduce_mean(tf.square(self.AE(noisy_x) - self.x))
        
        # noisy, l1, pixel
        # loss = tf.reduce_mean(tf.abs(self.AE(noisy_x) - self.x))

        # noisy+clean, l1, pixel
        # testing whether noisy loss can be used as regularizer
        # 0.42
        # mu = 0.5
        # 0.48
        mu = 0.2
        loss = (mu * tf.reduce_mean(tf.abs(self.AE(noisy_x) - self.x))
                + (1 - mu) * tf.reduce_mean(tf.abs(self.AE(self.x) - self.x)))
        ###############################
        ## Clean

        # xent, pixel, 0.515 (0.63)
        # loss = my_sigmoid_xent(self.AE1(self.x), self.x)
        # self.C0_loss = my_sigmoid_xent(self.AE1(self.x), self.x)
        # l2, pixel, 0.48
        # loss = tf.reduce_mean(tf.square(self.AE(self.x) - self.x))
        # l1, pixel, 0.52
        # loss = tf.reduce_mean(tf.abs(self.AE(self.x) - self.x))


        # L2, logits, .79
        # loss = tf.reduce_mean(tf.square(self.cnn_model.predict(self.AE(self.x)) -
        #                                 self.cnn_model.predict(self.x)))
        # L1, logits, .30
        # loss = tf.reduce_mean(tf.abs(self.cnn_model.predict(self.AE(self.x)) -
        #                              self.cnn_model.predict(self.x)))
        # L2, feature, 0.39
        # loss = tf.reduce_mean(tf.square(self.cnn_model.CNN(self.AE(self.x)) -
        #                                 self.cnn_model.CNN(self.x)))
        # L1, feature, .445
        # loss = tf.reduce_mean(tf.abs(self.cnn_model.CNN(self.AE(self.x)) -
        #                              self.cnn_model.CNN(self.x)))


        # xent, logits, .30
        # loss = my_sigmoid_xent(self.cnn_model.predict(self.AE(self.x)),
        #                        tf.nn.sigmoid(self.cnn_model.predict(self.x)))
        # xent, label, .38
        # loss = my_softmax_xent(self.cnn_model.predict(self.AE(self.x)),
        #                        self.y)
        
        # self.C1_loss = my_sigmoid_xent(rec_high, tf.nn.sigmoid(high))
        # self.C2_loss = my_softmax_xent(rec_logits, self.y)
        
        self.AE_loss = loss
        
        self.noisy_accuracy = my_accuracy_wrapper(self.cnn_model.predict(self.AE(noisy_x)),
                                                  self.y)

        self.accuracy = my_accuracy_wrapper(self.cnn_model.predict(self.AE(self.x)),
                                            self.y)

        self.clean_accuracy = my_accuracy_wrapper(self.cnn_model.predict(self.x),
                                                  self.y)

        self.setup_trainstep()
    def setup_trainstep(self):
        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step
        # These numbers are batches
        schedules = [[0, 1e-3], [400*20, 1e-4], [400*40, 1e-5], [400*60, 1e-6]]
        boundaries = [s[0] for s in schedules][1:]
        values = [s[1] for s in schedules]
        learning_rate = tf.train.piecewise_constant_decay(
            tf.cast(global_step, tf.int32),
            boundaries,
            values)
        self.AE_train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(
            # 1e-3).minimize(
                self.AE_loss,
                global_step=global_step,
                var_list=self.AE_vars)
        # self.AE_train_step = tf.train.AdamOptimizer(0.01).minimize(
        #     self.AE_loss, global_step=global_step, var_list=self.AE_vars)
    def train_AE(self, sess, train_x, train_y):
        """DEPRECATED Technically I don't need clean y, but I'm going to
monitor the noisy accuracy."""
        raise NotImplementedError()
        my_training(sess, self.x, self.y,
                    self.AE_loss, self.AE_train_step,
                    [self.noisy_accuracy, self.accuracy, self.clean_accuracy, self.global_step],
                    ['noisy_acc', 'accuracy', 'clean_accuracy', 'global step'],
                    train_x, train_y,
                    # Seems that the denoiser converge fast
                    patience=5,
                    additional_train_steps=[self.AE.updates],
                    print_interval=50,
                    # default: 100
                    # num_epochs=60,
                    do_plot=False)
    def test_AE(self, sess, test_x, test_y):
        """Test AE models."""
        # select 10
        images = []
        titles = []
        fringes = []

        test_x = test_x[:100]
        test_y = test_y[:100]

        to_run = {}
            
        # Test AE rec output before and after
        rec = self.AE(self.x)
        noisy_x = my_add_noise(self.x)
        noisy_rec = self.AE(noisy_x)

        to_run['x'] = self.x
        to_run['y'] = tf.argmax(self.y, 1)
        to_run['rec'] = rec
        to_run['noisy_x'] = noisy_x
        to_run['noisy_rec'] = noisy_rec
        print('Testing CNN and AE ..')
        res = sess.run(to_run, feed_dict={self.x: test_x, self.y: test_y})

        
        images.append(res['x'][:10])
        titles.append(res['y'][:10])
        fringes.append('x')
        
        for name in ['rec', 'noisy_x', 'noisy_rec']:
            images.append(res[name][:10])
            titles.append(['']*10)
            fringes.append('{}'.format(name))

        print('Plotting result ..')
        grid_show_image(images, filename='out.pdf', titles=titles, fringes=fringes)
        print('Done. Saved to {}'.format('out.pdf'))
    def train_AE(self, sess, train_x, train_y):
        print('Dummy training, do nothing.')
        pass
