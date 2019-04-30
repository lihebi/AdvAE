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
