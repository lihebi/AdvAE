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

