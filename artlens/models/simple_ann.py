"""Paper section III, figures 3/4: one biased weighted layer with sigmoid outputs."""


def build(learning_rate=0.001, seed=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2048,), dtype="float32"),
        tf.keras.layers.Dense(5, activation="sigmoid", use_bias=True,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                              bias_initializer="zeros"),
    ], name="simple_ann")
    # BCE, SGD and initialization are ArtLens conventions, not paper specifications.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    return model
