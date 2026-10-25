"""Independent RGB CNN baseline trained from scratch."""


def build(learning_rate=0.001, seed=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.Input(shape=(128, 128, 3), dtype="float32")]
    for filters in (16, 32, 64):
        layers.extend([tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
                       tf.keras.layers.MaxPooling2D(2)])
    layers.extend([tf.keras.layers.GlobalAveragePooling2D(),
                   tf.keras.layers.Dense(64, activation="relu"),
                   tf.keras.layers.Dropout(0.30, seed=seed),
                   tf.keras.layers.Dense(5, activation="softmax")])
    model = tf.keras.Sequential(layers, name="cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    return model
