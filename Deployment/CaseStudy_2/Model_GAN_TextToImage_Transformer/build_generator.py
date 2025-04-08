from tensorflow.keras import layers, models

def build_generator(latent_dim, num_classes):
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    merged_input = layers.Concatenate()([noise_input, label_embedding])

    x = layers.Dense(256)(merged_input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)

    x = layers.Dense(128 * 7 * 7)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Reshape((7, 7, 128))(x)

    for filters in [128, 64, 64, 32, 32]:
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    img_output = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)

    return models.Model([noise_input, label_input], img_output, name="Generator")
