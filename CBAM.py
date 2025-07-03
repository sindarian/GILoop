from tensorflow.keras.layers import Layer, Dense, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Multiply, Add, Activation, Concatenate, Lambda
import tensorflow as tf

class CBAM(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_one = Dense(channel // self.reduction_ratio,
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True)
        self.shared_dense_two = Dense(channel,
                                      kernel_initializer='he_normal',
                                      use_bias=True)
        self.conv_spatial = Conv2D(filters=1,
                                   kernel_size=7,
                                   padding='same',
                                   activation='sigmoid',
                                   kernel_initializer='he_normal')
        super(CBAM, self).build(input_shape)

    def compute_channel_attention(self, x):
        # average pooling path
        avg_pool = GlobalAveragePooling2D()(x)
        avg_pool = Reshape((1, 1, -1))(avg_pool)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))

        # max pooling path
        max_pool = GlobalMaxPooling2D()(x)
        max_pool = Reshape((1, 1, -1))(max_pool)
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

        # combine features from average and max pooling paths
        cbam_feature = Activation('sigmoid')(Add()([avg_out, max_out]))
        return Multiply()([x, cbam_feature])

    def compute_spatial_attention(self, x):
        avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        scale = self.conv_spatial(concat)
        return Multiply()([x, scale])

    def call(self, inputs):
        x = self.compute_channel_attention(inputs)
        x = self.compute_spatial_attention(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config
