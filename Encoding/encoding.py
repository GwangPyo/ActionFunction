import tensorflow as tf
from keras.layers import BatchNormalization, Activation, Lambda, Dense, Reshape, LeakyReLU, Input, Flatten
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from PIL import Image
import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy

BATCH_SIZE = 64


class AAE(object):
    def __init__(self, shape, latent_dim=48, channel=4):
        self.shape= shape
        self.shape_name = shape[0]
        self.name="AAE"
        self.img_rows = shape[0]
        self.img_cols = shape[1]
        self.channels = channel
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        optimizer = Adam(0.0003, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.encoder = self.build_encoder()
        self.encoder.summary()
        self.decoder = self.build_decoder()
        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.9999, 1e-4],
            optimizer=optimizer)

    def build_decoder(self):
        # shared weights
        model = Sequential()
        model.add(Dense(48, input_dim=self.latent_dim, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape((3, 4, 4)))
        model.add(Conv2DTranspose(filters=15,
                   kernel_size=2,
                   strides=(2, 4),
                   padding='same', activation='relu'))
        model.add(Conv2DTranspose(filters=10,
                   kernel_size=5,
                   strides=(7, 5),
                   padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=self.channels,
                   kernel_size=5,
                   strides=(5, 2),
                   padding='same'))
        model.add(Activation("sigmoid"))
        model.summary()
        z = Input(shape=(self.latent_dim, ))
        img = model(z)
        return Model(z, img)

    def build_discriminator(self):
        model = Sequential()
        # model.add(Input(shape=self.img_shape, name='encoder_input'))
        """
        model.add(Conv2D(filters=32,
                   kernel_size=7,
                   activation='relu',
                   strides=2,
                   padding='same'))
        model.add(Conv2D(filters=8, kernel_size=(1, 1)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(filters=4, kernel_size=2, strides=2,padding="same"))
        model.add(Flatten())
        """
        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1, activation="sigmoid"))
        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def build_encoder(self):
        img = Input(shape=self.img_shape)
        h = Conv2D(filters=55,
                   kernel_size=(16, 10),
                   activation='relu',
                   strides=9)(img)
        h = Conv2D(filters=5, kernel_size=(1, 1))(h)
        h = Conv2D(filters=15,
                   kernel_size=5,
                   activation='relu',
                   strides=2)(h)
        h = Conv2D(filters=5, kernel_size=(1, 1))(h)
        h = Flatten()(h)
        h = Dense(48, activation="elu")(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(lambda p: p[0] + tf.random_normal(K.shape(p[0])) * tf.exp(p[1] / 2),
                output_shape=lambda p: p[0])([mu, log_var])
        return Model(img, latent_repr)

    def save(self):
        self.decoder.save("decoder.h5".format(self.shape_name))
        self.encoder.save("encoder.h5".format(self.shape_name))
        self.discriminator.save("discriminator.h5".format(self.shape_name))

    def get_generator(self):
        return self.decoder

    def load(self):
        self.decoder.load_weights("decoder.h5".format(self.shape_name))
        self.encoder.load_weights("encoder.h5".format(self.shape_name))
        try:
            self.discriminator.load_weights("discriminator.h5".format(self.shape_name))
        except OSError:
            pass

    def load_data(self, path=None):
        if path is None:
            data = np.load("{}_dataset/query.npy".format(self.img_rows))
        else:
            data = np.load(path)
        return data

    def train(self, epochs, batch_size=128, sample_interval=50, data_path=None):

        # Load the dataset
        X_train = self.load_data(data_path)
        X_train = X_train / 255
        X_train = np.squeeze(X_train)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])


            # Plot the progress
            if epoch % 100 == 0:
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 :
                encoded_repr = self.encoder.predict_on_batch(imgs)
                img = self.decoder.predict_on_batch(encoded_repr)
                for i in range(5):
                    im = img[i][:, :, 0]
                    im = im * 255
                    im = np.uint8(im)
                    img_s = Image.fromarray(im, "L")
                    img_s.save("samples/img_{}_{}.png".format(epoch, i))

    def sample_images(self, epoch):

        noise = np.random.normal(0, 1, (10, self.latent_dim))
        """
        dev = np.random.uniform(low=0.5, high=1.5)
        noise = np.random.normal(0, dev, (1, self.latent_dim))
        for i in range(10):
            dev = np.random.uniform(low=0.5, high=1.5)
            noise_ = np.random.normal(0, dev, (1, self.latent_dim))
            noise = np.concatenate((noise, noise_))
        """
        AAE_IMG = self.decoder.predict(noise)
        # Rescale images 0 - 1
        AAE_IMG = 0.5 * AAE_IMG + 0.5
        AAE_IMG = np.uint8(AAE_IMG * 255)

        return AAE_IMG, epoch


def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon




class VAE(object):
    def __init__(self, shape, latent_dim=8, channel=4):
        self.shape= shape
        self.shape_name = shape[0]
        self.name="AAE"
        self.img_rows = shape[0]
        self.img_cols = shape[1]
        self.channels = channel
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        inputs = Input(shape=self.img_shape)
        z_mean, z_log_var = self.encoder(inputs)
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        outputs = self.decoder(z)

        self.vae = Model(input=inputs, output=outputs, name="VAEMlp")
        """
        Losses 
        """
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= self.img_cols * self.img_rows * self.channels

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)
        """
        compile 
        """
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer="Adam")
        self.vae.summary()

    def build_encoder(self):
        img = Input(shape=(20, 80, 4))
        h = Conv2D(filters=25,
                   kernel_size=5,
                   activation='relu',
                   strides=2,
                   padding='same')(img)
        h = Conv2D(filters=15, kernel_size=(1, 1))(h)
        h = Conv2D(filters=15,
                   kernel_size=5,
                   activation='relu',
                   strides=2,
                   padding='same')(h)
        h = Flatten()(h)
        h = Dense(64, activation="elu")(h)
        mu = Dense(self.latent_dim, name="z_mean")(h)
        log_var = Dense(self.latent_dim, name="z_log_var")(h)
        return Model(input=img, output=[mu, log_var])

    def build_decoder(self):
        # shared weights

        z = Input(shape=(self.latent_dim, ))
        x  = Dense(32,  activation="relu")(z)
        x = BatchNormalization(momentum=0.8)(x)
        x = Reshape((2, 4, 4))(x)
        x = Conv2DTranspose(filters=15,
                   kernel_size=5,
                   strides=(2, 4),
                   padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=10,
                   kernel_size=5,
                   strides=(4, 5),
                   padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        img = Conv2DTranspose(filters=3,
                   kernel_size=5,
                   strides=(5, 5),
                   padding='same', activation="tanh")(x)
        return Model(input=z, output=img)


class AtariEmbedding(object):
    def __init__(self, aae_object):
        self.embedding = aae_object.encoder

    def __call__(self, *args, **kwargs):
        self.embedding.predict_on_batch(*args, **kwargs)





if __name__ == '__main__':
    aae = AAE(shape=(210, 160), channel=3)
    cnt = 6
    while cnt < 10:
        aae.train(epochs=10000, sample_interval=1000, batch_size=2 ** cnt, data_path="obs_capture.npy")
        aae.save()
        cnt += 1