from google.colab import drive
drive.mount('/content/drive')
folder_path = "/content/drive/MyDrive/CNN_pred/"

import tensorflow as tf
from keras import layers, Model
import numpy as np
from matplotlib import pyplot as plt
import scipy.io

NumImages=1000   # number of data to be used
NumTotalImages= 10000  # number of data available
res=128
TestSplit= round(1*NumImages) # DATA USED FOR TRAINING

########################################### LOAD THE DATA #######################################################

dir = folder_path + "InputData/" + str(NumTotalImages)  + "_Images_L_50_Lw_50_radius_2.5_res_" + str(res) + ".mat"
mat = scipy.io.loadmat(dir)
x_data=mat['Img_array']
x_data = x_data[0:NumImages,:,:]

plt.imshow(x_data[0,:,:], interpolation=None) # reconstructs image from pixels, for testing
plt.show()

############################# SPLIT THE DATA ###########################################

(x_train, x_test) = (x_data[0:TestSplit,:,:], x_data[TestSplit:NumImages,:,:])

############################################ NORMALIZE DATA ##############################################################

x_train = x_train.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")

# VAE Implementation #

latent_dim = 128 # latent dimension
input_shape = (res, res, 1)  # image resolution

def sampling(args): # define a sampling layer
    z_mean, z_log_var = args # mean, log var for each image in latent space
    eps = tf.random.normal(shape=tf.shape(z_mean)) # random noise for each image in latent space, sampled from standard normal, THIS is the source of uncertainty
    return z_mean + tf.exp(0.5 * z_log_var) * eps # convert log var to std, then z = z_mean + sigma * eps returns a sample

# ---------- Encoder ----------
inputs = layers.Input(shape=input_shape)
x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same")(inputs)   # 64×64
x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)        # 32×32
x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)       # 16×16
x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)       # 8×8
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
z = layers.Lambda(sampling)([z_mean, z_log_var]) # wraps sampling() as a Keras layer, Output z is the latent vector fed to the decoder
encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder") # The encoder returns three things:z_mean → for KL lossz_log_var → for KL loss z → for reconstruction


# ---------- Decoder ----------
latent_inputs = layers.Input(shape=(latent_dim,)) #The decoder expects a latent vector z, During generation, we sample z ~ N(0, I)
x = layers.Dense(8 * 8 * 256, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 256))(x)
x = layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)  # 16×16
x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)  # 32×32
x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)   # 64×64
x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)   # 128×128
outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x) # Define the final output layer

decoder = Model(latent_inputs, outputs, name="decoder")

# ---------- VAE ----------
class VAE(Model):           # Custom training step --> Keras standard fit() cannot express KL divergence easily
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # Pipeline: image → encoder → μ, logσ² → sampling → z → decoder → image
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data) #Forward Pass
            reconstruction = self.decoder(z) # Forward Pass
            reconstr_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(data, reconstruction),axis=(1, 2))) #Reconstruction Loss - How close is the reconstructed image to the original? mse
            #reconstr_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction),axis=(1, 2))) #Reconstruction Loss - How close is the reconstructed image to the original?
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis=1)) # Forces latent distributions to be close to standard normalPrevents holes and discontinuities,Enables meaningful sampling
            total_loss = reconstr_loss + kl_loss # Tradeoff: Reconstruction → fidelity, KL → smooth latent space
        grads = tape.gradient(total_loss, self.trainable_weights) #Backpropagation
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) # Updates encoder + decoder weights

        return {"loss": total_loss, "recon": reconstr_loss, "kl": kl_loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0005))
vae.fit(x_train, epochs=200, batch_size=16) 


# Generate some images #

numGenImages = 15

z = tf.random.normal(shape=(numGenImages, latent_dim))
generated = decoder(z)*255
#generated = tf.where(generated < 200, 0, generated)

# grid dimensions
num_rows = 3
num_cols = 5

plt.figure(figsize=(num_cols * 3, num_rows * 3))
for i in range(numGenImages):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(generated[i, :, :, 0], interpolation=None) #
    plt.axis('off')
plt.suptitle('Generated Images', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
plt.show()