import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Generator, Mapping, Tuple, NamedTuple, Sequence

#================================
#==========  SNH model   ========
#================================
def Phi(features, out_dim, in_dim, width, no_layers, **kwargs):
  def sine_func(x):
      return jax.numpy.sin(x)
  layer_stack = []
  layer_stack.append(hk.Linear(width, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')))# first layer
  layer_stack.append(sine_func)
  for j in range(no_layers): # consecutive layers
    layer_stack.append(hk.Linear(width, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')))
    layer_stack.append(jax.nn.swish)
  layer_stack.append(hk.Linear(out_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')))# final layer

  return hk.Sequential(layer_stack)(features)


#================================
#=======  VQ VAE model   ========
#================================
class ResidualStack(hk.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = hk.Conv2D(output_channels=num_residual_hiddens,kernel_shape=(3, 3),stride=(1, 1),name="res3x3_%d" % i)
            conv1 = hk.Conv2D(output_channels=num_hiddens,kernel_shape=(1, 1),stride=(1, 1),name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))

    def __call__(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(jax.nn.relu(h))
            conv1_out = conv1(jax.nn.relu(conv3_out))
            h += conv1_out
        return jax.nn.relu(h)  # Resnet V1 style


class ConvEncoder(hk.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(ConvEncoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = hk.Conv2D(
          output_channels=self._num_hiddens // 2,
          kernel_shape=(4, 4),
          stride=(2, 2),
          name="enc_1")
        self._enc_2 = hk.Conv2D(
          output_channels=self._num_hiddens,
          kernel_shape=(4, 4),
          stride=(2, 2),
          name="enc_2")
        self._enc_3 = hk.Conv2D(
          output_channels=self._num_hiddens,
          kernel_shape=(3, 3),
          stride=(1, 1),
          name="enc_3")
        self._residual_stack = ResidualStack(
          self._num_hiddens,
          self._num_residual_layers,
          self._num_residual_hiddens)

    def __call__(self, x):
        h = jax.nn.relu(self._enc_1(x))
        h = jax.nn.relu(self._enc_2(h))
        h = jax.nn.relu(self._enc_3(h))
        return self._residual_stack(h)


class ConvDecoder(hk.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(ConvDecoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = hk.Conv2D(
          output_channels=self._num_hiddens,
          kernel_shape=(3, 3),
          stride=(1, 1),
          name="dec_1")
        self._residual_stack = ResidualStack(
          self._num_hiddens,
          self._num_residual_layers,
          self._num_residual_hiddens)
        self._dec_2 = hk.Conv2DTranspose(
          output_channels=self._num_hiddens // 2,
          # output_shape=None,
          kernel_shape=(4, 4),
          stride=(2, 2),
          name="dec_2")
        self._dec_3 = hk.Conv2DTranspose(
          output_channels=1,
          # output_shape=None,
          kernel_shape=(4, 4),
          stride=(2, 2),
          name="dec_3")
  
    def __call__(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = jax.nn.relu(self._dec_2(h))
        x_recon = self._dec_3(h)
        return x_recon
  

class VQVAEModel(hk.Module):
    def __init__(self, encoder, decoder, vqvae, pre_vq_conv1, data_variance, name=None):
        super(VQVAEModel, self).__init__(name=name)
        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

    # _train_h.shape (256, 256, 1, 256) each slice has shape (256, 256, 1)
    # inputs.shape (256, 256, 1)
    # z.shape (64, 64, 128)
    # vq_output[uantize].shape (64, 64, 128)
    # x_recon.shape (256, 256, 1)
    # inputs.shape (256, 256, 1)
    # z.shape (64, 64, 128)
    # vq_output[uantize].shape (64, 64, 128)
    # x_recon.shape (256, 256, 1)

    def __call__(self, inputs, is_training):
        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae(z, is_training=is_training)
        x_recon = self._decoder(vq_output['quantize'])
        recon_error = jnp.mean((x_recon - inputs) ** 2) / self._data_variance
        loss = recon_error + vq_output['loss']
        return {
          'z': z,
          'x_recon': x_recon,
          'loss': loss,
          'recon_error': recon_error,
          'vq_output': vq_output,
        }

#================================
#=======  GAUSS PRIOR model =====
#================================
# continuous latent embedding CNN Autoencoder

class D_Encoder(hk.Module):
  def __init__(self, hidden_size: int = 512, latent_size: int = 10, num_layers:int = 3):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size
    self._num_layers = num_layers

  def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = hk.Conv2D(output_channels=64,kernel_shape=(5, 5),stride=(4, 4),name="enc_2")(x)
    x = jax.nn.relu(x)
    x = hk.Conv2D(output_channels=128,kernel_shape=(3, 3),stride=(2, 2),name="enc_3")(x)
    x = jax.nn.relu(x)
    x = hk.Conv2D(output_channels=128,kernel_shape=(3, 3),stride=(1, 1),name="enc_4")(x)
    x = jax.nn.relu(x)
    x = hk.Flatten()(x)
    mean = hk.Linear(self._latent_size)(x)
    log_stddev = hk.Linear(self._latent_size)(x)
    stddev = jax.nn.softplus(log_stddev)
    return mean, stddev

class D_Decoder(hk.Module):
  def __init__(self, output_shape: Sequence[int], hidden_size:int = 512, latent_size:int = 10):
    super().__init__()
    self._output_shape = output_shape
    self._hidden_size = hidden_size
    self._latent_size = latent_size

  def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
    z = hk.Linear(5184*4)(z)
    z = z.reshape(-1, 72*2, 72*2, 1)
    z = jax.nn.relu(z)
    z = hk.Conv2DTranspose(output_channels=64,kernel_shape=(3, 3),stride=(2, 2),name="dec_3")(z)
    z = jax.nn.relu(z)
    z = hk.Conv2DTranspose(output_channels=32,kernel_shape=(5, 5),stride=(1, 1),name="dec_4")(z)
    z = jax.nn.relu(z)
    z = hk.Conv2DTranspose(output_channels=1,kernel_shape=(5, 5),stride=(2, 2),name="dec_5")(z)
    return z

class D_VAEOutput(NamedTuple):
    mean: jnp.ndarray
    stddev: jnp.ndarray
    logits: jnp.ndarray
    z: jnp.ndarray

class D_VariationalAutoEncoder(hk.Module):
  def __init__(self, output_shape: int, hidden_size: int = 512, latent_size: int = 10, num_layers: int = 3):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size
    self._output_shape = output_shape
    self._num_layers = num_layers

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    mean, stddev = D_Encoder(self._hidden_size, self._latent_size, self._num_layers)(x)
    z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    logits = D_Decoder(self._output_shape, self._hidden_size, self._latent_size)(z)
    return D_VAEOutput(mean, stddev, logits, z)