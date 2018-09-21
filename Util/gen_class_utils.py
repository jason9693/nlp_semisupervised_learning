import numpy as np
from Network.SSVAE.vae import VariationalAutoencoder
import params as par

def encode_predict_dataset( model_path,x, min_std = 0.0 ):

    VAE = VariationalAutoencoder( dim_x = par.max_length * par.embedding_dim, dim_z = 50 ) #Should be consistent with model being loaded
    with VAE.session:
        VAE.saver.restore( VAE.session, model_path )

        enc_x_ulab_mean, enc_x_ulab_var = VAE.encode( x )

        id_x_keep = np.std( enc_x_ulab_mean, axis = 0 ) > min_std

        enc_x_ulab_mean, enc_x_ulab_var = enc_x_ulab_mean[ :, id_x_keep ], enc_x_ulab_var[ :, id_x_keep ]

        data_ulab = np.hstack( [ enc_x_ulab_mean, enc_x_ulab_var ] )

    return data_ulab
