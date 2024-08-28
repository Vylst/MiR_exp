import tensorflow as tf

def conv_net_1d(activation = 'relu', init = tf.keras.initializers.Orthogonal(), laser_input_shape = 32, wp_input_shape=100, out_units = 6):
        
        input_laser = tf.keras.Input(shape=(laser_input_shape,1))
        input_wp = tf.keras.Input(shape=(wp_input_shape,))
        
        h1 = tf.keras.layers.Conv1D(filters= 32, kernel_size = 5, strides = 2, padding = 'valid', activation = activation, kernel_initializer = init)(input_laser)
        h2 = tf.keras.layers.Conv1D(filters= 64, kernel_size = 3, strides = 2, padding = 'valid', activation = activation, kernel_initializer = init)(h1)
        h3 = tf.keras.layers.Flatten()(h2)
        h4 = tf.keras.layers.Dense(units= 256, activation = activation, kernel_initializer = init)(h3)
        
        combined = tf.keras.layers.Concatenate()([h4, input_wp])
        
        h5 = tf.keras.layers.Dense(units = 128, activation = activation, kernel_initializer = init)(combined)
        output = tf.keras.layers.Dense(units = out_units, activation = 'linear', kernel_initializer = init)(h5)
        
        model = tf.keras.Model(inputs = [input_laser, input_wp], outputs = output)

        return model
    
    
m = conv_net_1d()
m.summary()


""" . The laser scan vector of the RawData
Representation is processed by layer one of the network,
while the nwp closest waypoints are concatenated with the
output of layer three and are then forwarded to layer four """