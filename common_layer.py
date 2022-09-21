import tensorflow as tf


class FeedForwardNetwork(tf.keras.models.Model):
    '''
        Transformer用のposition-wise feedforward network
    '''

    def __init__(self,hidde_dim:int,dropout_rate:float,*args,**kargs) -> None:
        super().__init__(*args,**kargs)
        self.hidden_dim = hidde_dim
        self.dropout_rate = dropout_rate

        self.filter_dense_layer = tf.keras.layers.Dense(hidde_dim*4,use_bias=True,activation=tf.nn.relu,name="filter_dense_layer")
        self.output_dense_layer = tf.keras.layers.Dense(hidde_dim,use_bias=True,name="output_layer")
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self,x:tf.Tensor,training:bool)->tf.Tensor:
        '''
            feed forward networkを適用する
            input: shape = [batch_size,length,hidden_dim]
            return :shape = [batch_size,length,hidden_dim]
        '''

        tensor = self.filter_dense_layer(input)
        tensor = self.output_dense_layer(tensor,training=training)
        return self.dropout_layer(tensor)

class ResidualNormalizationWrapper(tf.keras.layers.Layer):
    def __init__(self,layer:tf.kers.layers.Layer,dropout_rate:float,*args,**kargs) -> None:
        super().__init__(*args,**kargs)
        self.layer = layer
        self.layer_normalization = tf.keras.layers.LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self,input:tf.Tensor,training:bool, *args, **kargs) -> tf.Tensor:
        tensor = self.layer_normalization(input)
        tensor = self.layer(tensor,training=training,*args,**kargs)
        tensor = self.dropout_layer(tensor,training=training)

        return input + tensor
