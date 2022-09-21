import tensorflow as tf

from attention import SelfAttention
from common_layer import FeedForwardNetwork, ResidualNormalizationWrapper
from embedding import AddPositinalEncoding, TokenEmbedding


class Encoder(tf.keras.models.Model):
    '''
        トークン列をベクトル列にエンコードする
    '''
    def __init__(
        self,
        vocab_size:int,
        hopping_num:int,
        head_num:int,
        hidden_dim:int,
        dropout_rate:float,
        max_length:int,
        *args,
        **kargs
    ) -> None:
        super().__init__(*args,**kargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropput_rate = dropout_rate

        self.token_embedding = TokenEmbedding(vocab_size,hidden_dim)
        self.add_position_embedding = AddPositinalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: list[list[tf.keras.models.Model]] = []

        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim,head_num,dropout_rate,name="self_attention")
            ffn_layer = FeedForwardNetwork(hidden_dim,head_num,dropout_rate,name="ffn")
            self.attention_block_list.append([
                ResidualNormalizationWrapper(attention_layer,dropout_rate,name="self_attention_warpper"),
                ResidualNormalizationWrapper(ffn_layer,dropout_rate,name="ffn_wrapper")
            ])
        
        self.output_normalization = tf.keras.layers.LayerNormalization()

    def call(
        self,
        input: tf.Tensor,
        self_attention_mask:tf.Tensor,
        training:bool
    )-> tf.Tensor:
        '''
            モデルを実行する
            input: shape = [batch_size,length]
            return: shape = [batch_size,length,hidden_dim]
        '''
        emmbedded_input = self.token_embedding(input)
        emmbedded_input = self.add_position_embedding(emmbedded_input)
        query = self.input_dropout_layer(emmbedded_input,training = training)

        for i,layers in enumerate(self.attention_block_list):
            attention_layer,ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = attention_layer(query,attention_mask=self_attention_mask,training=training)
                query = ffn_layer(query,training=training)
        
        return self.output_normalization(query)

            
class Decoder(tf.keras.models.Model):
    '''
        エンコードされたベクトル列からトークン列を生成するDecoder
    '''
    def __init__(
        self,
        vocab_size:int,
        hopping_num:int,
        head_num:int,
        hiddin_dim:int,
        dropout_rate:float,
        max_length:int,
        *args,
        **kargs
        ) -> None:
        super().__init__(*args,**kargs)
        
