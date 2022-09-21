import tensorflow as tf


class MultiHeadAttention(tf.keras.models.Model):
    '''
    Multi-head Attention のモデルです。
    model = MultiheadAttention(
        hidden_dim=512,
        head_num=8,
        dropout_rate=0.1,
    )
    model(query, memory, mask, training=True)
    '''

    def __init__(self,hidden_dim:int,head_num:int,dropout_rate:float,*args,**kargs) -> None:
        super().__init__(*args,**kargs)
        '''
            コンストラクタ
            hidden_dim:隠れ層および出力の次元、head_numの倍数である必要がある
            head_num:ヘッドの数(multi head attentionにおけるattetionの数)
            dropout_rate:ドロップアプトする確率
        '''
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layers.Dense(hidden_dim,use_bias=False,name="q_dense_layer")
        self.k_dense_layer = tf.keras.layers.Dense(hidden_dim,use_bias=False,name="k_dense_layer")
        self.v_dense_layer = tf.keras.layers.Dense(hidden_dim,use_bias=False,name="v_dense_layer")

        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim,use_bias=False,name="output_dense_layer")
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self,input:tf.Tensor,memory:tf.Tensor,attention_mask:tf.Tensor,training:bool)->tf.Tensor:
        '''
            モデルの実行を行う
        '''

        q = self.q_dense_layer(input)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)

        q = self._split_head(q) # [batch_size,head_num,q_length,hidden_dim/head_num]
        k = self._split_head(k)
        v = self._split_head(v)

        depth = self.hidden_dim//self.head_num
        q *= depth ** -0.5 # for scaled dot procution

        # qとkの関連度を計算する
        logit = tf.matmul(q,k,transpose_b=True) # [batch_size,head_num,q_length,k_length]
        logit += tf.to_float(attention_mask) * input.dtype.min

        # softmaxで正規化する
        attention_weight = tf.nn.softmax(logit,name="attention_weight")
        attention_weight = self.attention_dropout_layer(attention_weight,training=training)

        # 重みに従ってvalueを引いていく
        attention_output = tf.matmul(attention_weight,v)
        attentiou_output = self._combine_head(attention_output)

        return self.output_dense_layer(attention_output)

    
    def _split_head(self,x:tf.Tensor)->tf.Tensor:
        '''
            入力のTensorをhidden_dimの次元をいくつかのheadに分割する
            入力 shape: [batch_size, length, hidden_dim] の時
            出力 shape: [batch_size, head_num, length, hidden_dim//head_num]
            となります。
        '''

        with tf.name_scope('split_head'):
            batch_size,length,hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x,[batch_size,length,self.head_num,self.hidden_dim//self.head_num])
            return tf.transpose(x,[0,2,1,3])
    
    def _combine_head(self,x:tf.Tensor)->tf.Tensor:
        '''
            入力のtensorの各headを結合する
            入力 shape: [batch_size, head_num, length, hidden_dim//head_num] の時
            出力 shape: [batch_size, length, hidden_dim]
        '''

        with tf.name_scope("combine_head"):
            batch_size,_,length,_ = tf.unstack(tf.shape(x))
            x = tf.transpose(x,[0,2,1,3])
            return tf.reshape(x,[batch_size,length,self.hidden_dim])

class SelfAttention(MultiHeadAttention):
    def call(
        self,
        input:tf.Tensor,
        attention_mask:tf.Tensor,
        training:bool
    )->tf.Tensor:
        return super().call(
            input=input,
            memory=input,
            attention_mask=attention_mask,
            training=training,
        )


class SimpleAttention(tf.keras.models.Model):
    def __init__(self,depth:int,*args,**kargs) -> None:
        super().__init__(*args,**kargs)
        #param depth: 隠れ層および出力の次元
        self.depth = depth

        self.q_dense_layer = tf.keras.layers.Dense(depth,use_bias=False,name="q_dense_layer")
        self.k_dense_layer = tf.keras.layers.Dense(depth,use_bias=False,name="k_dense_layer")
        self.v_dense_layer = tf.keras.layers.Dense(depth,use_bias=False,name="v_dense_layer")
        self.output_dense_layer = tf.keras.layers.Dense(depth,use_bias=False,name="output_dense_layer")
    
    def call(self,input:tf.Tensor,memory:tf.Tensor) -> tf.Tensor:
        #モデルの実行
        #input: queryのテンソル
        #memory: queryに情報を与えるmemoryのテンソル

        q = self.q_dense_layer(input) # [batch_size,q_length,depth]
        k = self.k_dense_layer(memory) # [batch_size,m_length,depth]
        v = self.v_dense_layer(memory)

        #qとkの内積を取ることでqとkの関連度を計算する
        #q *= depth ** -0.5
        logit = tf.matmul(q,k,transpose_b=True) # [batch_size,q_length,k_length]
        #logit += tf.to_float(attention_mask) * input.dtype.min

        # softmaxをとることで正規化する
        attention_weight = tf.nn.softmax(logit,name="attention_weight")

        # 重みに従ってvalueから情報を引いていく
        attention_output = tf.matmul(attention_weight,v) # [batch_size,q_length,depth]

        return self.output_dense_layer(attention_output)
