# https://ratsgo.github.io/natural%20language%20processing/2017/03/19/CNN/
import numpy as np






# train_data = pd.read_csv('../5-0. FastText/train_corpus.txt', delimiter='\t', names=['label', 'words'])
# test_data = pd.read_csv('../5-0. FastText/test_corpus.txt', delimiter='\t', names=['label', 'words'])

# def preprocess(data):
#     seq_len = 0
#     x_data = []
#     y_data = []
#     for idx, w in enumerate(data.words.values):
#         w = str(w).strip().split(' ')
#         row = [
#         for i in w:
#             si = i.split('_')
#             if len(si) == 2 and si[1][0] not in ['S','J','U']:
#                 row.append(i.replace('+','_'))
#         if len(row):
#             cnt = len(row)
#             if seq_len < cnt:
#                 seq_len = cnt
#             x_data.append(' '.join(row))
#             label = data.label.values[idx]
#             y_data.insert(0, [1,0] if label == '__label__P' else [0,1])
            
#     return seq_len, x_data, y_data

# seq_len, train_x, train_y = preprocess(train_data)
# _, test_x, test_y = preprocess(test_data)

# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(seq_len)
# train_x = list(vocab_processor.fit_transform(train_x))
# test_x = list(vocab_processor.fit_transform(test_x))
# vocab_dict = vocab_processor.vocabulary_._mapping
# word_index = sorted(vocab_dict.items(), key=lambda x: x[1])

# print(np.array(train_x).shape)
# print(np.array(train_y).shape)
# print(np.array(test_x).shape)
# print(np.array(test_y).shape)

# input_length = np.array(train_x).shape[1]
# output_length = 2
# words_count = len(word_index)
# level_count = 2
# filters = [3,4,5]
# filters_length = 32

# print(words_count)

# X = tf.placeholder(tf.int32, [None, input_length])
# Y = tf.placeholder(tf.float32, [None, output_length])

# with tf.name_scope("embedding"):    
#     W = tf.Variable(tf.random_uniform([words_count, level_count], -1., 1.))
#     print('W:',W)

#     embed = tf.nn.embedding_lookup(W, X)
#     print('embed.shape:',embed.shape)
    
#     embed_char = tf.expand_dims(embed, -1)
#     print('embed_char.shape:',embed_char.shape)

# pooled = []

# for _, f_no in enumerate(filters):
#     with tf.name_scope("conv-maxpool-%s" % f_no):
        
#         filter_shape = [f_no, level_count, 1, filters_length]
        
#         print(filter_shape)
#         print([1,input_length-f_no+1,1,1],'\n')
        
#         W = tf.Variable(
#             tf.truncated_normal(
#                 filter_shape, stddev=0.1
#             ), name='W'
#         )
#         b = tf.Variable(
#             tf.constant(0.1, shape=[filters_length]), name='b'
#         )
        
#         conv = tf.nn.conv2d(
#             embed_char,
#             W,
#             strides=[1,1,1,1], padding='VALID',
#             name='conv'
#         )
        
#         h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        
#         ksize=[1,input_length-f_no+1,1,1]
        
#         pool = tf.nn.max_pool(
#             h,
#             ksize=ksize,
#             strides=[1,1,1,1], padding='VALID',
#             name='pool'
#         )
#         pooled.append(pool)

# num_filters_total = filters_length * len(filters)
# num_filters_total

# pool = tf.concat(axis=3, values=pooled)
# pool

# pool_flat = tf.reshape(pool, [-1, num_filters_total])
# pool_flat

# with tf.name_scope("dropout"):
#     pool_drop = tf.nn.dropout(pool_flat, 0.5)
# pool_drop

# print([num_filters_total, output_length])

# with tf.name_scope("output"):
#     W = tf.get_variable(
#         'W', 
#         shape=[num_filters_total, output_length],
#         initializer=tf.contrib.layers.xavier_initializer()
#     )
#     b = tf.Variable(tf.constant(0.1, shape=[output_length]), name='b')

#     score = tf.nn.xw_plus_b(pool_drop, W, b, name='scores')
#     print(score)
#     predictions = tf.argmax(score, 1)
#     print(predictions)

# with tf.name_scope("loss"):
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=Y)
#     )
#     T = tf.train.AdamOptimizer(0.1).minimize(loss)

# with tf.name_scope("accuracy"):
#     correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for epoch in range(10):
#     _, c = sess.run([T, loss], feed_dict={X : train_x, Y: train_y})
#     # if epoch %10 == 0:
#     print(c)

# sess.close()