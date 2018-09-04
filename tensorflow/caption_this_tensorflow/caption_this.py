import math
import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import skimage
import pickle as pkl
import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter
from caption_generator import CaptionGenerator
from caption_generator_2 import CaptionGenerator2

model_path = './models/tensorflow'
vgg_path = './data/vgg16-20160129.tfmodel'
image_path = './horses.jpg'
model_path_transfer = './models/tf_final'
features_path = './data/feats.npy'
annotations_path = './data/results_20130124.token'

def get_data(annotations_path, features_path):
    annotations = pd.read_table(annotations_path, sep='\t', header=None, names=['image', 'caption'])
    return np.load(features_path, 'r'), annotations['caption'].values

 # function from Andre Karpathy's NeuralTalk
 # Processing a word vocabulary which consists on words the occur >= word_count_threshold times
def build_word_vocab(sentence_iterator, word_count_threshold=30):
    print('preprocessing %d word vocab' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'
    wordtoix = {}
    wordtoix['#START#'] = 0
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return wordtoix, ixtoword, bias_init_vector.astype(np.float32)


### Parameters ###
dim_embed = 256 # Embedding layer dimension
dim_hidden = 256 # LSTM hidden layer dimension
dim_in = 4096 # Image feature dimension coming VGG-16
batch_size = 1 # Number of <image, caption> pairs to consider in a fwd pass
momentum = 0.9 # Momentum coefficient for Adam parameter update
n_epochs = 25 # Number of training iterations (each with a fwd-bwd pass of the entire dataset)
learning_rate = 0.001


def train(learning_rate=0.001, continue_training=False, transfer=True):

    tf.reset_default_graph()

    feats, captions = get_data(annotations_path, features_path)
    wordtoix, ixtoword, init_b = build_word_vocab(captions)

    np.save('data/ixtoword', ixtoword)

    index = (np.arange(len(feats)).astype(int))
    np.random.shuffle(index)


    sess = tf.InteractiveSession()
    n_words = len(wordtoix)
    maxlen = np.max( [x for x in map(lambda x: len(x.split(' ')), captions) ] )
    caption_generator = CaptionGenerator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words, init_b)

    loss, image, sentence, mask = caption_generator.build_model()

    saver = tf.train.Saver(max_to_keep=100)
    global_step=tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                       int(len(index)/batch_size), 0.95)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.global_variables_initializer().run()

    if continue_training:
        if not transfer:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            saver.restore(sess, tf.train.latest_checkpoint(model_path_transfer))
    losses = []
    for epoch in range(n_epochs):
        for start, end in zip( range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):

            current_feats = feats[index[start:end]]
            current_captions = captions[index[start:end]]
            current_caption_ind = [x for x in map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)]

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] )

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array([x for x in map(lambda x: (x != 0).sum()+2, current_caption_matrix )])

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                image: current_feats.astype(np.float32),
                sentence : current_caption_matrix.astype(np.int32),
                mask : current_mask_matrix.astype(np.float32)
                })

            print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs), "\t Iter {}/{}".format(start,len(feats)))
        print("Saving the model from epoch: ", epoch)
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

def test_with_feature(sess, image, generated_words, ixtoword, idx=0): # Naive greedy search
    feats, captions = get_data(annotations_path, features_path)
    feat = np.array([feats[idx]])

    saver = tf.train.Saver()
    sanity_check= False
    # sanity_check=True
    if not sanity_check:
        saved_path = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, saved_path)
    else:
        tf.global_variables_initializer().run()

    generated_word_index= sess.run(generated_words, feed_dict={image:feat})
    generated_word_index = np.hstack(generated_word_index)

    generated_sentence = [ixtoword[x] for x in generated_word_index]
    print(generated_sentence)

def test_model_with_feature():
    image = None
    generated_words = None
    if not os.path.exists('data/ixtoword.npy'):
        print ('You must run 1. O\'reilly Training.ipynb first.')
    else:
        ixtoword = np.load('data/ixtoword.npy').tolist()
        n_words = len(ixtoword)
        maxlen = 15

        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        caption_generator = CaptionGenerator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words)
        image, generated_words = caption_generator.build_generator(maxlen=maxlen)
    test_with_feature(sess, image, generated_words, ixtoword, 55)


def test_with_image(graph, sess, image, images, generated_words, ixtoword, test_image_path=0): # Naive greedy search
    feat = read_image(test_image_path)
    fc7 = sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={images:feat})
    saver = tf.train.Saver()
    sanity_check=False
    # sanity_check=True
    if not sanity_check:
        saved_path = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, saved_path)
    else:
        tf.global_variables_initializer().run()

    generated_word_index= sess.run(generated_words, feed_dict={image:fc7})
    generated_word_index = np.hstack(generated_word_index)
    generated_words = [ixtoword[x] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.') + 1

    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    print(generated_sentence)


def test_model_with_image():
    if not os.path.exists('data/ixtoword.npy'):
        print ('You must run 1. O\'reilly Training.ipynb first.')
    else:
        tf.reset_default_graph()
        with open(vgg_path,'rb') as f:
            fileContent = f.read()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fileContent)

        images = tf.placeholder("float32", [1, 224, 224, 3])
        tf.import_graph_def(graph_def, input_map={"images":images})

        ixtoword = np.load('data/ixtoword.npy').tolist()
        n_words = len(ixtoword)
        maxlen=15
        graph = tf.get_default_graph()
        sess = tf.InteractiveSession(graph=graph)
        caption_generator = CaptionGenerator2(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words)
        graph = tf.get_default_graph()

    image, generated_words = caption_generator.build_generator(maxlen=maxlen)
    test_with_image(graph, sess, image, images, generated_words, ixtoword, image_path)



def crop_image(x, target_height=227, target_width=227, as_float=True):
    image = cv2.imread(x)
    if as_float:
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))


def read_image(path):
     img = crop_image(path, target_height=224, target_width=224)
     if img.shape[2] == 4:
         img = img[:,:,:3]
     img = img[None, ...]
     return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', default="test_feat")
    parser.add_argument('--continue_training', default=False)
    parser.add_argument('--transfer', default=False)
    args = parser.parse_args()

    if args.op == "train" and args.continue_training == False and args.transfer == False:
        print('Training from scratch ...')
        train(.001, False, False) # Train from scratch
    elif args.op == "train" and args.continue_training == True and args.transfer == True:
        print('Continuing training from pretrained weights @epoch500')
        train(.001, True, True)   # Continue training from pretrained weights @epoch500
    elif args.op == "train" and args.continue_training == True and args.transfer == False:
        print('Training from previously saved weights')
        train(.001, True, False)  # Train from previously saved weights
    elif args.op == "test_img":
        test_model_with_image()
    elif args.op == "test_feat":
        test_model_with_feature()
