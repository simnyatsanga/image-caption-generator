import os
import glob
import string
import argparse
from os import listdir
from pickle import dump
from pickle import load
from PIL import Image
from numpy import array
from numpy import argmax
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import preprocess_input
from keras.utils import plot_model
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint


class ImageCaptionGenerator(object):
	def __init__(self):
		self.prepare_image_data()
		self.prepare_text_data()
		self.training_features, self.training_desc = self.prepare_training_data()
		self.test_data_features, self.test_data_descriptions = self.load_test_data()
		self.tokenizer = self.prepare_tokenizer(self.training_desc)
		self.vocab_size = self.summarize_vocab(self.tokenizer)
		self.max_length = self.max_length_desc(self.training_desc)
		self.pretrained_model = self.load_pretrained_model('model_checkpoints/model_99.h5')


	def testing_params(self):
		return self.pretrained_model, self.tokenizer, self.max_length

	def extract_features(self, source):
		# load the model
		model = VGG16()
		# re-structure the model
		model.layers.pop()
		model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
		# summarize
		# print(model.summary())
		# extract features from each photo
		if os.path.isdir(source):
			features = dict()
			for name in listdir(directory):
				# load an image from file
				filename = directory + '/' + name
				feature = self.extract_feature(filename, model)
				# get image id
				image_id = name.split('.')[0]
				# store feature
				features[image_id] = feature
				print('>%s' % name)
			return features
		elif os.path.isfile(source):
			return self.extract_feature(source, model)
		else:
			raise Exception("Source for images needs to be a file or directory.")


	# extract a feature for a single photo
	def extract_feature(self, filename, model):
		# load the photo
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		return feature

	# load doc into memory
	def load_doc(self, filename):
		# open the file as read only
		file = open(filename, 'r')
		# read all text
		text = file.read()
		# close the file
		file.close()
		return text

	# extract descriptions for images
	def load_descriptions(self, doc):
		mapping = dict()
		# process lines
		for line in doc.split('\n'):
			# split line by white space
			tokens = line.split()
			if len(line) < 2:
				continue
			# take the first token as the image id, the rest as the description
			image_id, image_desc = tokens[0], tokens[1:]
			# remove filename from image id
			image_id = image_id.split('.')[0]
			# convert description tokens back to string
			image_desc = ' '.join(image_desc)
			# create the list if needed
			if image_id not in mapping:
				mapping[image_id] = list()
			# store description
			mapping[image_id].append(image_desc)
		return mapping

	def clean_descriptions(self, descriptions):
		# prepare translation table for removing punctuation
		table = str.maketrans('', '', string.punctuation)
		for key, desc_list in descriptions.items():
			for i in range(len(desc_list)):
				desc = desc_list[i]
				# tokenize
				desc = desc.split()
				# convert to lower case
				desc = [word.lower() for word in desc]
				# remove punctuation from each token
				desc = [w.translate(table) for w in desc]
				# remove hanging 's' and 'a'
				desc = [word for word in desc if len(word)>1]
				# remove tokens with numbers in them
				desc = [word for word in desc if word.isalpha()]
				# store as string
				desc_list[i] =  ' '.join(desc)

	# convert the loaded descriptions into a vocabulary of words
	def to_vocabulary(self, descriptions):
		# build a list of all description strings
		vocab = set()
		for key in descriptions.keys():
			[vocab.update(d.split()) for d in descriptions[key]]
		return vocab

	# save descriptions to file, one per line
	def save_descriptions(self, descriptions, filename):
		lines = list()
		for key, desc_list in descriptions.items():
			for desc in desc_list:
				lines.append(key + ' ' + desc)
		data = '\n'.join(lines)
		file = open(filename, 'w')
		file.write(data)
		file.close()

	# load a pre-defined list of photo identifiers
	def load_set(self, filename):
		doc = self.load_doc(filename)
		dataset = list()
		# process line by line
		for line in doc.split('\n'):
			# skip empty lines
			if len(line) < 1:
				continue
			# get the image identifier
			identifier = line.split('.')[0]
			dataset.append(identifier)
		return set(dataset)

	# load clean descriptions into memory
	def load_clean_descriptions(self, filename, dataset):
		# load document
		doc = self.load_doc(filename)
		descriptions = dict()
		for line in doc.split('\n'):
			# split line by white space
			tokens = line.split()
			# split id from description
			image_id, image_desc = tokens[0], tokens[1:]
			# skip images not in the set
			if image_id in dataset:
				# create list
				if image_id not in descriptions:
					descriptions[image_id] = list()
				# wrap description in tokens
				desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
				# store
				descriptions[image_id].append(desc)
		return descriptions

	# load photo features
	def load_photo_features(self, filename, dataset):
		# load all features
		all_features = load(open(filename, 'rb'))
		# filter features
		features = {k: all_features[k] for k in dataset}
		return features

	# convert a dictionary of clean descriptions to a list of descriptions
	def to_lines(self, descriptions):
		all_desc = list()
		for key in descriptions.keys():
			[all_desc.append(d) for d in descriptions[key]]
		return all_desc

	# fit a tokenizer given caption descriptions
	# every string extracted from the list of descriptions is encoded individually
	def create_tokenizer(self, descriptions):
		lines = to_lines(descriptions)
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(lines)
		return tokenizer

	# create sequences of images, input sequences and output words for an image
	# NOTE: This is memory ineffienct. Rather progressively load batch of images
	# def create_sequences(tokenizer, max_length, descriptions, photos):
	# 	X1, X2, y = list(), list(), list()
	# 	# walk through each image identifier
	# 	import ipdb; ipdb.set_trace()
	# 	for key, desc_list in descriptions.items():
	# 		# walk through each description for the image
	# 		for desc in desc_list:
	# 			# encode the sequence
	# 			seq = tokenizer.texts_to_sequences([desc])[0]
	# 			# split one sequence into multiple X,y pairs
	# 			for i in range(1, len(seq)):
	# 				# split into input and output pair
	# 				in_seq, out_seq = seq[:i], seq[i]
	# 				# pad input sequence
	# 				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
	# 				# encode output sequence
	# 				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
	# 				# store
	# 				X1.append(photos[key][0])
	# 				X2.append(in_seq)
	# 				y.append(out_seq)
	# 	return array(X1), array(X2), array(y)

	def create_sequences(self, tokenizer, max_length, desc_list, photo):
		X1, X2, y = list(), list(), list()
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photo)
				X2.append(in_seq)
				y.append(out_seq)
		return array(X1), array(X2), array(y)


	# data generator, intended to be used in a call to model.fit_generator()
	def data_generator(self, descriptions, photos, tokenizer, max_length):
		# loop for ever over images
		while 1:
			for key, desc_list in descriptions.items():
				# retrieve the photo feature
				photo = photos[key][0]
				in_img, in_seq, out_word = self.create_sequences(tokenizer, max_length, desc_list, photo)
				yield [[in_img, in_seq], out_word]


	# calculate the length of the description with the most words
	def get_max_length(self, descriptions):
		lines = self.to_lines(descriptions)
		return max(len(d.split()) for d in lines)

	#define the captioning model
	def define_model(self, vocab_size, max_length):
		# feature extractor model
		inputs1 = Input(shape=(4096,))
		fe1 = Dropout(0.5)(inputs1)
		fe2 = Dense(256, activation='relu')(fe1)
		# sequence model
		inputs2 = Input(shape=(max_length,))
		se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
		se2 = Dropout(0.5)(se1)
		se3 = LSTM(256)(se2)
		# decoder model
		decoder1 = add([fe2, se3])
		decoder2 = Dense(256, activation='relu')(decoder1)
		outputs = Dense(vocab_size, activation='softmax')(decoder2)
		# tie it together [image, seq] [word]
		model = Model(inputs=[inputs1, inputs2], outputs=outputs)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		# summarize model
		print(model.summary())
		plot_model(model, to_file='model.png', show_shapes=True)
		return model

	#map an integet to a word
	def word_for_id(self, integer, tokenizer):
		for word, index in tokenizer.word_index.items():
			if index == integer:
				return word
		return None

	# generate a description for an image
	def generate_desc(self, model, tokenizer, photo, max_length):
		# seed the generation process
		in_text = 'startseq'
		# iterate over the whole length of the sequence
		for i in range(max_length):
			# integer encode input sequence
			sequence = tokenizer.texts_to_sequences([in_text])[0]
			# pad input
			sequence = pad_sequences([sequence], maxlen=max_length)
			# predict next word
			yhat = model.predict([photo,sequence], verbose=0)
			# convert probability to integer
			yhat = argmax(yhat)
			# map integer to word
			word = self.word_for_id(yhat, tokenizer)
			# stop if we cannot map the word
			if word is None:
				break
			# append as input for generating the next word
			in_text += ' ' + word
			# stop if we predict the end of the sequence
			if word == 'endseq':
				break
		return in_text

	# evaluate the skill of the model
	def evaluate_model(self, model, descriptions, photos, tokenizer, max_length):
		actual, predicted = list(), list()
		# step over the whole set
		for key, desc_list in descriptions.items():
			# generate description
			yhat = self.generate_desc(model, tokenizer, photos[key], max_length)
			# store actual and predicted
			references = [d.split() for d in desc_list]
			actual.append(references)
			predicted.append(yhat.split())
		# calculate BLEU score
		print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
		print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
		print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
		print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


	def prepare_image_data(self):
		# Prepare the Image Data
		# extract features from all images
		directory = 'Flickr8k_image_data'
		if os.path.exists('features.pkl'):
			print('Features already extracted into \'features.plk\' file.')
		else:
			features = self.extract_features(directory)
			print('Extracted Features: %d' % len(features))
			# save to file
			dump(features, open('features.pkl', 'wb'))


	def prepare_text_data(self):
		# Prepare the Text Data
		# load file containing all text descriptions of the images
		filename = 'Flickr8k_text_data/Flickr8k.token.txt'
		# load descriptions
		doc = self.load_doc(filename)
		# parse descriptions
		descriptions = self.load_descriptions(doc)
		print('Loaded Descriptions: %d ' % len(descriptions))
		# clean descriptions
		self.clean_descriptions(descriptions)

		# summarize vocabulary
		vocabulary = self.to_vocabulary(descriptions)
		print('Vocabulary Size: %d' % len(vocabulary))

		# save descriptions
		self.save_descriptions(descriptions, 'descriptions.txt')


	def prepare_training_data(self):
		# load training dataset (6K)
		filename = 'Flickr8k_text_data/Flickr_8k.trainImages.txt'
		training_images_data = self.load_set(filename)
		print('Training Images Dataset: %d' % len(training_images_data))
		# descriptions
		training_data_descriptions = self.load_clean_descriptions('descriptions.txt', training_images_data)
		print('Descriptions for Training Images Dataset: %d' % len(training_data_descriptions))
		# photo features
		train_features = self.load_photo_features('features.pkl', training_images_data)
		print('Extracted Training Image Features: %d' % len(train_features))
		return train_features, training_data_descriptions


	def prepare_tokenizer(self, training_data_descriptions):
		if os.path.exists('tokenizer.pkl'):
			print('Tokenizer already created and saved into \'tokenizer.pkl\'')
			print('Loading tokenizer ...')
			tokenizer = load(open('tokenizer.pkl', 'rb'))
		else:
			# prepare tokenizer
			tokenizer = self.create_tokenizer(training_data_descriptions)
			# save the tokenizer
			dump(tokenizer, open('tokenizer.pkl', 'wb'))
		return tokenizer


	def summarize_vocab(self, tokenizer):
		vocab_size = len(tokenizer.word_index) + 1
		print('Vocabulary Size: %d' % vocab_size)
		return vocab_size

	def max_length_desc(self, training_data_descriptions):
		# determine the maximum sequence length
		max_length = self.get_max_length(training_data_descriptions)
		print('Maximum Description Length (in words): %d' % max_length)
		return max_length

	def load_pretrained_model(self, filename):
		print('Loading latest model ...')
		return load_model(filename)


	def load_test_data(self):
		# load test set
		filename = 'Flickr8k_text_data/Flickr_8k.devImages.txt'
		test_images_data = self.load_set(filename)
		print('Test Images Dataset: %d' % len(test_images_data))
		# descriptions
		test_data_descriptions = self.load_clean_descriptions('descriptions.txt', test_images_data)
		print('Descriptions for Test Images Dataset: %d' % len(test_data_descriptions))
		# photo features
		test_features = self.load_photo_features('features.pkl', test_data_descriptions)
		print('Extracted Test Image Features: %d' % len(test_features))
		return test_features, test_data_descriptions


	def clean_old_model_checkpoints(self):
		models = glob.glob('*.h5')
		for model in models:
			try:
				os.remove(model)
			except:
				print('Failed to remove %d' % model)


	def train(self, vocab_size, training_data_descriptions, train_features, tokenizer, max_length):
		self.clean_old_model_checkpoints()
		# define the model
		model = self.define_model(vocab_size, max_length)

		# train the model, run epochs manually and save after each epoch
		epochs = 100
		steps = len(training_data_descriptions)
		for i in range(epochs):
			print('Running epoch %d' %i)
			# create the data generator
			generator = self.data_generator(training_data_descriptions, train_features, tokenizer, max_length)
			# fit for one epoch
			model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
			# save model
			model.save('model_' + str(i) + '.h5')


	def evaluate(self, model, test_data_descriptions, test_features, tokenizer, max_length):
		self.evaluate_model(model, test_data_descriptions, test_features, tokenizer, max_length)


	def test(self, model, tokenizer, max_length, photo_file):
		# pre-define the max sequence length (from training)
		max_length = 34
		# load and prepare the photograph
		photo = self.extract_features(photo_file)
		# generate description
		description = self.generate_desc(self.pretrained_model, self.tokenizer, photo, self.max_length)
		return description


# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--op', default='evaluate')
#
# 	# All the preparation
# 	prepare_image_data()
# 	prepare_text_data()
# 	training_features, training_desc = prepare_training_data()
# 	test_data_features, test_data_descriptions = load_test_data()
# 	tokenizer = prepare_tokenizer(training_desc)
# 	vocab_size = summarize_vocab(tokenizer)
# 	max_length = max_length_desc(training_desc)
# 	pretrained_model = load_pretrained_model('model_99.h5')
#
#
# 	args = parser.parse_args()
#
# 	if args.op == 'train':
# 		print('ALL SET FOR TRAINING ...')
# 		train(vocab_size, training_desc, training_features, tokenizer, max_length)
# 	elif args.op == 'evaluate':
# 		print('ALL SET FOR EVALUATING ...')
# 		evaluate(pretrained_model, test_data_descriptions, test_data_features, tokenizer, max_length)
# 	elif args.op == 'test':
# 		print('ALL SET FOR TESTING ...')
# 		test(pretrained_model, tokenizer, max_length)
# 	else:
# 		raise Exception('Choose valid operation: \'train\', \'evaluate\' or \'test\'')
