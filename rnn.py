# coding: utf-8
from rnnmath import *
from utils import *
import numpy as np
from sys import stdout
import time

class RNN(object):
	'''
	This class implements Recurrent Neural Networks.
	'''

	def __init__(self, vocab_size, hidden_dims):
		'''
		initialize the RNN with random weight matrices.

		vocab_size : size of vocabulary that is being used
		hidden_dims	: number of hidden units
		'''
		self.vocab_size = vocab_size
		self.hidden_dims = hidden_dims

		# matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
		self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
		self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
		self.W = np.random.randn(self.vocab_size, self.hidden_dims)*np.sqrt(0.1)

		# matrices to accumulate weight updates
		self.deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
		self.deltaV = np.zeros((self.hidden_dims, self.vocab_size))
		self.deltaW = np.zeros((self.vocab_size, self.hidden_dims))

	def apply_deltas(self, learning_rate):
		'''
		update the RNN's weight matrices with corrections accumulated over some training instances
		'''
		# apply updates to U, V, W
		self.U += learning_rate*self.deltaU
		self.W += learning_rate*self.deltaW
		self.V += learning_rate*self.deltaV

		# reset matrices
		self.deltaU.fill(0.0)
		self.deltaV.fill(0.0)
		self.deltaW.fill(0.0)

	def predict(self, x):
		'''
		predict an output sequence y for a given input sequence x

		x : list of words, as indices, e.g.: [0, 4, 2]

		returns	y,s
		y : matrix of probability vectors for each input word
		s : matrix of hidden layers for each input word

		'''
		s = np.zeros((len(x)+1, self.hidden_dims))
		y = np.zeros((len(x), self.vocab_size))
		one_hot_x = []
		for w in x:
			one_hot_x.append(make_onehot(w,self.vocab_size))
		x = np.array(one_hot_x)

		for t in range(len(x)):
			s_prev = s[t-1,:] # in order to keep up with the code provided wherver we deal with s we suppose that t is the previous state and t+1 is the current one
			net_in = np.dot(self.V,x[t,:].T) + np.dot(self.U,s_prev.T)
			s_cur = sigmoid(net_in)
			net_out = np.dot(self.W,s_cur)
			y_cur = softmax(net_out)
			s[t,:] = s_cur #as stated above we suppose that t+1 is the current state and t is the previous one
			y[t,:] = y_cur


		return y,s

	def acc_deltas(self, x, d, y, s):
		i=1
		lr=1
		'''
		accumulate updates for V, W, U
		standard back propagation

		We use deltaV, deltaW, deltaU to accumulate updates over time
		'''
		for t in reversed(range(len(x))):
				i=2
				dout = make_onehot(d[t],len(y[t]))-y[t]
				self.deltaW = self.deltaW + lr*(np.outer(dout,s[t]))
				f_t = s[t]*(np.ones(s[t].shape)-s[t])
				din = (self.W.T).dot(dout)*f_t
				self.deltaV += lr*(np.outer(din,make_onehot(x[t],self.vocab_size)))
				self.deltaU += lr*(np.outer(din,s[t-1]))


	def acc_deltas_bptt(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)

		We use deltaV, deltaW, deltaU to accumulate updates over time
		'''

		lr=1
		for t in np.arange(len(x))[::-1]:
				i=2
				dout = make_onehot(d[t],len(y[t]))-y[t]
				self.deltaW = self.deltaW + lr*(np.outer(dout,s[t]))
				f_t = s[t]*(1-s[t])
				din = (self.W.T).dot(dout)*f_t
				pq=0
				pq = max(0,t-steps)
				
				for step in np.arange(pq,t+1)[::-1]:
					self.deltaV += lr * (np.outer( din,make_onehot(x[step], self.vocab_size)))
					self.deltaU += lr * (np.outer(din, s[step-1]))

					f_t = s[step-1] * (1 - s[step-1])
					din = (self.U.T).dot(din) * f_t

	def compute_loss(self, x, d):
		'''
		compute the loss between predictions y for x, and desired output d.

		first predicts the output for x using the RNN, then computes the loss w.r.t. d

		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]

		return loss		the combined loss for all words
		'''

		loss = 0.
		y,s= self.predict(x)
		
		for i in xrange(len(d)):

			pred = make_onehot(d[i],self.vocab_size)
			loss-=pred.dot(np.log(y[i]))

		return loss

	def compute_mean_loss(self, X, D):
		'''
		compute the mean loss between predictions for corpus X and desired outputs in corpus D.

		X : corpus of sentences x1, x2, x3, [...], each a list of words as indices.
		D : corpus of desired outputs d1, d2, d3 [...], each a list of words as indices.

		return mean_loss : average loss over all words in D
		'''

		mean_loss = 0.
		N = 0.
		for i in xrange(len(D)):
			N +=len(X[i])
			mean_loss+=self.compute_loss(X[i],D[i])

		mean_loss = mean_loss/N

		return mean_loss

	def generate_sequence(self, start, end, maxLength):
		'''
		generate a new sequence, using the RNN
		'''
		sequence = [start]
		loss = 0.
		x = [start]

		x_next = -1 #initializing the variable that holds the next word generated

		for i in range(maxLength):
			if(x_next == end):
				break
			y,_ = self.predict(x)
			resamplings = 0
			if unknown != -1:
				x_next = multinomial_sample(y[-1])
				while (x_next == unknown or x_next == start) and resamplings < 10: #resampling every time that the unknown word or the start token is returned for a fixed number of iterations to avoid infinite loop
					x_next = multinomial_sample(y[-1])
					resamplings += 1
			else:
				x_next = multinomial_sample(y[-1])
				while x_next == start and resamplings < 10:
					x_next = multinomial_sample(y[-1])
					resamplings += 1
			loss += -np.log(y[-1][x_next])
			x.append(x_next)

		sequence = x

		seq_len = len(sequence)-1 #not taking into account the start symbol
		if sequence[-1] == end: # not tsking into account the end symbol if it is generated
			seq_len -= 1

		if seq_len > 0:
			return sequence, loss/seq_len
		else:
			return sequence,0

	def train(self, X, D, X_dev, D_dev, epochs=10, learning_rate=0.5, anneal=5, back_steps=0, batch_size=100, min_change=0.0001, log=True):
		'''
		train the RNN on some training set X, D while optimizing the loss on a dev set X_dev, D_dev
		'''
		if log:
			stdout.write("\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
			stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
			stdout.write("\nVocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
			stdout.write("\nSteps for back propagation: {0}".format(back_steps))
			stdout.write("\nInitial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
			stdout.write("\n\ncalculating initial mean loss on dev set")
			stdout.flush()

		t_start = time.time()

		loss_sum = sum([len(d) for d in D_dev])
		initial_loss = sum([self.compute_loss(X_dev[i], D_dev[i]) for i in range(len(X_dev))])/loss_sum

		if log or not log:
			stdout.write(": {0}\n".format(initial_loss))
			stdout.flush()

		prev_loss = initial_loss
		loss_watch_count = -1
		min_change_count = -1

		a0 = learning_rate

		best_loss = initial_loss
		bestU, bestV, bestW = self.U, self.V, self.W
		best_epoch = 0

		for epoch in range(epochs):
			if anneal > 0:
				learning_rate = a0/((epoch+0.0+anneal)/anneal)
			else:
				learning_rate = a0

			if log:
				stdout.write("\nepoch %d, learning rate %.04f" % (epoch+1, learning_rate))
				stdout.flush()

			t0 = time.time()
			count = 0

			# use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
			permutation = np.random.permutation(range(len(X)))
			if log:
				stdout.write("\tinstance 1")
			for i in range(len(X)):
				c = i+1
				if log:
					stdout.write("\b"*len(str(i)))
					stdout.write("{0}".format(c))
					stdout.flush()
				p = permutation[i]
				x_p = X[p]
				d_p = D[p]

				y_p, s_p = self.predict(x_p)
				if back_steps == 0:
					self.acc_deltas(x_p, d_p, y_p, s_p)
				else:
					self.acc_deltas_bptt(x_p, d_p, y_p, s_p, back_steps)

				if i % batch_size == 0:
					self.deltaU /= batch_size
					self.deltaV /= batch_size
					self.deltaW /= batch_size
					self.apply_deltas(learning_rate)

			if len(X) % batch_size > 0:
				mod = len(X) % batch_size
				self.deltaU /= mod
				self.deltaV /= mod
				self.deltaW /= mod
				self.apply_deltas(learning_rate)

			loss = sum([self.compute_loss(X_dev[i], D_dev[i]) for i in range(len(X_dev))])/loss_sum

			if log:
				stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
				stdout.write("\tnew loss: {0}".format(loss))
				stdout.flush()

			if loss < best_loss:
				best_loss = loss
				bestU, bestV, bestW = self.U.copy(), self.V.copy(), self.W.copy()
				best_epoch = epoch

			# make sure we change the RNN enough
			if abs(prev_loss - loss) < min_change:
				min_change_count += 1
			else:
				min_change_count = 0
			if min_change_count > 2:
				print("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
				break

			prev_loss = loss

		t = time.time() - t_start

		if min_change_count <= 2:
			print("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))
		print("best observed loss was {0}, at epoch {1}".format(best_loss, (best_epoch+1)))

		print("setting U, V, W to matrices from best epoch")
		self.U, self.V, self.W = bestU, bestV, bestW

		return best_loss

if __name__ == "__main__":
	import sys
	from utils import *
	mode = sys.argv[1].lower()

	if mode == "estimate":
		'''
		starter code for parameter estimation.
		'''

		data_folder = sys.argv[2]
		vocab_size = 2000
		train_size = 1000
		dev_size = 1000

		# get the data set vocabulary
		vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		word_to_num = invert_dict(num_to_word)

		# calculate loss vocabulary words due to vocab_size
		fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
		print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))

		docs = load_dataset(data_folder + '/ptb-train.txt')
		S_train = docs_to_indices(docs, word_to_num)
		X_train, D_train = seqs_to_lmXY(S_train)

		# Load the dev set (for tuning hyperparameters)
		docs = load_dataset(data_folder + '/ptb-dev.txt')
		S_dev = docs_to_indices(docs, word_to_num)
		X_dev, D_dev = seqs_to_lmXY(S_dev)

		X = X_train[:train_size]
		D = D_train[:train_size]
		X_dev = X_dev[:dev_size]
		D_dev = D_dev[:dev_size]
		#25,2,0.5
		# q = best unigram frequency from omitted vocab
		# this is the best expected loss out of that set
		q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

		hid_un = [25,50]
		look_back=[0,2,5]
		lr = [0.05,0.1,0.5]
		best_loss = 0.
		best_lr = 0
		best_hid = 0
		best_lb = 0
		for i in hid_un:
			for j in look_back:
				for k in lr:
					rnn = RNN(vocab_size,i)
					loss = rnn.train(X, D, X_dev, D_dev, epochs=10, learning_rate=k, anneal=5, back_steps=j,
						  batch_size=100, min_change=0.0001, log=True)
					if loss>tot_loss:
						best_loss=loss
						best_lr = k
						best_hid = i
						best_lb = j


		best_loss = 100
		best_hdim = -1
		best_lookback = -1
		best_lr = -1


		print("\nestimation finished.\n\tbest loss {0}:\tbest hidden/lookback/learn rate: {1}/{2}/{3}".format(best_loss, best_params[0], best_params[1], best_params[2]))

	if mode == "train":
		'''
		starter code for parameter estimation.
		change this to different values, or use it to get you started with your own testing class
		'''

		data_folder = sys.argv[2]
		train_size = 25000
		dev_size = 1000
		vocab_size = 2000

		hdim = int(sys.argv[3])
		lookback = int(sys.argv[4])
		lr = float(sys.argv[5])

		# get the data set vocabulary
		vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		word_to_num = invert_dict(num_to_word)

		# calculate loss vocabulary words due to vocab_size
		fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
		print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))

		docs = load_dataset(data_folder + '/ptb-train.txt')
		S_train = docs_to_indices(docs, word_to_num)
		X_train, D_train = seqs_to_lmXY(S_train)

		# Load the dev set (for tuning hyperparameters)
		docs = load_dataset(data_folder + '/ptb-dev.txt')
		S_dev = docs_to_indices(docs, word_to_num)
		X_dev, D_dev = seqs_to_lmXY(S_dev)

		X_train = X_train[:50]
		D_train = D_train[:50]
		X_dev_train = X_dev[:25]
		D_dev_train = D_dev[:25]

		# q = best unigram frequency from omitted vocab
		# this is the best expected loss out of that set
		q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

		##########################
		#I have found that the best parameters are learning rate = 0.5,look_back=2, hid_un=25
		rnn = RNN(vocab_size, hdim)
		loss = rnn.train(X_train, D_train, X_dev_train, D_dev_train, epochs=10, learning_rate=lr, anneal=5, back_steps=lookback,
						 batch_size=100, min_change=0.0001, log=True)
		##########################
		print loss

		np.save(" rnn.U.npy", rnn.U)
		np.save(" rnn.V.npy", rnn.V)
		np.save(" rnn.W.npy", rnn.W)

		run_loss = -1
		adjusted_loss = -1
		run_loss=rnn.compute_mean_loss(X_dev,D_dev)
		adjust_loss(run_loss,fraction_lost,q,mode="basic")
		print "Unadjusted: %.03f" % np.exp(run_loss)
		print "Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss)

	if mode == "generate":
		'''
		starter code for sequence generation
		'''
		data_folder = sys.argv[2]
		rnn_folder = sys.argv[3]
		maxLength = int(sys.argv[4])

		# get saved RNN matrices and setup RNN
		U,V,W = load(rnn_folder + "/rnn.U.npy"), load(rnn_folder + "/rnn.V.npy"), load(rnn_folder + "/rnn.W.npy")
		vocab_size = len(V[0])
		hdim = len(U[0])

		r = RNN(vocab_size, hdim)
		r.U = U
		r.V = V
		r.W = W

		# get vocabulary
		vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		word_to_num = invert_dict(num_to_word)

		for i in range(1000):

			sequence,run_loss = r.generate_sequence(word_to_num["<s>"], word_to_num["</s>"], maxLength, word_to_num["UUUNKKK"])
			if(run_loss<3.0 and len(sequence) >= 7):
				print "Predicted sentence: \n"

				generated_sentence = num_to_word[sequence[0]] #it will always be the start symbol
				if len(sequence) > 1:
					for index in sequence[1:]:
						generated_sentence += " " + num_to_word[index]

				print generated_sentence

				print "\nReporting the mean loss and perplexity of the generated sentence..."
				print "Mean loss: %.03f" % run_loss
				print "Unadjusted Perplexity: %.03f" % np.exp(run_loss)