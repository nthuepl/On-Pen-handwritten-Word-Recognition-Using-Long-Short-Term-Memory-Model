from spell import correction
from Levenshtein import levenshtein, index
import numpy as np
 

class Lexicon():
	def __init__(self):

		vocab_list = []
		# load from file
		fp = open('20k.txt', 'r')
		for line in fp.readlines():
			word = line[:-1]
			vocab_list.append(word)
		fp.close()

		self.word_list = []
		self.page_table = [0 for i in range(20)]
		self.cnt = -1
		for length in range(1, 18):
			self.page_table[length] = len(self.word_list)
			word_n = []
			for i in range(len(vocab_list)):
				if len(vocab_list[i]) == length:
					word_n.append(vocab_list[i])
			word_n = sorted(word_n, key=str.upper)
			self.word_list += word_n

		au_w, bp_w, ce_w, fl_w, hn_w, rv_w = (0.136, 0.695, 0.628, \
													0.501, 0.917, 0.139)
		sub_w, del_w, ins_w = (1.389, 1.925, 1.954)
		self.del_map = np.array([del_w for i in range(26)])
		self.ins_map = np.array([ins_w for i in range(26)])
		self.sub_map = np.array([[sub_w for i in range(26)] for j in range(26)])
		self.sub_map[index('a'), index('u')] = au_w
		self.sub_map[index('p'), index('b')] = bp_w
		self.sub_map[index('e'), index('c')] = ce_w
		self.sub_map[index('l'), index('f')] = fl_w
		self.sub_map[index('r'), index('v')] = rv_w
		self.sub_map[index('n'), index('h')] = hn_w

	def __iter__(self):
		return self

	def __next__(self):
		if self.cnt >= len(self.word_list):
			raise StopIteration
		else:
			self.cnt += 1
			return str(self.word_list[cnt])

	def __str__(self):
		return "Lexicon:\n\t"+str(len(self.word_list))+" words\n\t"+"page table: "+str(self.page_table)

	def page(self, i):
		return str(self.word_list[i])

	def index(self, i):
		return int(self.page_table[i])

	def tolist(self):
		return list(self.word_list)

	def leven_fit(self, word, area=None):
		answer = ''
		MIN = 20
		head, tail = 0, len(self.word_list)-1
		if area != None:
			head, tail = area
		# for w in lexicon:
		for w in self.word_list[head:tail]:
			d = levenshtein(word, w, insert_costs=self.ins_map, delete_costs=self.del_map, substitute_costs=self.sub_map)
			if d < MIN:
				MIN = d
				answer = w
				if d == 0:
					break
		return answer

class Corrector():
	def __init__(self, lexicon=None):
		self.lex = lexicon
		if lexicon == None:
			self.lex = Lexicon()
		print(self.lex)

		self.MSR = {'r': 0.09, 'other': 0.514, 'n': 0.196, 'e': 0.139, 'l': 0.208, 'p': 0.227, 'a': 0.142}
		self.dw = 0.459

	def correction(self, word):
		tmp = str(word)
		word = correction(word, MSR=self.MSR, distance_weight=self.dw)

		if word == None:
			word = self.lex.leven_fit(tmp, area=(self.lex.index(len(tmp)), self.lex.index(len(tmp)+2)))

		return word

if __name__ == '__main__':

	corrector = Corrector()
	print(corrector.lex)
