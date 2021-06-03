from pprint import pprint
from Parser import Parser,Parser_ch
import util
import os
import glob
import math
import nltk
import jieba
import argparse
import time
import json
import re
import csv
from nltk.corpus import stopwords
import pymysql
import numpy as np
class VectorSpace:
	""" A algebraic model for representing text documents as vectors of identifiers. 
	A document is represented as a vector. Each dimension of the vector corresponds to a 
	separate term. If a term occurs in the document, then the value in the vector is non-zero.
	"""

	#Collection of document term vectors
	documentVectors = []
	documentVectors_tf_idf = []

	#Mapping of vector index to keyword
	vectorKeywordIndex=[]

	#Tidies terms
	parser=None
	
	IDFVector = []

	ischinese=False

	def __init__(self, documents=[],doc_id=[], ischinese=False):
		self.documentVectors=[]
		self.ischinese=ischinese

		if ischinese == False :
			self.parser = Parser()
		elif ischinese == True :
			self.parser = Parser_ch()
		
		if(len(documents)>0):
			self.build(documents,doc_id)


	def build(self,documents,doc_ids):
		""" Create the vector space for the passed document strings """
		self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
		print('over index')
		
		self.IDFVector = self.getIDFVector(documents)
		print('over idf')
		
		for document in documents:
			doc_tfidf_vector = [a*b for a,b in zip( self.IDFVector, self.makeVector(document) )]
			doc_vector_norm = doc_tfidf_vector/np.linalg.norm(doc_tfidf_vector))
			
			### do svd ###


		#self.documentVectors = [self.makeVector(document) for document in documents]
		#print('over docvec')
		#self.documentVectors_tf_idf = [[a*b for a,b in zip(self.IDFVector,documentVector)] for documentVector in self.documentVectors]
		#print('over docvec tfidf')
	def getVectorKeywordIndex(self, documentList):
		""" create the keyword associated to the position of the elements within the document vectors """
		#Mapped documents into a single word string vocabularyString = " ".join(documentList)
		vocabularyString = " ".join(documentList)
		
		#(English mode)in tokenise function, vocabularyString will be removed punctuations, stemmed, splited to a list of words.
		#(Chinese mode)in tokenise function, vocabularyString will be removed punctuations, segmented to a list of words by jieba. 
		vocabularyList = self.parser.tokenise(vocabularyString)
			
		#Remove common words which have no search value		 
		vocabularyList = self.parser.removeStopWords(vocabularyList)
		uniqueVocabularyList = util.removeDuplicates(vocabularyList)

		vectorIndex={}
		offset=0
		#Associate a position with the keywords which maps to the dimension on the vector used to represent this word		 
		for word in uniqueVocabularyList:
			vectorIndex[word]=offset
			offset+=1
		
		return vectorIndex	#(keyword:position)

	def getIDFVector(self, documentList):

		count = [0] * len(self.vectorKeywordIndex)

		for doc in documentList:
			docstring = self.parser.tokenise(doc)
			uniquedocstring = util.removeDuplicates(docstring)		
			for word in uniquedocstring:
				if self.vectorKeywordIndex.get(word)!= None:
					count[self.vectorKeywordIndex[word]] += 1
		IDF = [math.log(len(documentList)/word) for word in count]

		return IDF


	def makeVector(self, wordString,no_tokenise=False):
		""" @pre: unique(vectorIndex) """

		#Initialise vector with 0's
		vector = [0] * len(self.vectorKeywordIndex)
		if no_tokenise == False:
			# if wordString is a document or an English input query, it has to be tokenize
			wordList = self.parser.tokenise(wordString)
		elif no_tokenise == True:
			# if wordString is a chinese input query, we don't have to tokenize again couse it has been split after input
			wordList = wordString
		for word in wordList:
			#some of words are stop words, so they may not exist in vectorKeywordIndex
			if self.vectorKeywordIndex.get(word)!= None:
				vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
		

		return vector


	def buildQueryVector(self, termList,only_NV=False):
		""" convert query string into a term vector """
		# boolean value only_NV is used to determine whether the function should ignore non-nounce and non-verb words.
		if only_NV == True:
			pos_tags = nltk.pos_tag(termList)
			tags = set(['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ'])
			termList = [word for word,pos in pos_tags if pos in tags]
			query = self.makeVector(termList, no_tokenise=True)

		if only_NV == False:
			query = self.makeVector(termList,self.ischinese)

		return query


	def related(self,documentId):
		""" find documents that are related to the document indexed by passed Id within the document Vectors"""
		ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
		return ratings


	def search(self,searchList,sim_func="cosine",TF_IDF=False):
		""" search for documents that match based on a list of terms """
		# set sim_func "cosine" to calculate Cosine similarity
		# set sim_func "euclidean" to calculate Euclidean Distance. The smaller the distance, the higher the similarity.
		# boolean value TF_IDF is used to determine whether take IDF into similarity calculation
		queryVector = self.buildQueryVector(searchList)
		
		if sim_func == "cosine" :
			if(TF_IDF==True):
				ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors_tf_idf]
			elif(TF_IDF==False):
				ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
		
		elif sim_func == "euclidean" :
			if(TF_IDF==True):
				ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors_tf_idf]
			elif(TF_IDF==False):
				ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]

		return ratings

	def search_relevance_feedback(self,original,feedback):
		""" search for documents that match based on a list of terms """
		originalVector = self.buildQueryVector(original)
		feedbackVector = self.buildQueryVector(feedback,only_NV=True)
		queryVector = [ o+f*0.5 for o,f in zip(originalVector,feedbackVector)]
		ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors_tf_idf]
		
		return ratings

def clean(string):
	stop_words = set(stopwords.words('english')) 
	string = ' '.join(string)

	""" remove any nasty grammar tokens from string """
	puncs = ["!",";",",",":",".","?","(",")","[","]","{","}","'",'"',"<",">","&"]
	for punc in puncs: 
		string = string.replace(punc," ") 
	
	string = string.replace("\n"," ")
	string = string.replace("\s+"," ")
	string = string.lower()
	
	string = [word for word in string.split(" ") if word not in stop_words]
	string = ' '.join(string)
	string = re.sub(' +', ' ',string)
	return string

def read_news_from_db():
	documents = []
	doc_id = []
	# SQL query: read
	print('start')	
	with open("../msn_train_withID.json") as jsonfile:
		news = json.load(jsonfile)
		for element in news["all"]:
			doc_id.append(element["newsID"])
			documents.append(clean(element["body"]))
	print('over')
	
	with open("../msn_test_withID.json") as jsonfile:
		news = json.load(jsonfile)
		for element in news["all"]:
			doc_id.append(element["newsID"])
			documents.append(clean(element["body"]))
	print('over')
	
	with open("../msn_dev_withID.json") as jsonfile:
		news = json.load(jsonfile)
		for element in news["all"]:
			doc_id.append(element["newsID"])
			documents.append(clean(element["body"]))
	print('over')


	return documents, doc_id

def isContainChinese(s):
	for c in s:
		if ('\u4e00' <= c <= '\u9fa5'):
			return True
	return False


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
		
	documents, doc_id = read_news_from_db()
	
	vectorSpace = VectorSpace(documents,doc_id)
	

###################################################
