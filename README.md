Using Active Learning to Expand Training Data for Implicit Discourse Relation Recognition
=====

It is slightly simplified implementation of our Using Active Learning to Expand Training Data for Implicit Discourse Relation Recognition paper in Tensorflow.


Requirements
-----
	Python 3.5
	Tensorflow 1.4
	Numpy
	sklearn
  
Preprocess:
----
	training a basic model: python train.py 
	Run by excuting the command of train.py. The default parameters are used.
  	
	active learning step: python active_learning_2.py
	
Data sets:
----
	PDTB2.0
	Sec 00~01 implicit: dev set
	Sec 02~20 implicit: train set
	Sec 21~22 implicit: test set
	Sec 00~24 explicit: unlabeled set
  
 
