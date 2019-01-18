import numpy as np
import joblib
from sklearn import svm

class DTModelBinary():
	def __init__(self, path_to_model):
		self.clf_stop = joblib.load(path_to_model+"dt_stop.joblib")
		self.clf_left = joblib.load(path_to_model+"dt_left.joblib")
		self.clf_right = joblib.load(path_to_model+"dt_right.joblib")
		self.clf_forward = joblib.load(path_to_model+"dt_forward.joblib")

	def predict(self, data):
		if self.clf_stop.predict(data)[0] == 0:
			return 0 #stop
		elif self.clf_right.predict(data)[0] == 0:
			return 2 #right
		elif self.clf_forward.predict(data)[0] == 0:
			return 3 #forward
		elif self.clf_left.predict(data)[0] == 0:
			return 1 #left
		
		return 4

class DTModelMulti():
	def __init__(self, path_to_model):
		self.clf = joblib.load(path_to_model+"dt.joblib")
		

	def predict(self, data):
		return self.clf.predict(data)[0]

class SVMModelBinary():
	def __init__(self, path_to_model):
		self.clf_stop = joblib.load(path_to_model+"svm_stop.joblib")
		self.clf_left = joblib.load(path_to_model+"svm_left.joblib")
		self.clf_right = joblib.load(path_to_model+"svm_right.joblib")
		self.clf_forward = joblib.load(path_to_model+"svm_forward.joblib")

	def predict(self, data):
		stop = self.clf_stop.predict(data)[0]
		right = self.clf_right.predict(data)[0]
		forward = self.clf_forward.predict(data)[0]
		left = self.clf_left.predict(data)[0]

		if stop+right+forward+left == 3:
			if stop == 0:
				return 0 #stop
			elif right == 0:
				return 2 #right
			elif forward == 0:
				return 3 #forward
			elif left == 0:
				return 1 #left




		# if self.clf_stop.predict(data)[0] == 0:
		# 	return 0 #stop
		# elif self.clf_right.predict(data)[0] == 0:
		# 	return 2 #right
		# elif self.clf_forward.predict(data)[0] == 0:
		# 	return 3 #forward
		# elif self.clf_left.predict(data)[0] == 0:
		# 	return 1 #left
		
		return 4

class SVMModelMulti():
	def __init__(self, path_to_model):
		self.clf = joblib.load(path_to_model+"svm.joblib")
		

	def predict(self, data):
		return self.clf.predict(data)[0]

# if __name__ == '__main__':
# 	model = SVMModel("../model/svm.joblib")
# 	data = np.zeros((1,40))
# 	print(model.predict(data)[0])
