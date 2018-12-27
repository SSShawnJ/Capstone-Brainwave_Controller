import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

def extract_training_data(SEGMENT_SIZE=15):
	import os

	training_file_path_nonstop = ["../data/lawrence/power_data_law_left.csv",
							"../data/lawrence/power_data_law_right.csv",
							"../data/lawrence/power_data_law_forward.csv",
							"../data/alex/power_data_alex_left.csv",
							"../data/alex/power_data_alex_right.csv",
							"../data/alex/power_data_alex_forward.csv",
							"../data/arfa/power_data_arfa_left.csv",
							# "../data/arfa/power_data_law_right.csv",
							"../data/arfa/power_data_arfa_forward.csv",
							"../data/shawn/power_data_shawn_left.csv",
							"../data/shawn/power_data_shawn_right.csv",
							"../data/shawn/power_data_shawn_forward.csv"
							]
	training_file_path_stop = [ "../data/lawrence/power_data_law_stop.csv",
								"../data/alex/power_data_alex_stop.csv",
								"../data/arfa/power_data_arfa_stop.csv",
								"../data/shawn/power_data_shawn_stop.csv",
								]
	output_file_path = "../data/training/training_data.csv"
	os.system("rm "+ output_file_path)


	for FILE in training_file_path_nonstop:
		f = open(FILE, "r")
		f.readline()

		element_count = 0
		array = []

		for line in f:
			row = line.split(",")
			if (row[1]=="0"):
				continue

			array_temp = []
			for num in row:
				array_temp.append(float(num))
			array.append(array_temp)
		f.close()

		arr_sum = np.zeros(len(array[0])-2)
		#print(arr_sum.shape)
		label = int(array[0][1])
		#print(label)

		output_features = []
		output_labels = []

		for arr in array:
			arr_sum += np.array(arr[2:])
			element_count +=1

			if element_count == SEGMENT_SIZE:
				arr_sum /=SEGMENT_SIZE
				output_features.append(arr_sum)
				output_labels.append(label)

				arr_sum = np.zeros(len(array[0])-2)
				element_count=0

		if element_count != 0:
			arr_sum /=element_count
			output_features.append(arr_sum)
			output_labels.append(label)

		data=np.c_[output_features,output_labels]
		f = open(output_file_path,'a')
		np.savetxt(f, data, delimiter=",")
		f.close()

	for FILE in training_file_path_stop:
		f = open(FILE, "r")
		f.readline()

		element_count = 0
		array = []

		for line in f:
			row = line.split(",")

			array_temp = []
			for num in row:
				array_temp.append(float(num))
			array.append(array_temp)
		f.close()

		arr_sum = np.zeros(len(array[0])-2)
		#print(arr_sum.shape)
		label = 0

		#print(label)

		output_features = []
		output_labels = []

		for arr in array:
			arr_sum += np.array(arr[2:])
			element_count +=1

			if element_count == SEGMENT_SIZE:
				arr_sum /=SEGMENT_SIZE
				output_features.append(arr_sum)
				output_labels.append(label)

				arr_sum = np.zeros(len(array[0])-2)
				element_count=0

		if element_count != 0:
			arr_sum /=element_count
			output_features.append(arr_sum)
			output_labels.append(label)

		data=np.c_[output_features,output_labels]
		f = open(output_file_path,'a')
		np.savetxt(f, data, delimiter=",")
		f.close()

def load_data(path_to_data = "../data/training/training_data.csv"):

	features=[]
	y=[]

	f=open(path_to_data,"r")
	for line in f:
		array=line.strip().split(",")
		for i in range(len(array)-1):
			array[i]=float(array[i])
		features.append(array[:-1])
		y.append(int(float(array[-1])))
	f.close()

	return np.array(features),np.array(y)

def create_training_data():
	feature, y = load_data()

	# Do not use relative Bnad Powers for now .
	# It can have NaN problem and also does not effect the final result much.
	feature = np.c_[feature[:,0:20],feature[:,40:60]]
	# print("feature vector shape:",feature.shape)
	data = np.c_[feature, y]
	np.random.seed(seed=2)
	data=shuffle(data)
	np.savetxt("../data/training/training_set.csv", data, delimiter=",")


if __name__ == '__main__':
	# extract_training_data(SEGMENT_SIZE=3)
	# create_training_data()


	#### Actual Training ####
	#
	# TODO:
	# 1. Try to improve the overral training accuracy and F1 score.
	#     a. Hyperparamter tuning (SEGMENT_SIZE, model related parameters, etc.)
	#     b. Try other classifiers (Neural Networks(MLP, 1-D CNN, RNN), Decision Tree, etc.)
	# 2. Try to use less features to achieve resonable accuracy.
	#
	# 
	# Current Benchmark (TRY TO IMPROVE IT!):
	# training_accuracy 0.9931437277805993
	# test_accuracy: 0.9827411167512691
	# test F1 score (each class): [0.9978678  0.98168498 0.97272727 0.97864078]
	# test F1 score (weigted): 0.9827235943012146
	#
	#########################
	import joblib

	feature, y = load_data("../data/training/training_set.csv")
	print("feature vector shape:",feature.shape)

	X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.2, random_state=0)

	clf = svm.SVC(kernel='rbf', C=45)
	clf.fit(X_train,y_train)
	#clf = joblib.load('svm.joblib')

	pred_training = clf.predict(X_train)
	pred = clf.predict(X_test)

	joblib.dump(clf, 'svm.joblib') 

	print("Hyperparameter Value:\n\tSEGMENT_SIZE:3\n\tSVM kernel:'rbf'\n\tSVM C:45")
	print("training_accuracy", accuracy_score(y_train,pred_training))
	print("test_accuracy:",accuracy_score(y_test,pred))
	print("test F1 score (each class):", f1_score(y_test, pred, average=None) )
	print("test F1 score (weigted):", f1_score(y_test, pred, average='weighted') )
	# print(y_test)
	# print(pred)
	for i in range(len(y_test)):
		if y_test[i] != pred[i]:
			print("predict value:%d, truth value:%d" % (pred[i], y_test[i]))









