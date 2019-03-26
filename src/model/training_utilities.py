import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn import tree
# from sklearn.preprocessing import normalize

key_map = {0:"stop", 1:"left",2:"right",3:"forward"}

def extract_training_data(SEGMENT_SIZE=15):
	import os

	training_file_path_nonstop = [#"../data/lawrence/power_data_law_left.csv",
	# 						"../data/lawrence/power_data_law_right.csv",
	# 						"../data/lawrence/power_data_law_forward.csv",
	# 						"../data/alex/power_data_alex_left.csv",
	# 						"../data/alex/power_data_alex_right.csv",
	# 						"../data/alex/power_data_alex_forward.csv",
	# 						"../data/arfa/power_data_arfa_left.csv",
	# 						# "../data/arfa/power_data_law_right.csv",
	# 						"../data/arfa/power_data_arfa_forward.csv",
							# "../data/shawn/power_data_shawn_left.csv",
							# "../data/shawn/power_data_shawn_left_2.csv",
							# "../data/shawn/power_data_shawn_left_3.csv",
							# "../data/shawn/power_data_shawn_left_4.csv",
							# "../data/shawn/power_data_shawn_left_5.csv",
							# "../data/shawn/power_data_shawn_right.csv",
							# "../data/shawn/power_data_shawn_right_2.csv",
							# "../data/shawn/power_data_shawn_right_3.csv",
							# "../data/shawn/power_data_shawn_forward.csv",
							# "../data/shawn/power_data_shawn_forward_2.csv",
							# "../data/shawn/power_data.csv",
							# "../data/shawn/power_data 2.csv",
							# "../data/shawn/power_data_r.csv",
							"../data/shawn/power_data_l.csv",
							# "../data/shawn/power_data_f.csv",
							"../data/power_data_ff.csv",
							"../data/power_data_fff.csv",
							#"../data/power_data_ffff.csv",
							# "../data/power_data_r.csv",
							# "../data/power_data_rr.csv",
							# "../data/power_data_rrr.csv",
							# "../data/power_data_rrrr.csv",
							"../data/power_data_rrrrr.csv",
							"../data/power_data_rrrrrrrr.csv",
							"../data/power_data_l.csv",
							# "../data/power_data_f.csv"
							]
	training_file_path_stop = [ #"../data/lawrence/power_data_law_stop.csv",
								#"../data/alex/power_data_alex_stop.csv",
								#"../data/arfa/power_data_arfa_stop.csv",
								"../data/shawn/power_data_shawn_stop.csv",
								"../data/shawn/power_data_shawn_stop_2.csv",
								"../data/shawn/power_data_shawn_stop_3.csv"

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

		arr_sum = []
		#print(arr_sum.shape)
		label = int(array[0][1])
		#print(label)

		output_features = []
		output_labels = []

		for arr in array:
			arr_sum.append(arr[2:])

			if len(arr_sum) == SEGMENT_SIZE:
				feature = np.sum(arr_sum, axis = 0)
				output_features.append(feature)
				output_labels.append(label)

				arr_sum = []

		# if len(arr_sum) > 0:
		# 	arr_sum /=element_count
		# 	output_features.append(arr_sum)
		# 	output_labels.append(label)

		data=np.c_[output_features,output_labels]
		f = open(output_file_path,'a')
		np.savetxt(f, data, delimiter=",")
		f.close()

	for FILE in training_file_path_stop:
		f = open(FILE, "r")
		f.readline()

		array = []

		for line in f:
			row = line.split(",")

			array_temp = []
			for num in row:
				array_temp.append(float(num))
			array.append(array_temp)
		f.close()

		arr_sum = []
		#print(arr_sum.shape)
		label = 0

		#print(label)

		output_features = []
		output_labels = []

		for arr in array:
			arr_sum.append(arr[2:])

			if len(arr_sum) == SEGMENT_SIZE:
				feature = np.sum(arr_sum, axis = 0)
				output_features.append(feature)
				output_labels.append(label)

				arr_sum = []

		# if element_count != 0:
		# 	arr_sum /=element_count
		# 	output_features.append(arr_sum)
		# 	output_labels.append(label)

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
		arr = []
		for i in range(len(array)-1):
			arr.append(float(array[i]))
		features.append(arr)
		y.append(int(float(array[-1])))
	f.close()

	return np.array(features),np.array(y)

def create_training_data():
	feature, y = load_data()

	# Do not use relative Bnad Powers for now .
	# It may have NaN problem and also does not effect the final result much.
	#feature = feature[:,0:20]
	feature = np.c_[feature[:,0:20],feature[:,40:60]]
	# print("feature vector shape:",feature.shape)
	data = np.c_[feature, y]
	# np.random.seed(seed=2)
	data=shuffle(data)
	np.savetxt("../data/training/training_set.csv", data, delimiter=",")


def binary_model(feature, y, model = "svm", kernel = 'rbf', C = 1, gamma = 0.7, min_samples_split = 8):
	for i in range(len(key_map)):
		y_copy = np.copy(y)
		for j in range(len(y_copy)):
			if y_copy[j] != i:
				y_copy[j] = 1 
			else:
				y_copy[j] = 0

		print("feature vector shape:",feature.shape)
		print("label vector shape:",y_copy.shape)
		unique, counts = np.unique(y_copy, return_counts=True)
		class_weight = (dict(zip(unique, counts)))
		print(class_weight)

		X_train, X_test, y_train, y_test = train_test_split(feature, y_copy, test_size=0.2)
		#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

		
		if model == "dt":
			clf = tree.DecisionTreeClassifier(min_samples_split = min_samples_split)
		else:
			if i == 2:
				clf = svm.SVC(kernel= kernel, C = C, gamma = gamma, class_weight = 'balanced')
			else:
				clf = svm.SVC(kernel= kernel, C= C, gamma = gamma, class_weight = 'balanced')

		clf.fit(X_train,y_train)
		#clf = joblib.load('svm.joblib')
		#clf.fit(X_train,y_train)
		#clf = SVMModel('svm.joblib')

		pred_training = clf.predict(X_train)
		pred = clf.predict(X_test)

		joblib.dump(clf, model+"_"+ key_map.get(i)+'.joblib') 
		estimator(y_train, y_test, pred_training, pred)

def multi_class_model(feature, y, model = "svm", kernel = 'rbf', C = 1, min_samples_split = 8):
	# for j in range(len(y)):
	# 		if y[j] == 3:
	# 			y[j] = 1 
	print("feature vector shape:",feature.shape)
	print("label vector shape:",y.shape)
	unique, counts = np.unique(y, return_counts=True)
	class_weight = (dict(zip(unique, counts)))
	print(class_weight)

	X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.2)
	#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

	
	if model == "dt":
		clf = tree.DecisionTreeClassifier(min_samples_split = min_samples_split)
	else:
		clf = svm.SVC(kernel= kernel, probability=True, C = C)

	clf.fit(X_train,y_train)
	#clf = joblib.load('svm.joblib')
	#clf.fit(X_train,y_train)
	#clf = SVMModel('svm.joblib')

	pred_training = clf.predict(X_train)
	pred = clf.predict(X_test)

	joblib.dump(clf, model+".joblib") 
	estimator(y_train, y_test, pred_training, pred)

def estimator(y_train, y_test, pred_training, pred):
	#print("Hyperparameter Value:\n\tSEGMENT_SIZE:3\n\tSVM kernel:'rbf'\n\tSVM C:45")
	print("training_accuracy", accuracy_score(y_train,pred_training))
	print("test_accuracy:",accuracy_score(y_test,pred))
	print("test F1 score (each class):", f1_score(y_test, pred, average=None) )
	print("test F1 score (weigted):", f1_score(y_test, pred, average='weighted') )
	# print(y_test)
	# print(pred)
	# for i in range(len(y_test)):
	# 	if y_test[i] != pred[i]:
	# 		print("predict value:%d, truth value:%d" % (pred[i], y_test[i]))

def PCA(X, n_components = 30):
	from sklearn.decomposition import PCA
	pca = PCA(n_components)
	X_new = pca.fit_transform(X)

	print(pca.explained_variance_)  
	#print(pca.get_covariance())

	print(pca.singular_values_)  
	return X_new

if __name__ == '__main__':
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
	extract_training_data(SEGMENT_SIZE=3)
	create_training_data()

	feature, y = load_data("../data/training/training_set.csv")
	#feature_new = PCA(feature)
	binary_model(feature, y, model="svm", kernel = 'rbf', C = 0.06, gamma = 'scale')
	#multi_class_model(feature, y, model="dt", min_samples_split = 8)
	#multi_class_model(feature, y, model="svm", C = 5, kernel = 'rbf')






