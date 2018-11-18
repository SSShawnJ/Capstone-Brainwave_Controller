import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

def extract_training_data(SEGMENT_SIZE=15):

	training_file_path_nonstop = ["../data/lawrence/power_data_law_left.csv",
							"../data/lawrence/power_data_law_right.csv",
							"../data/lawrence/power_data_law_forward.csv",
							"../data/alex/power_data_alex_left.csv",
							"../data/alex/power_data_alex_right.csv",
							"../data/alex/power_data_alex_forward.csv",
							"../data/arfa/power_data_arfa_left.csv",
							# "../data/arfa/power_data_law_right.csv",
							"../data/arfa/power_data_arfa_forward.csv",
							]
	training_file_path_stop = [ "../data/lawrence/power_data_law_stop.csv",
								"../data/alex/power_data_alex_stop.csv",
								"../data/arfa/power_data_arfa_stop.csv",
								]
	output_file_path = "../data/training/training_data.csv"



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


if __name__ == '__main__':
	#extract_training_data()

	feature, y = load_data()
	feature = np.c_[feature[:,0:20],feature[:,40:60]]
	print(feature.shape)
	np.random.seed(seed=2)

	feature,y = shuffle(feature,y)
	#np.savetxt("Features.csv", feature, delimiter=",")
	#np.savetxt("Labels.csv", y, delimiter=",")

	X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.4, random_state=0)

	clf = svm.SVC(kernel='rbf', C=50)
	clf.fit(X_train,y_train)

	pred_training = clf.predict(X_train)
	pred=clf.predict(X_test)

	print("training_accuracy", accuracy_score(y_train,pred_training))
	print("test_accuracy:",accuracy_score(y_test,pred))
	print("test f1 score (each class):", f1_score(y_test, pred, average=None) )
	print("test f1 score (weigted):", f1_score(y_test, pred, average='weighted') )
	print(y_test)
	print(pred)









