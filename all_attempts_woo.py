import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as labenc, OneHotEncoder as onehot
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.externals import joblib



# part I: first attempts; some struggling



# read data
gen_age = pd.read_csv('gender_age_train.csv', dtype = {'device_id': np.str})
phone_brand = pd.read_csv('phone_brand_device_model.csv', dtype = {'device_id': np.str})



# "translate" phone_brand and device_model to alphabets
alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

# grab Chinese keys and alphabet values
brand_keys = phone_brand.phone_brand.unique()
model_keys = phone_brand.device_model.unique()
brand_values = [alpha_1 + alpha_2 for alpha_1 in alpha for alpha_2 in alpha]
brand_values = [brand_values[i] for i in range(len(brand_keys))]

# translate with a dictionary for brands
brand_dict = {key: value for key, value in zip(brand_keys, brand_values)}
phone_brand['phone_brand_en'] = [brand_dict[ch] for ch in phone_brand.phone_brand]

# and likewise for models:
model_values = [alpha_1 + alpha_2 + alpha_3 for alpha_1 in alpha for alpha_2 in alpha for alpha_3 in alpha]
model_values = [model_values[i] for i in range(len(model_keys))]
model_dict = {key: value for key, value in zip(model_keys, model_values)}
phone_brand['device_model_en'] = [model_dict[ch] for ch in phone_brand.device_model]

# some cleaning
del alpha, brand_dict, model_dict, brand_keys, model_keys
phone_brand = phone_brand.drop(['phone_brand', 'device_model'], 1)

# merge data
train = gen_age.merge(phone_brand, how = 'left', on = 'device_id')
train = train.drop_duplicates()
del gen_age



# encode brands and models into numeric labels
brands_encoder = labenc()
models_encoder = labenc()
brands_encoder.fit(brand_values)
models_encoder.fit(model_values)
brands_encoded = brands_encoder.transform(brand_values)
models_encoded = models_encoder.transform(model_values)
brands_encoded = brands_encoded.reshape(len(brands_encoded), 1)
models_encoded = models_encoded.reshape(len(models_encoded), 1)

# encode numeric labels into one-hots
brands_onehot = onehot(n_values = len(brands_encoded), handle_unknown = 'ignore')
models_onehot = onehot(n_values = len(models_encoded), handle_unknown = 'ignore')
brands_onehot.fit(brands_encoded)
models_onehot.fit(models_encoded)
del brand_values, model_values, phone_brand



# standardize age and encode class labels for building models
age_mean = train.age.mean()
age_variance = train.age.var()
train.age = (train.age - age_mean) / age_variance

y_group = train.group
gender_encoder = labenc()
group_encoder = labenc()
train.gender = gender_encoder.fit_transform(train.gender)
y_group = group_encoder.fit_transform(y_group)



# split training data for model tuning and generalization estimation
train, test, y_group_train, y_group_test = (
    train_test_split(train, y_group, test_size = 0.2, random_state = 0, stratify = y_group))
y_gender_train, y_gender_test = train['gender'], test['gender']
y_age_train, y_age_test = train['age'], test['age']
X_train, X_test = train[['phone_brand_en', 'device_model_en']], test[['phone_brand_en', 'device_model_en']]



# transform data with encoders
brand_labels_train, brand_labels_test = (
	brands_encoder.transform(X_train['phone_brand_en']),
	brands_encoder.transform(X_test['phone_brand_en']))
model_labels_train, model_labels_test = (
	models_encoder.transform(X_train['device_model_en']),
	models_encoder.transform(X_test['device_model_en']))
	
labels = [brand_labels_train, brand_labels_test, model_labels_train, model_labels_test]
labels = [L.reshape(len(L), 1) for L in labels]

brand_labels_train = labels[0]
brand_labels_test = labels[1]
model_labels_train = labels[2]
model_labels_test = labels[3]
del labels

brand_labels_train = brands_onehot.transform(brand_labels_train).toarray()
brand_labels_test = brands_onehot.transform(brand_labels_test).toarray()
model_labels_train = models_onehot.transform(model_labels_train).toarray()
model_labels_test = models_onehot.transform(model_labels_test).toarray()



# train and tune 1st order model parameters via grid search
classifier_brands = SGDClassifier(
	loss = 'log', 
	penalty = 'elasticnet', 
	random_state = 0, 
	verbose = 1, 
	n_jobs = 3)
classifier_models = SGDClassifier(
	loss = 'log', 
	penalty = 'elasticnet', 
	random_state = 0, 
	verbose = 1, 
	n_jobs = 3)
regressor_brands = SGDRegressor(
	loss = 'squared_loss', 
	penalty = 'elasticnet', 
	random_state = 0, 
	verbose = 1)
regressor_models = SGDRegressor(
	loss = 'squared_loss', 
	penalty = 'elasticnet', 
	random_state = 0, 
	verbose = 1)
	
gbm_classifier_brands = GradientBoostingClassifier(
	loss = 'exponential',
	min_samples_split = 1,
	random_state = 0)
gbm_classifier_models = GradientBoostingClassifier(
	loss = 'deviance',
	min_samples_split = 1,
	random_state = 0)
gbm_regressor_brands = GradientBoostingRegressor(
	loss = 'huber',
	min_samples_split = 1,
	random_state = 0)
gbm_regressor_models = GradientBoostingRegressor(
	loss = 'lad',
	min_samples_split = 1,
	random_state = 0)
	
params_1 = {'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
		  'l1_ratio': [i / 5 for i in range(1, 5)]}
params_2 = {'max_depth': [1, 2, 3, 5, 8]}

logistic_reg_brands = GridSearchCV(
	classifier_brands, 
	params_1, 
	scoring = 'log_loss', 
	n_jobs = 3, 
	cv = 5, 
	error_score = 0)
logistic_reg_models = GridSearchCV(
	classifier_models, 
	params_1, 
	scoring = 'log_loss', 
	n_jobs = 3, 
	cv = 5, 
	error_score = 0)
linear_reg_brands = GridSearchCV(
	regressor_brands, 
	params_1, 
	n_jobs = 3, 
	cv = 6, 
	error_score = 0)
linear_reg_models = GridSearchCV(
	regressor_models, 
	params_1, 
	n_jobs = 3, 
	cv = 5, 
	error_score = 0)
	
gbm_class_brands = GridSearchCV(
	gbm_classifier_brands, 
	params_2, 
	scoring = 'log_loss', 
	n_jobs = 3, 
	cv = 6,  
	error_score = 0)
gbm_class_models = GridSearchCV(
	gbm_classifier_models, 
	params_2, 
	scoring = 'log_loss', 
	n_jobs = 3, 
	cv = 5,  
	error_score = 0)

logistic_reg_brands.fit(brand_labels_train, y_gender_train)
logistic_reg_models.fit(model_labels_train, y_gender_train)
linear_reg_brands.fit(brand_labels_train, y_age_train)
linear_reg_models.fit(model_labels_train, y_age_train)
gbm_class_brands.fit(brand_labels_train, y_gender_train)
gbm_class_models.fit(model_labels_train, y_gender_train)



# save models for parameter generation for test dataset
models = ['logistic_reg_brands',
		  'logistic_reg_models',
		  'linear_reg_brands',
		  'linear_reg_models',
		  'gbm_class_brands',
		  'gbm_class_models']

for mod in models:
	joblib.dump(eval(mod + '.best_estimator_'), mod + '.pkl')




# train a set of models on both brands and models together
brand_model_labels_train = np.concatenate([brand_labels_train, model_labels_train], 1)
brand_model_labels_test = np.concatenate([brand_labels_test, model_labels_test], 1)


naive_gaussian = GaussianNB()
naive_multi = MultinomialNB()
naive_multi_params = {'alpha':[0, .0001, .005, .01, .1, .5, 1]}

ada_classifier = AdaBoostClassifier(random_state = 0)
ada_regressor = AdaBoostRegressor(random_state = 0)
ada_classifier_params = {'n_estimators': [150], 'learning_rate': [.01, .1, .5, .8, 1]}
ada_regressor_params = {'n_estimators': [150], 'learning_rate': [.04, .3, .6, .9]}

random_forest_classifier = RandomForestClassifier(criterion = 'entropy', 
												  min_samples_split = 1, 
												  n_jobs = 3, 
												  random_state = 0)
random_forest_regressor = RandomForestRegressor(min_samples_split = 1, 
												random_state = 0)
random_forest_params = {'n_estimators': [15, 40, 100]}



naive_multi = GridSearchCV(
	naive_multi, 
	naive_multi_params, 
	n_jobs = 3, 
	cv = 5,
	verbose = 1, 
	error_score = 0)
ada_classifier = GridSearchCV(
	ada_classifier, 
	ada_classifier_params, 
	n_jobs = 3, 
	cv = 5, 
	error_score = 0)
ada_regressor = GridSearchCV(
	ada_regressor, 
	ada_regressor_params, 
	n_jobs = 3, 
	cv = 5, 
	error_score = 0)
random_forest_classifier = GridSearchCV(
	random_forest_classifier,
	random_forest_params, 
	n_jobs = 3, 
	cv = 5, 
	error_score = 0)
random_forest_regressor = GridSearchCV(
	random_forest_regressor,
	random_forest_params, 
	n_jobs = 3, 
	cv = 5, 
	error_score = 0,
	verbose = 1)
	
bunch_of_classifiers = ['ada_classifier',
						'random_forest_classifier',
						'naive_multi']
bunch_of_regressors = ['ada_regressor',
					   'random_forest_regressor']

for classifier in bunch_of_classifiers:
	eval(classifier + '.fit(brand_model_labels_train, y_group_train)')
	joblib.dump(eval(classifier + '.best_estimator_'), classifier + '.pkl')
for regressor in bunch_of_regressors:
	eval(regressor + '.fit(brand_model_labels_train, y_age_train)')
	joblib.dump(eval(regressor + '.best_estimator_'), regressor + '.pkl')










# part II: categorical feature dummies as csr matrices



import pandas as pd		# code based on kaggle's dune_dweller
import numpy as np
from scipy.sparse import hstack		# for concatenating
from scipy.sparse import csr_matrix		# for manageable memory usage in generating dummy variables
from sklearn.metrics import log_loss	# cost metric for the competition
from sklearn.linear_model import LogisticRegression		# simple, fast model apt with sparse features
from sklearn.naive_bayes import MultinomialNB			# fast out-of-box model
from sklearn.ensemble import ExtraTreesClassifier		# cheaper random forest without bagging
from sklearn.svm import LinearSVC						# linear, faster support vector machine
from sklearn.preprocessing import LabelEncoder			# for encoding columns for csr_matrix dummies
from sklearn.grid_search import GridSearchCV			# for hyperparemeter optimization
from sklearn.externals import joblib		# for keeping models around


# load data
gen_age_train = pd.read_csv('gender_age_train.csv', index_col='device_id')		# setting index allows for convenient merge
gen_age_train['train_index'] = np.arange(len(gen_age_train))					# create new training and test indices for 
gen_age_test = pd.read_csv('gender_age_test.csv', index_col = 'device_id')		# compression into sparse row matrix
gen_age_test['test_index'] = np.arange(len(gen_age_test))
phone_brand_model = pd.read_csv('phone_brand_device_model.csv')		# there are 6 duplicate devices in brand_model
phone_brand_model = phone_brand_model.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv('events.csv', parse_dates=['timestamp'], index_col='event_id')
app_evs = pd.read_csv('app_events.csv', usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
													# a subset of features: the left out feature 'is_installed' is 
													# just a column of all ones
app_labs = pd.read_csv('app_labels.csv')					# take only the relevant label_ids for apps; many labels
app_labs = app_labs.loc[app_labs.app_id.isin(app_evs.app_id.unique())]		# do not have corresponding app_ids


# encode categorical features 
brand_enc = LabelEncoder().fit(phone_brand_model.phone_brand)
mod_brands = phone_brand_model.phone_brand.str.cat(phone_brand_model.device_model)	# concat model and brand,
model_enc = LabelEncoder().fit(mod_brands)						# because of duplicate model names for different brands
app_enc = LabelEncoder().fit(app_evs.app_id)
label_enc = LabelEncoder().fit(app_labs.label_id)


# set encoded category labels 
phone_brand_model['brand'] = brand_enc.transform(phone_brand_model['phone_brand'])
phone_brand_model['model'] = model_enc.transform(mod_brands)
app_evs['app'] = app_enc.transform(app_evs.app_id)
app_labs['app'] = app_enc.transform(app_labs.app_id)
app_labs['label'] = label_enc.transform(app_labs.label_id)

gen_age_train['brand'] = phone_brand_model['brand']
gen_age_train['model'] = phone_brand_model['model']
gen_age_test['brand'] = phone_brand_model['brand']
gen_age_test['model'] = phone_brand_model['model']


# merge device_ids via event_id onto app_events to match train/test indices with app labels for sparse matrices creation
all_device_apps = (app_evs.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
				  .groupby(['device_id','app'])['app'].agg(['size'])
                  .merge(gen_age_train[['train_index']], how='left', left_index=True, right_index=True)
                  .merge(gen_age_test[['test_index']], how='left', left_index=True, right_index=True)
                  .reset_index())			# merge with train/test indices for device identification
# likewise, match label_id...labels with train/test indices for sparse matrices creation
all_device_labels = (all_device_apps[['device_id','app']]
					.merge(app_labs[['app','label']])
					.groupby(['device_id','label'])['app'].agg(['size'])
					.merge(gen_age_train[['train_index']], how='left', left_index=True, right_index=True)
					.merge(gen_age_test[['test_index']], how='left', left_index=True, right_index=True)
					.reset_index())

train_apps = all_device_apps.dropna(subset=['train_index'])		# take all apps with train indices,
train_labels = all_device_labels.dropna(subset=['train_index'])	# all category labels with train indices,
test_apps = all_device_apps.dropna(subset=['test_index'])		# and all apps with test indices,
test_labels = all_device_labels.dropna(subset=['test_index'])	# and likewise for category labels


# create compressed sparse row matrices with dimensions n x l for n rows and l unique labels per feature,
# the cheaper alternative vs pandas.get_dummies and OneHotEncoder
brand_sparse_train = csr_matrix((np.ones(len(gen_age_train)), (gen_age_train.train_index, gen_age_train.brand)))
model_sparse_train = csr_matrix((np.ones(len(gen_age_train)), (gen_age_train.train_index, gen_age_train.model)))
brand_sparse_test = csr_matrix((np.ones(len(gen_age_test)), (gen_age_test.test_index, gen_age_test.brand)))
model_sparse_test = csr_matrix((np.ones(len(gen_age_test)), (gen_age_test.test_index, gen_age_test.model)))
app_sparse_train = csr_matrix((np.ones(len(train_apps)), (train_apps.train_index, train_apps.app)), 
							   shape=(len(gen_age_train), len(app_enc.classes_)))
app_sparse_test = csr_matrix((np.ones(len(test_apps)), (test_apps.test_index, test_apps.app)), 
							  shape=(len(gen_age_test), len(app_enc.classes_)))
label_sparse_train = csr_matrix((np.ones(len(train_labels)), (train_labels.train_index, train_labels.label)), 
								 shape=(len(gen_age_train), len(label_enc.classes_)))
label_sparse_test = csr_matrix((np.ones(len(test_labels)), (test_labels.test_index, test_labels.label)), 
								shape=(len(gen_age_test), len(label_enc.classes_)))
                      

# concatenate all sparse matrices for model fitting/prediction
train_sparse = hstack((brand_sparse_train, model_sparse_train, app_sparse_train, label_sparse_train), format='csr')
test_sparse =  hstack((brand_sparse_test, model_sparse_test, app_sparse_test, label_sparse_test), format='csr')

# encode target labels to prepare for model fitting
group_enc = LabelEncoder().fit(gen_age_train.group)
group_labels = group_enc.transform(gen_age_train.group)


# set models and parameters
naive = MultinomialNB()
machine = LinearSVC(penalty = 'l1', dual = False, random_state = 0)
perceptron = LogisticRegression(dual = False, solver = 'lbfgs', multi_class = 'multinomial')

naive_params = {'alpha':[.0001, .005, .01, .1, .5, 1]}
machine_params = {'C': [.0005, .001, .005, .01]}
perceptron_params = {'C': [.0005, .003, .008, .03]}

forest = ExtraTreesClassifier(min_samples_split = 1, random_state = 0, n_estimators = 120, n_jobs = 3)


# tune hyperparameters
naive = GridSearchCV(naive, naive_params, n_jobs = 3, cv = 5, error_score = 0)
machine = GridSearchCV(machine, machine_params, n_jobs = 3, cv = 5, error_score = 0)
perceptron = GridSearchCV(perceptron, perceptron_params, n_jobs = 3, cv = 5, error_score = 0)


# fit models
classifiers = ['naive', 'machine', 'perceptron']
for classifier in classifiers:
	eval(classifier + '.fit(train_sparse, group_labels)')
	joblib.dump(eval(classifier + '.best_estimator_'), classifier + '.pkl')
forest.fit(train_sparse, group_labels)
joblib.dump(forest, 'forest.pkl')

# output predictions for later usage
p1 = pd.DataFrame(naive.predict_proba(test_sparse), index = gen_age_test.index, columns = group_enc.classes_)
p2 = pd.DataFrame(naive.predict_proba(train_sparse), index = gen_age_train.index, columns = group_enc.classes_)
p3 = pd.DataFrame(machine.predict(test_sparse), index = gen_age_test.index, columns = ['linearSVC'])
p4 = pd.DataFrame(machine.predict(train_sparse), index = gen_age_train.index, columns = ['linearSVC'])
p5 = pd.DataFrame(perceptron.predict_proba(train_sparse), index = gen_age_train.index, columns = group_enc.classes_)
p6 = pd.DataFrame(perceptron.predict_proba(test_sparse), index = gen_age_test.index, columns = group_enc.classes_)
p7 = pd.DataFrame(forest.predict_proba(train_sparse), index = gen_age_train.index, columns = group_enc.classes_)
p8 = pd.DataFrame(forest.predict_proba(test_sparse), index = gen_age_test.index, columns = group_enc.classes_)
p1.to_csv('multinomialNB_prediction.csv', index = True)
p2.to_csv('multinomialNB_prediction_train.csv', index = True)
p3.to_csv('linearSVC_prediction.csv', index = True)
p4.to_csv('linearSVC_prediction_train.csv', index = True)
p5.to_csv('logisticReg_prediction_train.csv', index = True)
p6.to_csv('logisticReg_prediction_test.csv', index = True)
p7.to_csv('xrandom_trees_prediction_train.csv', index = True)
p8.to_csv('xrandom_trees_prediction.csv', index = True)




# part III: stacking experiments



# load predictions, except for the weakest one: linear support vector machine
logreg = pd.read_csv('logreg_predictions.csv', index_col = 'device_id')
logreg_train = pd.read_csv('logreg_predictions_train.csv', index_col = 'device_id')
naive = pd.read_csv('multinomialNB_predictions.csv', index_col = 'device_id')
naive_train = pd.read_csv('multinomialNB_predictions_train.csv', index_col = 'device_id')
xrandom = pd.read_csv('xrandom_trees_prediction.csv', index_col = 'device_id')
xrandom_train = pd.read_csv('xrandom_trees_prediction_train.csv', index_col = 'device_id')


# merge train/test predictions for new dataframe
test_pred = (logreg.merge(naive, left_index = True, right_index = True)
			.merge(xrandom, left_index = True, right_index = True)).as_matrix()
train_pred = (logreg_train.merge(naive_train, left_index = True, right_index = True)
			 .merge(xrandom_train, left_index = True, right_index = True)).as_matrix()


# load and encode group labels
gen_age_train = pd.read_csv('gender_age_train.csv', index_col = 'device_id')
gen_age_test = pd.read_csv('gender_age_test.csv', index_col = 'device_id')
group_enc = LabelEncoder().fit(gen_age_train.group)
group_labels = group_enc.transform(gen_age_train.group)


# simple model
perceptron = LogisticRegression(dual = False, solver = 'lbfgs', multi_class = 'multinomial')
perceptron_params = {'C': [.001, 0.005, 0.01, 0.05]}
perceptron = GridSearchCV(perceptron, perceptron_params, n_jobs = 3, cv = 5, error_score = 0)
perceptron.fit(train_pred, group_labels)

joblib.dump(perceptron.best_estimator_, 'logreg_2nd_layer.pkl')
p = pd.DataFrame(perceptron.predict_proba(test_pred), index = gen_age_test.index, columns = group_enc.classes_)
p.to_csv('logreg_layer2_prediction.csv', index = True)


# another one with the best estimators; let's see if we overfit
test1 = pd.read_csv('logreg_predictions.csv', index_col = 'device_id')
test2 = pd.read_csv('logreg_predictions2.csv', index_col = 'device_id')
test3 = pd.read_csv('logreg_predictions3.csv', index_col = 'device_id')
test4 = pd.read_csv('logreg_predictions4.csv', index_col = 'device_id')
train1 = pd.read_csv('logreg_predictions_train.csv', index_col = 'device_id')
train2 = pd.read_csv('logreg_predictions_train2.csv', index_col = 'device_id')
train3 = pd.read_csv('logreg_predictions_train3.csv', index_col = 'device_id')
train4 = pd.read_csv('logreg_predictions_train4.csv', index_col = 'device_id')

gen_age_train = pd.read_csv('gender_age_train.csv', index_col = 'device_id')
gen_age_test = pd.read_csv('gender_age_test.csv', index_col = 'device_id')
group_enc = LabelEncoder().fit(gen_age_train.group)
group_labels = group_enc.transform(gen_age_train.group)

test_pred = (test1.merge(test2, left_index = True, right_index = True)
			.merge(test3, left_index = True, right_index = True)
			.merge(test4, left_index = True, right_index = True))
train_pred = (train1.merge(train2, left_index = True, right_index = True)
			 .merge(train3, left_index = True, right_index = True)
			 .merge(train4, left_index = True, right_index = True))

perceptron2 = LogisticRegression(dual = False, solver = 'lbfgs', multi_class = 'multinomial')
perceptron_params = {'C': [.001, 0.005, 0.01, 0.05]}
perceptron2 = GridSearchCV(perceptron2, perceptron_params, n_jobs = 3, cv = 5, error_score = 0)
perceptron2.fit(train_pred, group_labels)

joblib.dump(perceptron2.best_estimator_, '.pkl')
p = pd.DataFrame(perceptron2.predict_proba(test_pred), index = gen_age_test.index, columns = group_enc.classes_)
p.to_csv('pseudonet_prediction.csv', index = True)
