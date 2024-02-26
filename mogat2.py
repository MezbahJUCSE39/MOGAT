# Options
addRawFeat = True
base_path = ''
feature_networks_integration = [ 'exp','coe','cli','met','mut','cna','lnc', 'mir']
#feature_networks_integration = [ 'exp']
node_networks = [ 'exp','coe','cli','met','mut','cna', 'lnc', 'mir']
#node_networks = [ 'exp']
int_method = 'MLP' # 'MLP' or 'XGBoost' or 'RF' or 'SVM'
xtimes = 50 
xtimes2 = 10 

feature_selection_per_network = [False]*len(feature_networks_integration)
top_features_per_network = [50, 50, 50]
optional_feat_selection = False
boruta_runs = 100
boruta_top_features = 50

max_epochs = 500
min_epochs = 200
patience = 30
learning_rates = [0.01, 0.001, 0.0001]
#learning_rates = [0.0001]
# hid_sizes = [16, 32, 64, 128, 256, 512] 
hid_sizes = [512] 
random_state = 404

# MOGAT run
print('MOGAT is setting up!')
from lib import function
import time, io
import os, pyreadr, itertools
import pickle5 as pickle
from sklearn.metrics import f1_score, accuracy_score
import statistics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
import pandas as pd
import numpy as np
import os
import torch
import argparse
import errno
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class Data:
    def __init__(self, x=None, y=None):
        self.x=x
        self.y=y

    def cpu(self):
        self.x = self.x.cpu()
        self.y = self.y.cpu()

if ((True in feature_selection_per_network) or (optional_feat_selection == True)):
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    utils = importr('utils')
    rFerns = importr('rFerns')
    Boruta = importr('Boruta')
    pracma = importr('pracma')
    dplyr = importr('dplyr')
    import re

# Parser
parser = argparse.ArgumentParser(description='''An integrative node classification framework, called MOGAT 
(a cancer subtype prediction methodology), that utilizes graph attentions on multiple datatype-specific networks that are annotated with multiomics datasets as node features. 
This framework is model-agnostic and could be applied to any classification problem with properly processed datatypes and networks.
In our work, MOGAT was applied specifically to the breast cancer subtype prediction problem by applying attentions on patient similarity networks
constructed based on multiple biological datasets from breast tumor samples.''')
parser.add_argument('-data', "--data_location", nargs = 1, default = ['sample_data'])

args = parser.parse_args()
dataset_name = args.data_location[0]
print(dataset_name)

path = base_path + "data/" + dataset_name
if not os.path.exists(path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        
device = torch.device('cpu')
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#torch.cuda.set_device(6)


data_path_node =  base_path + 'data/' + dataset_name +'/'
run_name = 'MOGAT_'+  dataset_name + '_results_1'
save_path = base_path + run_name + '/'
excel_file = save_path + "MOGAT_results.xlsx"

if not os.path.exists(base_path + run_name):
    os.makedirs(base_path + run_name + '/')

file = base_path + 'data/' + dataset_name +'/labels.pkl'
print("Reading:", file)
with open(file, 'rb') as f:
    labels = pickle.load(f)

file = base_path + 'data/' + dataset_name + '/mask_values.pkl'
if os.path.exists(file):
    with open(file, 'rb') as f:
        train_valid_idx, test_idx = pickle.load(f)
else:
    train_valid_idx, test_idx= train_test_split(np.arange(len(labels)), test_size=0.20, shuffle=True, stratify=labels, random_state=random_state)

start = time.time()

is_first = 0

print('MOGAT is running..')

addFeatures = []
t = range(len(node_networks))
trial_combs = []
for r in range(1, len(t) + 1):
    trial_combs.extend([list(x) for x in itertools.combinations(t, r)])
new_trial_combs = []
for set1 in trial_combs:
    new_trial_combs.append(list(set1))
trial_combs = new_trial_combs

device = torch.device('cpu')

for trials in range(len(trial_combs)):
    node_networks2 = [node_networks[i] for i in trial_combs[trials]] # list(set(a) & set(feature_networks))
    netw_base = node_networks2[0]
    emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
    with open(emb_file, 'rb') as f:
        #emb = pickle.load(f)
        emb = CPU_Unpickler(f).load()
    emb = emb.cpu()
    if len(node_networks2) > 1:
        for netw_base in node_networks2[1:]:
            emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
            with open(emb_file, 'rb') as f:
                #emb = pickle.load(f)
                cur_emb = CPU_Unpickler(f).load()
                cur_emb = cur_emb.cpu()
            emb = torch.cat((emb, cur_emb), dim=1)
    emb = emb.cpu()        
    if addRawFeat == True:
        is_first = 0
        addFeatures = feature_networks_integration
        for netw in addFeatures:
            file = base_path + 'data/' + dataset_name +'/'+ netw +'.pkl'
            with open(file, 'rb') as f:
                feat = CPU_Unpickler(f).load()
            if is_first == 0:
                allx = torch.tensor(feat.values, device=device).float()
                is_first = 1
            else:
                allx = torch.cat((allx, torch.tensor(feat.values, device=device).float()), dim=1)   
        
        if optional_feat_selection == True:     
            allx_flat = [item for sublist in allx.tolist() for item in sublist]
            allx_temp = robjects.FloatVector(allx_flat)
            robjects.globalenv['allx_matrix'] = robjects.r('matrix')(allx_temp)
            robjects.globalenv['allx_x'] = robjects.IntVector(allx.shape)
            robjects.globalenv['labels_vector'] = robjects.IntVector(labels.tolist())
            robjects.globalenv['top'] = boruta_top_features
            robjects.globalenv['maxBorutaRuns'] = boruta_runs
            robjects.r('''
                require(rFerns)
                require(Boruta)
                labels_vector = as.factor(labels_vector)
                allx_matrix <- Reshape(allx_matrix, allx_x[1])
                allx_data = data.frame(allx_matrix)
                colnames(allx_data) <- 1:allx_x[2]
                allx_data <- allx_data %>%
                    mutate('Labels' = labels_vector)
                boruta.train <- Boruta(allx_data$Labels ~ ., data= allx_data, doTrace = 0, getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
                    ''')
            boruta_signif = robjects.globalenv['boruta_signif']
            robjects.r.rm("allx_matrix")
            robjects.r.rm("labels_vector")
            robjects.r.rm("allx_data")
            robjects.r.rm("boruta_signif")
            robjects.r.rm("thr")
            topx = []
            for index in boruta_signif:
                t_index=re.sub("`","",index)
                topx.append((np.array(allx).T)[int(t_index)-1])
            topx = np.array(topx)
            emb = torch.cat((emb, torch.tensor(topx.T, device=device)), dim=1)
            print('Top ' + str(boruta_top_features) + " features have been selected.")
        else:
            print(emb.get_device())
            print(allx.get_device())
            emb = torch.cat((emb, allx), dim=1)
    
    data = Data(x=emb, y=labels)
    
    data.cpu()
    train_mask = np.array([i in set(train_valid_idx) for i in range(data.x.shape[0])])
    data.train_mask = torch.tensor(train_mask, device=device)
    test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
    data.test_mask = torch.tensor(test_mask, device=device)
    X_train = pd.DataFrame(data.x[data.train_mask].numpy())
    X_test = pd.DataFrame(data.x[data.test_mask].numpy())
    y_train = pd.DataFrame(data.y[data.train_mask].numpy()).values.ravel()
    y_test = pd.DataFrame(data.y[data.test_mask].numpy()).values.ravel()
    print("Second Model Training Started")

    if int_method == 'MLP':
        params = {'hidden_layer_sizes': [ (64, 32)],
                  'learning_rate_init': [0.001],
                  'max_iter': [ 1500],
                  'n_iter_no_change': [100]}
        search = RandomizedSearchCV(estimator = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True), 
                                    return_train_score = True, scoring = 'f1_macro', 
                                    param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        search.fit(X_train, y_train)
        model = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True,
                              max_iter = search.best_params_['max_iter'], 
                              n_iter_no_change = search.best_params_['n_iter_no_change'],
                              hidden_layer_sizes = search.best_params_['hidden_layer_sizes'],
                              learning_rate_init = search.best_params_['learning_rate_init'])
        
    elif int_method == 'XGBoost':
        params = {'reg_alpha':range(0,10,1), 'reg_lambda':range(1,10,1) ,'max_depth': range(1,6,1), 
                  'min_child_weight': range(1,10,1), 'gamma': range(0,6,1),
                  'learning_rate':[0, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
                  'max_delta_step': range(0,10,1), 'colsample_bytree': [0.5, 0.7, 1.0],
                  'colsample_bylevel': [0.5, 0.7, 1.0], 'colsample_bynode': [0.5, 0.7, 1.0]}
        fit_params = {'early_stopping_rounds': 10,
                     'eval_metric': 'mlogloss',
                     'eval_set': [(X_train, y_train)]}
        
              
        search = RandomizedSearchCV(estimator = XGBClassifier(use_label_encoder=False, n_estimators = 1000, 
                                                                  fit_params = fit_params, objective="multi:softprob", eval_metric = "mlogloss", 
                                                                  verbosity = 0), return_train_score = True, scoring = 'f1_macro',
                                        param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        
        search.fit(X_train, y_train)
        
        model = XGBClassifier(use_label_encoder=False, objective="multi:softprob", eval_metric = "mlogloss", verbosity = 0,
                              n_estimators = 1000, fit_params = fit_params,
                              reg_alpha = search.best_params_['reg_alpha'],
                              reg_lambda = search.best_params_['reg_lambda'],
                              max_depth = search.best_params_['max_depth'],
                              min_child_weight = search.best_params_['min_child_weight'],
                              gamma = search.best_params_['gamma'],
                              learning_rate = search.best_params_['learning_rate'],
                              max_delta_step = search.best_params_['max_delta_step'],
                              colsample_bytree = search.best_params_['colsample_bytree'],
                              colsample_bylevel = search.best_params_['colsample_bylevel'],
                              colsample_bynode = search.best_params_['colsample_bynode'])
                            
    elif int_method == 'RF':
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                  'max_depth': max_depth,
                  'min_samples_split': [2, 5, 7, 10],
                  'min_samples_leaf': [1, 2, 5, 7, 10], 
                 'min_impurity_decrease':[0,0.5, 0.7, 1, 5, 10],
                 'max_leaf_nodes': [None, 5, 10, 20]}
        
        search = RandomizedSearchCV(estimator = RandomForestClassifier(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4,  n_iter = xtimes, verbose = 0)
        
        search.fit(X_train, y_train)
        model=RandomForestClassifier(n_estimators = search.best_params_['n_estimators'],
                                     max_depth = search.best_params_['max_depth'],
                                     min_samples_split = search.best_params_['min_samples_split'],
                                     min_samples_leaf = search.best_params_['min_samples_leaf'],
                                     min_impurity_decrease = search.best_params_['min_impurity_decrease'],
                                     max_leaf_nodes = search.best_params_['max_leaf_nodes'])

    elif int_method == 'SVM':
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
                  'kernel': ['linear', 'rbf']}
        
        search = RandomizedSearchCV(SVC(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4, n_iter = xtimes, verbose = 0)
        
        search.fit(X_train, y_train)
        model = SVC(kernel=search.best_params_['kernel'],
                  C = search.best_params_['C'],
                  gamma = search.best_params_['gamma'])

 
    av_result_acc = list()
    av_result_wf1 = list()
    av_result_mf1 = list()
    av_tr_result_acc = list()
    av_tr_result_wf1 = list()
    av_tr_result_mf1 = list()
 
        
    for ii in range(xtimes2):
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        y_pred = [round(value) for value in predictions]
        preds = model.predict(pd.DataFrame(data.x.numpy()))
        av_result_acc.append(round(accuracy_score(y_test, y_pred), 3))
        av_result_wf1.append(round(f1_score(y_test, y_pred, average='weighted'), 3))
        av_result_mf1.append(round(f1_score(y_test, y_pred, average='macro'), 3))
        tr_predictions = model.predict(X_train)
        tr_pred = [round(value) for value in tr_predictions]
        av_tr_result_acc.append(round(accuracy_score(y_train, tr_pred), 3))
        av_tr_result_wf1.append(round(f1_score(y_train, tr_pred, average='weighted'), 3))
        av_tr_result_mf1.append(round(f1_score(y_train, tr_pred, average='macro'), 3))
        
    if xtimes2 == 1:
        av_result_acc.append(round(accuracy_score(y_test, y_pred), 3))
        av_result_wf1.append(round(f1_score(y_test, y_pred, average='weighted'), 3))
        av_result_mf1.append(round(f1_score(y_test, y_pred, average='macro'), 3))
        av_tr_result_acc.append(round(accuracy_score(y_train, tr_pred), 3))
        av_tr_result_wf1.append(round(f1_score(y_train, tr_pred, average='weighted'), 3))
        av_tr_result_mf1.append(round(f1_score(y_train, tr_pred, average='macro'), 3))
        

    result_acc = str(round(statistics.median(av_result_acc), 3)) + '+-' + str(round(statistics.stdev(av_result_acc), 3))
    result_wf1 = str(round(statistics.median(av_result_wf1), 3)) + '+-' + str(round(statistics.stdev(av_result_wf1), 3))
    result_mf1 = str(round(statistics.median(av_result_mf1), 3)) + '+-' + str(round(statistics.stdev(av_result_mf1), 3))
    tr_result_acc = str(round(statistics.median(av_tr_result_acc), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_acc), 3))
    tr_result_wf1 = str(round(statistics.median(av_tr_result_wf1), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_wf1), 3))
    tr_result_mf1 = str(round(statistics.median(av_tr_result_mf1), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_mf1), 3))
    
    
    df = pd.DataFrame(columns=['Comb No', 'Used Embeddings', 'Added Raw Features', 'Selected Params', 'Train Acc', 'Train wF1','Train mF1', 'Test Acc', 'Test wF1','Test mF1'])
    x = [trials, node_networks2, addFeatures, search.best_params_, 
         tr_result_acc, tr_result_wf1, tr_result_mf1, result_acc, result_wf1, result_mf1]
    df = df.append(pd.Series(x, index=df.columns), ignore_index=True)
    
    print('Combination ' + str(trials) + ' ' + str(node_networks2) + ' >  selected parameters = ' + str(search.best_params_) + 
      ', train accuracy = ' + str(tr_result_acc) + ', train weighted-f1 = ' + str(tr_result_wf1) +
      ', train macro-f1 = ' +str(tr_result_mf1) + ', test accuracy = ' + str(result_acc) + 
      ', test weighted-f1 = ' + str(result_wf1) +', test macro-f1 = ' +str(result_mf1))

    # if trials == 0:
    #     if addRawFeat == True:
    #         function.append_df_to_excel(excel_file, df, sheet_name = int_method + '+Raw', index = False, header = True)
    #     else:
    #         function.append_df_to_excel(excel_file, df, sheet_name = int_method, index = False, header = True)
    # else:
    #     if addRawFeat == True:
    #         function.append_df_to_excel(excel_file, df, sheet_name = int_method + '+Raw', index = False, header = False)
    #     else:
    #         function.append_df_to_excel(excel_file, df, sheet_name = int_method, index = False, header = False)

end = time.time()
print('It took ' + str(round(end - start, 1)) + ' seconds for all runs.')
print('MOGAT is done.')
print('Results are available at ' + excel_file)
