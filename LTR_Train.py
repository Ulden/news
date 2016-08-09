import os

from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

import pyltr

from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from LTR_data_prepare import merge_dataset

textdir = './data_collecting/live_text/'
live_text_files = os.listdir(textdir)
dataset = merge_dataset(live_text_files)

randomset = dataset

y = randomset['f_score']
qids = randomset['file']
X = randomset.drop('f_score', axis = 1)

TX = X[:int(len(y) / 3)]
Ty = y[:int(len(y) / 3)]
Tqids = qids[:int(len(y) / 3)]

VX = X[int(len(y) / 3):int(len(y) / 3) * 2]
Vy = y[int(len(y) / 3):int(len(y) / 3) * 2]
Vqids = qids[int(len(y) / 3):int(len(y) / 3) * 2]

EX = X[int(len(y) / 3) * 2:len(y)]
Ey = y[int(len(y) / 3) * 2:len(y)]
Eqids = qids[int(len(y) / 3) * 2:len(y)]
'''
metric = pyltr.metrics.NDCG(k = 10)
pipeline_obj = Pipeline([('rf',RandomForestRegressor())])

parameters = {
    'rf__max_features': ['auto', 'sqrt', 'log2', 0.5, 0.75, 0.8],
    'rf__max_leaf_nodes': [2, 50, 100, 200, 500, 1000],
    'rf__min_samples_leaf': [2, 50, 100, 200, 500, 1000],
    'rf__min_samples_split': [2, 8, 10, 15, 20],
    'rf__n_estimators': [200, 1500, 2000],
    'rf__min_weight_fraction_leaf': [0, 0.25, 0.5],
    #'LambdaMART__metric': [metric],
    #'LambdaMART__n_estimators': [200, 500, 1000, 1500, 2000],
    #'LambdaMART__learning_rate': [0.1, 0.2, 0.4, 0.5, 0.8],
    #'LambdaMART__max_features': ['auto', 'sqrt', 'log2', 0.5, 0.75, 0.8],
    #'LambdaMART__min_samples_leaf': [2, 4, 10, 20, 50, 100, 200, 500, 1000],
    #'LambdaMART__max_leaf_nodes': [2, 4, 10, 20, 50, 100, 200, 500, 1000],
    #'LambdaMART__min_samples_split': [2, 4, 8, 10, 15, 20]
}
grid_search = GridSearchCV(pipeline_obj, parameters, n_jobs = -1,verbose= 1)
print('grid_search', '\n', grid_search, '\n')
grid_search.fit(X = TX,y = Ty)

best_parameters = dict(grid_search.best_estimator_.get_params())
for param_name in sorted(parameters.keys()):
    print("\t%s: %r\n" % (param_name, best_parameters[param_name]))


# RandomForest
rf = RandomForestRegressor(
        max_features = 'auto',
        min_samples_leaf = 64,
        max_leaf_nodes = 128,
        min_samples_split = 4,
        n_estimators=9000,
        verbose = 1,
        n_jobs = -1, )
rf.fit(TX, Ty)

print(cross_val_score(rf,EX,Ey))
print(cross_val_score(rf,VX,Vy))
'''

# LambdaMART
metric = pyltr.metrics.NDCG(k = 10)
monitor = pyltr.models.monitors.ValidationMonitor(VX, Vy, Vqids, metric = metric, stop_after = 300)
model = pyltr.models.LambdaMART(
        metric = metric,
        n_estimators = 9000,
        learning_rate = 0.1,
        max_features = 'auto',
        max_leaf_nodes = 10,
        min_samples_leaf = 64,
        verbose = 1,
        warm_start = True
)
model.fit(TX, Ty, Tqids,monitor = monitor)
Epred = model.predict(EX)
Vpred = model.predict(VX)

print('Our model:', metric.calc_mean(Eqids, Ey, Epred))
print('Our model:', metric.calc_mean(Vqids, Vy, Vpred))