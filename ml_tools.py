import xgboost as xgb
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import pandas as pd
import pprint
#from sklearn.cross_validation import train_test_split as tts

def predict(X_train, y_train, X_test, params, iterations, y_test=None):
    assert X_train.shape[1] == X_test.shape[1]

    dtrain = xgb.DMatrix(X_train, label=np.array(y_train))
    if y_test is not None:
        dtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(dtrain, 'eval'), (dtest, 'train')]

        def eval_logloss(preds, dtrain):
            labels = dtrain.get_label()
            preds = preds.reshape(-1, 9)
            return 'log_loss', metrics.log_loss(labels, preds)

        bst = xgb.train(params, dtrain, iterations, watchlist,
                        feval=eval_logloss)
    else:
        dtest = xgb.DMatrix(X_test)
        bst = xgb.train(params, dtrain, iterations)

    pred = bst.predict(dtest)
    return pred.reshape(-1, 9)

def cross_valid(X, y, params, iterations, n_folds=6, silent=True):
    print 'Running cross validation'
    pprint.pprint(params)
    print 'Iterations:', iterations
    print 'X shape', X.shape

    y_size = len(y)
    if hasattr(X, 'values'):
        X = X.values
    y = np.array(y)

    kf = cross_validation.KFold(y_size, n_folds=n_folds, shuffle=True,
                                random_state=params['seed'])
    y_pred = np.zeros((y_size, 9))

    logs = []
    for train, test in kf:
        X_train, X_test = X[train, :], X[test, :]
        y_train, y_test = y[train], y[test]

        predictions = predict(X_train, y_train, X_test, params, iterations,
                              None if silent else y_test)
        y_pred[test] = predictions

        logs.append(metrics.log_loss(y_test, predictions))
        print 'Current log_loss:', logs[-1]

    print 'Final log_loss: %s (avg: %s, stddev: %s)' % (
                                                metrics.log_loss(y, y_pred),
                                                np.mean(logs),
                                                np.std(logs))

def feature_importance(X_train, y_train, params, iterations):
    dtrain = xgb.DMatrix(X_train, label=np.array(y_train))
    bst = xgb.train(params, dtrain, iterations)
    return bst.get_fscore()

def dump_submission(preds, indexes, fname='output.csv'):
    subm = pd.read_csv('sampleSubmission.csv')
    preds = pd.DataFrame(preds)
    preds.columns = subm.columns[1:]
    if hasattr(indexes, 'values'):
        indexes = indexes.values
    preds['Id'] = indexes
    preds.set_index('Id', inplace=True)

    preds.to_csv(fname)
    print 'Saved to', fname

def permuate(data):
    return data.reindex(np.random.permutation(data.index))
