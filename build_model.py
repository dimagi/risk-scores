
import pandas as pd 
import numpy as np
import pickle
import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def main():
    #load data
    print('Loading Data')
    data = pd.read_parquet(settings.DATA_PATH)
    
    #split data 
    print('Splitting Data')
    features = data.drop('mortality', axis=1)
    labels = data['mortality']
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state = 42, stratify = labels) #stratified
   
    #scale the data 
    print('Scaling Data')
    ss = StandardScaler()
    train_features_scaled = ss.fit_transform(train_features)
    test_features_scaled = ss.transform(test_features)
    train_labels = np.array(train_labels)
    # %%
    print('Creating param grid for fine-tuning')
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
    min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
    min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
    bootstrap = [True, False]
    rf_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

    #Create the model to be tuned
    rf_base = RandomForestClassifier()

    # Create the random search Random Forest
    rf_random = RandomizedSearchCV(estimator = rf_base, param_distributions = rf_grid, 
                                    n_iter = 10, cv = 3, verbose = 2, random_state = 42, scoring ='roc_auc',
                                    n_jobs = -1)

    # Fit the random search model
    print('Running random search cv to tune params')
    rf_random.fit(train_features_scaled, train_labels)
    #print(rf_random.cv_results_)

    # View the best parameters from the random search
    rf = rf_random.best_estimator_
    filename = settings.MODEL_PATH
    pickle.dump(rf, open(filename, 'wb'))
    print('Model saved at {}'.format(filename))


    y_true, y_pred = test_labels, rf.predict(test_features_scaled)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    f, ax = plt.subplots(figsize=(14,10))
    ax.plot(fpr, tpr, color='b',label=r'Test ROC',lw=2, alpha=.8)
    plt.show() #put better graph and save it 
    

if __name__ == '__main__':
    main()
