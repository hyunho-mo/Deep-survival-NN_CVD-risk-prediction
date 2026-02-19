import os
import numpy as np
import pandas as pd
import time
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import viz
import matplotlib.pyplot as plt
from collections import defaultdict
from lifelines import CoxPHFitter
from lifelines import WeibullFitter
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
import logging
import uuid
import time
import utils
import random 
random.seed(0)
np.random.seed(0)

import scipy.stats as st
from scipy.special import logit, expit



localtime   = time.localtime()
TIMESTRING  = time.strftime("%m%d%Y%M", localtime)

current_dir = os.path.dirname(os.path.abspath(__file__))
tabular_dir = os.path.join(current_dir, 'Tabular')
figure_dir = os.path.join(current_dir, 'Figure_stats')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

cph_dir = os.path.join(current_dir, 'cph_results')
if not os.path.exists(cph_dir):
    os.makedirs(cph_dir)


category = "cacs0"
da_dir = os.path.join(current_dir, 'DAFT')
daft_dir = os.path.join(da_dir, 'daft')
calib_table_dir = os.path.join(daft_dir, 'calib_table')
category_dir = os.path.join(calib_table_dir, category)
if not os.path.exists(category_dir):
    os.makedirs(category_dir)

cox_dir = os.path.join(category_dir, "cox")
if not os.path.exists(cox_dir):
    os.makedirs(cox_dir)

# def h5_to_array(hdf5_group):

#     x_array = np.array(hdf5_group['x'])
#     t_array = np.array(hdf5_group['t'])
#     e_array = np.array(hdf5_group['e'])

#     return x_array, t_array, e_array


def format_dataset_to_df(dataset, duration_col, event_col, trt_idx = None):
    xdf = pd.DataFrame(dataset['x'])
    if trt_idx is not None:
        xdf = xdf.rename(columns={trt_idx : 'treat'})

    dt = pd.DataFrame(dataset['t'], columns=[duration_col])
    censor = pd.DataFrame(dataset['e'], columns=[event_col])
    cdf = pd.concat([xdf, dt, censor], axis=1)
    return cdf

# def format_datasets(train_df, val_df, test_df, duration_col, event_col):

#     train_t_array = train_df.pop(duration_col).to_frame().to_numpy()
#     train_e_array = train_df.pop(event_col).to_frame().to_numpy()
#     train_array = train_df.to_numpy()

#     val_t_array = val_df.pop(duration_col).to_frame().to_numpy()
#     val_e_array = val_df.pop(event_col).to_frame().to_numpy()
#     val_array = val_df.to_numpy()

#     test_t_array = test_df.pop(duration_col).to_frame().to_numpy()
#     test_e_array = test_df.pop(event_col).to_frame().to_numpy()
#     test_array = test_df.to_numpy()

#     train_dict = {"x":train_array , "t":train_t_array , "e":train_e_array }
#     val_dict = {"x":val_array , "t":val_t_array , "e":val_e_array }
#     test_dict = {"x":test_array , "t":test_t_array , "e":test_e_array }


#     return train_dict, val_dict, test_dict


def format_datasets(train_df, val_df, test_df, duration_col, event_col):

    train_t_array = train_df.pop(duration_col).to_frame().to_numpy()
    train_e_array = train_df.pop(event_col).to_frame().to_numpy()
    train_array = train_df.to_numpy()

    # val_t_array = val_df.pop(duration_col).to_frame().to_numpy()
    # val_e_array = val_df.pop(event_col).to_frame().to_numpy()
    # val_array = val_df.to_numpy()

    test_t_array = test_df.pop(duration_col).to_frame().to_numpy()
    test_e_array = test_df.pop(event_col).to_frame().to_numpy()
    test_array = test_df.to_numpy()

    train_dict = {"x":train_array , "t":train_t_array , "e":train_e_array }
    # val_dict = {"x":val_array , "t":val_t_array , "e":val_e_array }
    test_dict = {"x":test_array , "t":test_t_array , "e":test_e_array }


    return train_dict, train_dict, test_dict

def evaluate_model(model, dataset, bootstrap = False):
    def ci(model):
        def cph_ci(x, t, e, **kwargs):
            # print (x)
            return concordance_index(
                event_times= t, 
                predicted_scores= -model.predict_partial_hazard(x), 
                event_observed= e,
            )
        return cph_ci

    def mse(model):
        def cph_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(-model.predict_partial_hazard(x).values)
            return ((hr_pred - hr) ** 2).mean()
        return cph_mse  

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(ci(model), dataset)
    
    # Calcualte MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics


def evaluate_model_weibull(model, dataset, bootstrap = False):
    def ci(model):
        def cph_ci(x, t, e, **kwargs):
            # print (x)
            x_df = pd.DataFrame(x)
            output = model.predict_median(x_df)
            # print (x_df)
            return concordance_index(
                event_times= t, 
                predicted_scores= model.predict_median(x_df), 
                event_observed= e,
            )
        return cph_ci

    def mse(model):
        def cph_mse(x, hr, **kwargs):
            x_df = pd.DataFrame(x)
            hr_pred = np.squeeze(model.predict_median(x_df).values)
            return ((hr_pred - hr) ** 2).mean()
        return cph_mse  

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(ci(model), dataset)
    
    # Calcualte MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics



def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[WORC Warning] Cannot compute a confidence interval for a single iteration.')
        print('[WORC Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        # When we cannot compute a CI, just give the averages
        CI = (metric_average, metric_average)
    return CI


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment', default= None, help='name of the experiment that is being run', required = False)
    parser.add_argument('-cac', help="inclusion of cac variable", default=False, action="store_true")
    parser.add_argument('--imputation', help="different methods for data imputataion", default="mice", type=str)
    parser.add_argument('--outcome', help="outcome definition", default="CHD", type=str)
    # return parser.parse_args()

    args = parser.parse_args()
    # print("Arguments:",args)
    imputation = args.imputation
    outcome = args.outcome
    cac = args.cac

    cac_string = "TRF"
    ## Load Dataset
    timetoevent_csv_path =  os.path.join(tabular_dir,'RS_CT_subcohort_time2event_%s_%s_linked.csv' %(outcome, imputation))
    df = pd.read_csv(timetoevent_csv_path, sep = ",", na_values=' ')  





    # df = df.loc[df["sexe"] == 0] ## Men only
    # df = df.loc[df["sexe"] == 1] ## Women only
    ## Drop ergoid and rs_cohort
    df = df.drop(['ergoid', 'rs_cohort', 'patient_name'], axis=1)
    # df = df.drop(['ergoid', 'rs_cohort', 'sexe'], axis=1)

    if cac:
        # df = df.drop(['CACvol', 'AVC', 'AVCvol'], axis=1)
        df = df.drop(['Ln_CACvol'], axis=1)
        # df = df.drop(['Ln_CAC'], axis=1)
        # print ("check")
        cac_string = "CAC"
    else:
        df = df.drop(['Ln_CAC', 'Ln_CACvol'], axis=1)

    


    # # Only with log CAC
    # df = df[['Ln_CAC',  "lenfol", "fstat"]]

    # # Only with log CACvol
    # df = df[['Ln_CACvol',  "lenfol", "fstat"]]

    # # with log CACvol
    # df = df[['Ln_CACvol', 'Ln_CAC', "lenfol", "fstat"]]

    # # with age sex cac
    # df = df[['Ln_CAC', 'scanage', 'sexe', "lenfol", "fstat"]]
    

    DURATION_COL = "lenfol"
    EVENT_COL = "fstat"

    print ("vairables: ", df.columns)

    print ("Study population", len(df))
    print ("num of positive instance", df[EVENT_COL].values.ravel().sum())
    ## Startified train_test_split
    y = df.pop(EVENT_COL).to_frame()
    X = df

    # X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.2, random_state=1000)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,stratify=y_train, test_size=0.2, random_state=1000)
    
    
        ## Stratified k fold
    skf = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    # skf = StratifiedKFold(n_splits=5,random_state=None,shuffle=False)


    val_cindex_list = []
    test_cindex_list = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        print("len(train_index)", len(train_index))
        # print(f"  Test:  index={test_index}")
        print("len(test_index)", len(test_index))

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,stratify=y_train, test_size=0.2, random_state=0)

        # print (X_train)
        # print (X_val)
        # print (X_test)
        # print (len(X_train))
        # print (len(X_val))
        # print (len(X_test))


        train_df = X_train
        train_df[EVENT_COL] = y_train[EVENT_COL]
        val_df = X_val
        val_df[EVENT_COL] = y_val[EVENT_COL]
        test_df = X_test
        test_df[EVENT_COL] = y_test[EVENT_COL]


        train_dict, val_dict, test_dict  = format_datasets(train_df, val_df, test_df, DURATION_COL, EVENT_COL)
        # train_dict, val_dict, test_dict  = format_datasets(train_df, train_df, test_df, DURATION_COL, EVENT_COL)

        train_df = format_dataset_to_df(train_dict, DURATION_COL, EVENT_COL)

        cf = CoxPHFitter(penalizer = 0.01)
        # cf = CoxPHFitter()
        results = cf.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)
        # print ("cf.concordance_index_", cf.concordance_index_)
        cf.print_summary()
        print("Train Likelihood: " + str(cf.log_likelihood_))

        metrics = evaluate_model(cf, val_dict)
        print("Valid metrics: " + str(metrics))
        val_cindex = metrics['c_index']

        metrics = evaluate_model(cf, test_dict, bootstrap=False)
        test_cindex = metrics['c_index']
        print("Test metrics no bootstrap: " + str(metrics))

        val_cindex_list.append(val_cindex)
        test_cindex_list.append(test_cindex)

        min_time = min(test_dict["t"].flatten())
        max_time = max(test_dict["t"].flatten())
        times = np.arange(min_time, max_time, 365)
        # predicted_scores= -cf.predict_partial_hazard(test_dict['x'])
        predicted_scores= cf.predict_partial_hazard(test_dict['x'])


        train_y_df = pd.DataFrame([])
        train_y_df[EVENT_COL] = train_dict["e"].flatten().astype(bool)
        train_y_df[DURATION_COL] = train_dict["t"].flatten()
        print (train_y_df)


        # train_y_df  = train_y_df.loc[(train_y_df[DURATION_COL]<= 4000) & (train_y_df[DURATION_COL]>= 55)]

        y_train_array = train_y_df.to_records(index=False)

        test_y_df = pd.DataFrame([])
        test_y_df[EVENT_COL] = test_dict["e"].flatten().astype(bool)
        test_y_df[DURATION_COL] = test_dict["t"].flatten()

        # test_y_df  = test_y_df.loc[(test_y_df[DURATION_COL]<= 4010) & (test_y_df[DURATION_COL]>= 50)]
        y_test_array = test_y_df.to_records(index=False)

        # predicted_scores.to_csv(os.path.join(cph_dir,"predicted_%s.csv" %i), index=False)
        # train_y_df.to_csv(os.path.join(cph_dir,"train_y_%s.csv" %i), index=False)
        # test_y_df.to_csv(os.path.join(cph_dir,"test_y_%s.csv" %i), index=False)


        partial_hazard_list_normalized = (predicted_scores - predicted_scores.min()) / (predicted_scores.max() - predicted_scores.min())

        calib_df = pd.DataFrame([])
        calib_df["event"] = test_dict["e"].flatten().astype(bool)
        calib_df["time"] = test_dict["t"].flatten()
        calib_df["output"] = predicted_scores
        calib_df["normalized_output"] = partial_hazard_list_normalized
        calib_df["time"].loc[(calib_df["time"]>=3650.0)] = 3650.0
        calib_table_filename = "cox_calib_%s_%s.csv" %(category,i)
        calib_table_filepath = os.path.join(cox_dir, calib_table_filename)
        calib_df.to_csv(calib_table_filepath)


        print (y_train_array)

        cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train_array, y_test_array, predicted_scores, times)
        fig = plt.figure()
        plt.plot(times, cph_auc, marker="o")
        plt.axhline(cph_mean_auc, linestyle="--")
        plt.xlabel("days from enrollment")
        plt.ylabel("time-dependent AUC")
        plt.grid(True)
        plt.savefig(os.path.join(figure_dir, 'TDAUC_CPH_%s_%s_%s.png' %(outcome, cac_string, i)))



    skf = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i+5}:")
        # print(f"  Train: index={train_index}")
        print("len(train_index)", len(train_index))
        # print(f"  Test:  index={test_index}")
        print("len(test_index)", len(test_index))

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,stratify=y_train, test_size=0.2, random_state=0)

        # print (X_train)
        # print (X_val)
        # print (X_test)
        # print (len(X_train))
        # print (len(X_val))
        # print (len(X_test))


        train_df = X_train
        train_df[EVENT_COL] = y_train[EVENT_COL]
        val_df = X_val
        val_df[EVENT_COL] = y_val[EVENT_COL]
        test_df = X_test
        test_df[EVENT_COL] = y_test[EVENT_COL]


        train_dict, val_dict, test_dict  = format_datasets(train_df, val_df, test_df, DURATION_COL, EVENT_COL)
        # train_dict, val_dict, test_dict  = format_datasets(train_df, train_df, test_df, DURATION_COL, EVENT_COL)

        train_df = format_dataset_to_df(train_dict, DURATION_COL, EVENT_COL)

        cf = CoxPHFitter(penalizer = 0.01)
        # cf = CoxPHFitter()
        results = cf.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)
        # print ("cf.concordance_index_", cf.concordance_index_)
        cf.print_summary()
        print("Train Likelihood: " + str(cf.log_likelihood_))

        metrics = evaluate_model(cf, val_dict)
        print("Valid metrics: " + str(metrics))
        val_cindex = metrics['c_index']

        metrics = evaluate_model(cf, test_dict, bootstrap=False)
        test_cindex = metrics['c_index']
        print("Test metrics no bootstrap: " + str(metrics))

        val_cindex_list.append(val_cindex)
        test_cindex_list.append(test_cindex)

        min_time = min(test_dict["t"].flatten())
        max_time = max(test_dict["t"].flatten())
        times = np.arange(min_time, max_time, 365)
        # predicted_scores= -cf.predict_partial_hazard(test_dict['x'])
        predicted_scores= cf.predict_partial_hazard(test_dict['x'])


        train_y_df = pd.DataFrame([])
        train_y_df[EVENT_COL] = train_dict["e"].flatten().astype(bool)
        train_y_df[DURATION_COL] = train_dict["t"].flatten()
        print (train_y_df)


        # train_y_df  = train_y_df.loc[(train_y_df[DURATION_COL]<= 4000) & (train_y_df[DURATION_COL]>= 55)]

        y_train_array = train_y_df.to_records(index=False)

        test_y_df = pd.DataFrame([])
        test_y_df[EVENT_COL] = test_dict["e"].flatten().astype(bool)
        test_y_df[DURATION_COL] = test_dict["t"].flatten()

        # test_y_df  = test_y_df.loc[(test_y_df[DURATION_COL]<= 4010) & (test_y_df[DURATION_COL]>= 50)]
        y_test_array = test_y_df.to_records(index=False)

        # predicted_scores.to_csv(os.path.join(cph_dir,"predicted_%s.csv" %i), index=False)
        # train_y_df.to_csv(os.path.join(cph_dir,"train_y_%s.csv" %i), index=False)
        # test_y_df.to_csv(os.path.join(cph_dir,"test_y_%s.csv" %i), index=False)


        partial_hazard_list_normalized = (predicted_scores - predicted_scores.min()) / (predicted_scores.max() - predicted_scores.min())

        calib_df = pd.DataFrame([])
        calib_df["event"] = test_dict["e"].flatten().astype(bool)
        calib_df["time"] = test_dict["t"].flatten()
        calib_df["output"] = predicted_scores
        calib_df["normalized_output"] = partial_hazard_list_normalized
        calib_df["time"].loc[(calib_df["time"]>=3650.0)] = 3650.0
        calib_table_filename = "cox_calib_%s_%s.csv" %(category,i+5)
        calib_table_filepath = os.path.join(cox_dir, calib_table_filename)
        calib_df.to_csv(calib_table_filepath)




        print (y_train_array)

        cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train_array, y_test_array, predicted_scores, times)
        fig = plt.figure()
        plt.plot(times, cph_auc, marker="o")
        plt.axhline(cph_mean_auc, linestyle="--")
        plt.xlabel("days from enrollment")
        plt.ylabel("time-dependent AUC")
        plt.grid(True)
        plt.savefig(os.path.join(figure_dir, 'TDAUC_CPH_%s_%s_%s.png' %(outcome, cac_string, i)))









    print ("val_cindex_list", val_cindex_list)
    print ("test_cindex_list", test_cindex_list)

    val_mean = np.mean(val_cindex_list)
    val_std = np.std(val_cindex_list)
    test_mean = np.mean(test_cindex_list)
    test_std = np.std(test_cindex_list)


    print ("val_mean +- std %s +- %s" %(round(val_mean, 4), round(val_std, 4)))
    print ("test_mean +- std %s +- %s" %(round(test_mean, 4), round(test_std, 4)))



    ci = compute_confidence(np.array(test_cindex_list), len(train_index), len(test_index), alpha=0.95)
    print ("ci", ci)

    metrics = test_cindex_list
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics)-1, loc=mean, scale=st.sem(metrics))

    print ('mean', mean)
    print ('confidence_interval', conf_interval)




if __name__ == '__main__':
    main()    

