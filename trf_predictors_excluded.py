import os
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
import argparse
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from datetime import datetime
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from functools import reduce
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from impyute.imputation.cs import mean
from impyute.imputation.cs import median
from impyute.imputation.cs import mode
from impyute.imputation.cs import em
from impyute.imputation.cs import fast_knn
from impyute.imputation.cs import random as random_imp
# from missingpy import KNNImputer
from fancyimpute import SimpleFill, KNN, NuclearNormMinimization, SoftImpute, BiScaler, IterativeSVD, MatrixFactorization
# from em_impute import impute_em



import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
from torchvision import datasets
from torchvision import transforms


from tqdm.notebook import tqdm_notebook as tqdm
import torch.nn.functional as F

from utils import SampleDataset, AE1, AE2, autoencoder_train, autoencoder_predict, get_imputation

from torch import nn
from tqdm import tqdm

from utils import binary_sampler, uniform_sampler, sample_batch_index, hint_for_mar
from utils import normalization, renormalization, rounding

from gain_rs import gain

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# import utils 

# disable chained assignments
pd.options.mode.chained_assignment = None 


## Access to the server and the ESS data folder
current_dir = os.path.dirname(os.path.abspath(__file__))
storage_dir = "/data/scratch/hmo"
segmented_img_dir = os.path.join(storage_dir, 'ergo_heart')

tabular_dir = os.path.join(current_dir, 'Tabular')
if not os.path.exists(tabular_dir):
    os.makedirs(tabular_dir)



figure_dir = os.path.join(current_dir, 'Figure')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)




############### EGAIN ######################
    
def d_loss(M, D_prob):
    return -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob + 1e-8))


def g_loss(M, D_prob, alpha, X, G_sample):
    return -torch.mean((1 - M) * torch.log(D_prob + 1e-8)) + \
           alpha * torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M)


class Generator(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super(Generator, self).__init__()
        self.d_w1 = nn.Linear(dim * 2, h_dim)
        # nn.init.xavier_normal_(self.d_w1.weight)
        self.d_w2 = nn.Linear(h_dim, h_dim)
        # nn.init.xavier_normal_(self.d_w2.weight)
        self.d_w3 = nn.Linear(h_dim, dim)
        # nn.init.xavier_normal_(self.d_w3.weight)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], 1)
        x = self.relu(self.d_w1(x))
        # x += torch.normal(0.0, 0.01, size=x.shape)
        x = self.relu(self.d_w2(x))
        # x += torch.normal(0.0, 0.01, size=x.shape)
        x = self.sigmoid(self.d_w3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super(Discriminator, self).__init__()
        self.d_w1 = nn.Linear(dim * 2, h_dim)
        # nn.init.xavier_normal_(self.d_w1.weight)
        self.d_w2 = nn.Linear(h_dim, h_dim)
        # nn.init.xavier_normal_(self.d_w2.weight)
        self.d_w3 = nn.Linear(h_dim, dim)
        # nn.init.xavier_normal_(self.d_w3.weight)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        x = torch.cat([x, h], 1)
        x = self.relu(self.d_w1(x))
        x = self.relu(self.d_w2(x))
        x = self.sigmoid(self.d_w3(x))
        return x



############################################



def mort_col(df, mort_colname, value):
    df[mort_colname] = df["RS_CVDmort2011"]
    df[mort_colname].loc[(df[mort_colname] == value)] = 1
    df[mort_colname].loc[df[mort_colname] != 1] = 0
    return df

def interv_col (df, fstat_col, inc_col):
    df[fstat_col] = df[inc_col]
    df[fstat_col].loc[(df[fstat_col] == 1)|(df[fstat_col] == 8)] = 1
    df[fstat_col].loc[df[fstat_col] != 1] = 0
    return df 

def date_col (df, inc_date_col, endat_col, fstat_col, max_follow):
    df[inc_date_col] = df[endat_col].loc[df[fstat_col] == 1]
    df=df.fillna({inc_date_col:max_follow})
    df[inc_date_col] = pd.to_datetime(df[inc_date_col]).dt.normalize()
    return df

def draw_km_curve(lenfol_col, fstat_col, argument):
    kmf = KaplanMeierFitter()
    kmf.fit(lenfol_col, fstat_col)
    
    kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_
    ts = ci.index
    low, high = np.transpose(ci.values)
    fig = plt.figure()

    plt.fill_between(ts, low, high, color='gray', alpha=0.3)
    kmf.survival_function_.plot(ax=plt.gca())
    plt.ylabel('%s survival function (KM curve)' %argument)
    plt.savefig(os.path.join(figure_dir, 'km_curve_%s.png' %argument))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-trf', help="Traditional risk factors (TRF)", default=False, action="store_true")
    parser.add_argument('-ext_trf', help="Extended TRF", default=False, action="store_true")
    parser.add_argument('-ext_lab', help="Exteded lab variables", default=False, action="store_true")
    parser.add_argument('--outcome', help="outcome definition", default="CHD", type=str)
    parser.add_argument('--imputation', help="different methods for data imputataion", default="mice", type=str)
    parser.add_argument('--category', help="category", default="cacs0", type=str)

    # Predictors:
    # Traditional risk factors (TRF): Age, sex, prev_DM, systolic BP, non-HDL cholesterol, current smoking
    # Extended TRF: TRF, BMI, famHxMI, BP-lowering med, statin, aspirin, past smoking
    # Exteded lab: Ext TRF, proBNP, hsTnT, CRP, eGFR, triglycerides, LDL (Martin-Hopkins; i/o nonHDL)

    args = parser.parse_args()
    trf = args.trf
    ext_trf = args.ext_trf
    ext_lab = args.ext_lab

    outcome = args.outcome

    imputation = args.imputation

    category = args.category

    # template_csv_path =  os.path.join(current_dir, 'MyDigiTwin_RS_template.csv')
    study_popluation_csv_path =  os.path.join(tabular_dir, 'RS_CT_subcohort_%s_outcomes_linked_%s_excluded.csv' %(outcome,category))
    # processed_csv_path =  os.path.join(current_dir, 'RS_time2event.csv')


    if ext_lab: ## Total CHD includes hard CHD
        ext_trf = True

    if ext_trf:   ## Extended ascvd includes ascvd
        trf = True

    ## Load study population csv file to dataframe
    df = pd.read_csv(study_popluation_csv_path, sep = ",", na_values=' ')    
    print ("number of participants: ", len(df))

    df = df.loc[(df["CAC_AVC_available"]==1) | (df["CAC_AVC_available"]==2)]

    ## Add log of CACvol
    df["Ln_CACvol"] = np.log(df["CACvol"].values + 1) 

    ## Add past_smoking based on the value of 'smoking' variable
    df["past_smoking"] = df["smoking"]
    df["past_smoking"].loc[(df["past_smoking"] == 1)] = 1 ## Set "past" of smoking variable as '1' of 'past smoking' variable
    df["past_smoking"].loc[df["past_smoking"] != 1] = 0 

    ######################## Predictors ###########################
    if trf: # Traditional risk factors (TRF)
        argument = "TRF"
        # Traditional risk factors (TRF): Age, sex, prev_DM, systolic BP, non-HDL cholesterol, current smoking
        trf_cols = ['scanage', 'sexe', 'prev_DM', "sbp", "nonHDL", "curr_smoking"]

        # for col in trf_cols:
        #     # print (col)
        #     if col == 'prev_DM':
        #         df=df.fillna({col:0})

            # print (df[col].isnull().values.ravel().sum()/len(df)*100)
        predictor_cols = trf_cols

    if ext_trf: # Traditional risk factors (TRF)
        argument = "extendedTRF"
        # Extended TRF: TRF, BMI, famHxMI, BP-lowering med, statin, aspirin, past smoking
        # ext_trf_cols = trf_cols + ["BMI", "famHxMI65", "blpdrug", "aspirin", "past_smoking"]
        # ext_trf_cols = trf_cols + ["BMI", "famHxMI65", "blpdrug", "aspirin", "past_smoking"]
        ext_trf_cols = trf_cols + ["BMI", "famHxMI65", "blpdrug", "past_smoking", "statin"]
        predictor_cols = ext_trf_cols


    if ext_lab: # Traditional risk factors (TRF)
        argument = "extededLAB"
        # Exteded lab: Ext TRF, proBNP, hsTnT, CRP, eGFR, triglycerides, LDL (Martin-Hopkins; i/o nonHDL)
        # ext_lab_cols = ext_trf_cols + ["hsTnT", "crp_mg", "tg_mmol", "LDL_Martin_mmol"]
        ext_lab_cols = ext_trf_cols + ["CRP", "GFR"]
        predictor_cols = ext_lab_cols


    ## Add calcification score columns
    predictor_cols = predictor_cols + ["Ln_CAC", "Ln_CACvol"]

    ## Save time-to-event data


    # ## Exclude participants who might never visit the research center (decision rule for those participants: missing more than five test variables)
    # non_test_cols = ['age', 'sexe', 'prev_DM', "smoking"]
    # test_predictor_cols = list(set(predictor_cols) - set(non_test_cols))
    # df["test_nan"] = df[test_predictor_cols].isnull().sum(axis=1)
    # df = df.drop(df[(df["test_nan"]>=5)].index)
    # df = df.drop('test_nan', axis=1)



    for col in predictor_cols:
        print (col)
        print (df[col].isnull().values.ravel().sum(), "/", len(df))
        print (df[col].isnull().values.ravel().sum()/len(df)*100)


    ## Define the needed cols for time to event
    # col_id = df.columns[:2].values.tolist()
    col_id = ['ergoid', 'rs_cohort', 'patient_name']
    print ("col_id", col_id)
    col_tte = col_id + predictor_cols + ["fstat","lenfol"]
    print ("col_tte", col_tte)
    df_tte = df[col_tte]

    


    df_for_imputation = df_tte.drop(columns=['ergoid', 'rs_cohort', 'patient_name', "fstat", "lenfol"])

    if imputation == "mice":
        ##### MICE Imputation
        ## Define imputer
        imputer = IterativeImputer(random_state=0, max_iter=5)
        # Instead of discarding, use MICE imputation for "ldlchol_result_all_m_1"
        
        imputer.fit(df_for_imputation)
        imputed_array = imputer.transform(df_for_imputation)
        print ("imputed_array.shape", imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)


    elif imputation == "missforest":
        imputer = MissForest()
        imputer.fit(df_for_imputation)
        imputed_array = imputer.transform(df_for_imputation)
        print ("imputed_array.shape", imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)       

    elif imputation == "knn":
        imputer = KNN(k=10)    
        imputed_array = imputer.fit_transform(df_for_imputation)
        print ("imputed_array.shape", imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)      

    elif imputation == "simple":
        imputer = SimpleFill()
        imputed_array = imputer.fit_transform(df_for_imputation)
        print ("imputed_array.shape", imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   


    # matrix completion using convex optimization to find low-rank solution
    # that still matches observed values. Slow!
    elif imputation == "nnm":
        imputer = NuclearNormMinimization()
        imputed_array = imputer.fit_transform(df_for_imputation)
        print ("imputed_array.shape", imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   


    # Instead of solving the nuclear norm objective directly, instead
    # induce sparsity using singular value thresholding
    elif imputation == "softimpute":
        imputer = BiScaler()
        input_array = df_for_imputation.to_numpy()
        imputed_array_normalized = imputer.fit_transform(input_array)
        imputer = SoftImpute()
        imputed_array = imputer.fit_transform(imputed_array_normalized)
        print ("imputed_array.shape", imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   

    # Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from Missing value estimation methods for DNA microarrays by Troyanskaya et. al.
    elif imputation == "iter_svd":
        imputer = IterativeSVD(rank=10)
        imputed_array = imputer.fit_transform(df_for_imputation)
        print ("imputed_array.shape", imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   

    # MatrixFactorization: Direct factorization of the incomplete matrix into low-rank U and V, with per-row and per-column biases, as well as a global bias. Solved by SGD in pure numpy.
    elif imputation == "mat_fact":
        imputer = MatrixFactorization()
        input_array = df_for_imputation.to_numpy()
        imputed_array = imputer.fit_transform(input_array)
        print ("imputed_array.shape", imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   

    # https://joon3216.github.io/research_materials/2019/em_imputation_python.html
    # https://joon3216.github.io/research_materials/2019/em_imputation.html
    elif imputation == "em_impute":
        input_array = df_for_imputation.to_numpy()
        imputed_array = impute_em(input_array)
        print ("imputed_array.shape", imputed_array['X_imputed'].shape)
        df_imputed = pd.DataFrame(imputed_array['X_imputed'], columns = df_for_imputation.columns)   

    elif imputation == "mean":
        input_array = df_for_imputation.to_numpy()
        imputed_array = mean(input_array)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   

    elif imputation == "median":
        input_array = df_for_imputation.to_numpy()
        imputed_array = median(input_array)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   

    elif imputation == "mode":
        input_array = df_for_imputation.to_numpy()
        imputed_array = mode(input_array)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)  

    elif imputation == "random":
        input_array = df_for_imputation.to_numpy()
        imputed_array = random_imp(input_array)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)  

    elif imputation == "em":
        input_array = df_for_imputation.to_numpy()
        imputed_array = em(input_array)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)  

    elif imputation == "fast_knn":
        input_array = df_for_imputation.to_numpy()
        imputed_array = fast_knn(input_array)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)      

    # https://datascience.oneoffcoder.com/autoencoder-data-imputation.html
    elif imputation == "autoencoder":
        
        N_df = df_tte.dropna() # N: represents data that is not missing (will be used for training)
        M_df = df_tte[df_for_imputation.isnull().any(axis=1)] # M: represents data that is missing (will be used for testing)

        NM_df = pd.concat([N_df, M_df])
        N_df = N_df.drop(columns=['ergoid', 'rs_cohort', 'patient_name', "fstat", "lenfol"])
        M_df = M_df.drop(columns=['ergoid', 'rs_cohort', 'patient_name', "fstat", "lenfol"])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)

        norm_n, norm_parameters_n = normalization(N_df.values)
        norm_m, norm_parameters_m = normalization(M_df.fillna(0.0).values)
        norm_m_raw, norm_parameters_m_raw = normalization(M_df.values)

        N_ds = SampleDataset(X=norm_n, device=device)
        M_ds = SampleDataset(X=norm_m, device=device)

        N_dl = DataLoader(N_ds, batch_size=64, shuffle=False, num_workers=1)
        M_dl = DataLoader(M_ds, batch_size=64, shuffle=False, num_workers=1)

        model_1 = AE1(input_size=N_df.shape[1]).double().to(device)
        opt_1 = torch.optim.Adam(model_1.parameters(), lr=1e-3, weight_decay=1e-8)
        loss_1 = autoencoder_train(model_1, opt_1, N_dl, device)


        # model_2 = AE2(dim=N_df.shape[1]).double().to(device)
        # opt_2 = torch.optim.SGD(model_2.parameters(), momentum=0.99, lr=0.01, nesterov=True)
        # loss_2 = autoencoder_train(model_2, opt_2, N_dl, device)

        
        plt.style.use('fivethirtyeight')
        fig = plt.figure()
        ax = loss_1['loss'].plot(kind='line', figsize=(15, 4), title='MSE Loss', ylabel='MSE', label='AE1')
        _ = ax.set_xticks(list(range(1, 21, 1)))
        _ = ax.legend()        
        plt.savefig(os.path.join(figure_dir, 'autoencoder_training.png'))

        # plt.style.use('fivethirtyeight')
        # fig = plt.figure()
        # ax = loss_2['loss'].plot(kind='line', figsize=(15, 4), title='MSE Loss', ylabel='MSE', label='AE2')
        # _ = ax.set_xticks(list(range(1, 21, 1)))
        # _ = ax.legend()        
        # plt.savefig(os.path.join(figure_dir, 'autoencoder_training.png'))

        M_pred = np.vstack([autoencoder_predict(model_1, items, device) for items, _ in M_dl])


        ## get imputation
        # M_imputed = np.array([get_imputation(M_df.values[r,:], M_pred[r,:]) for r in range(M_df.shape[0])])
        M_imputed = np.array([get_imputation(norm_m_raw[r,:], M_pred[r,:]) for r in range(norm_m.shape[0])])
       
        # M_imputed_df = pd.DataFrame(M_imputed, columns = M_df.columns)   
        # M_imputed_df.to_csv(os.path.join(current_dir, 'RS_time2event_%s_m_imputed.csv' %imputation), sep = ",", index = False)

        not_missing_array = renormalization(norm_n, norm_parameters_n)
        M_imputed = renormalization(M_imputed, norm_parameters_m_raw)

        # not_missing_array = N_df.to_numpy()
        # missing_array = M_df.to_numpy()

        imputed_array = np.concatenate((not_missing_array, M_imputed), axis=0)


        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   


        ori_array = np.concatenate((not_missing_array, M_df.fillna(0.0).values), axis=0)
        # df_ori = pd.DataFrame(ori_array, columns = df_for_imputation.columns)   


        # df_ori.insert(loc=0, column='ergoid', value=df_tte['ergoid'].values)
        # df_ori.insert(loc=1, column='rs_cohort', value=df_tte['rs_cohort'].values)
        # df_ori.insert(loc=len(df_ori.columns.values.tolist())-1, column='lenfol', value=df_tte['lenfol'].values)

        # df_ori.to_csv(os.path.join(current_dir, 'RS_time2event_%s_ori.csv' %imputation), sep = ",", index = False)


    # https://github.com/philipperemy/EGAIN-pytorch/tree/master
    elif imputation == "gain":

        ## input argument
        batch_size = 128
        # batch_size = 64
        hint_rate = 0.9
        # hint_rate = 0.5
        alpha = 100
        # iterations = 700
        iterations = 600
        # iterations = 30
        mechanism = False

        gain_parameters = {'batch_size': batch_size, 'hint_rate': hint_rate, 'alpha': alpha, 'iterations': iterations}

        input_array = df_for_imputation.to_numpy()

        ## Load data
        ## miss_data_x: data with missing values
        ## data_m: indicator matrix for missing components
        miss_data_x = input_array

        ## Impute missing data based in gain function in gain_rs
        # imputed_array, d_loss_list, g_loss_list, concordance_list = gain(miss_data_x, gain_parameters, mechanism, df_for_imputation.columns, df_tte["fstat"].values, df_tte["lenfol"].values)
        imputed_array, d_loss_list, g_loss_list = gain(miss_data_x, gain_parameters, mechanism, df_for_imputation.columns, df_tte["fstat"].values, df_tte["lenfol"].values)

        fig = plt.figure()
        plt.plot(d_loss_list)
        plt.xlabel("epochs", fontsize=16)
        plt.title("GAIN training d_loss", fontsize=16)
        plt.savefig(os.path.join(figure_dir, 'gain_train_d_loss.png'))

        fig = plt.figure()
        plt.plot(g_loss_list)
        plt.xlabel("epochs", fontsize=16)
        plt.title("GAIN training g_loss", fontsize=16)
        plt.savefig(os.path.join(figure_dir, 'gain_train_g_loss.png'))
        
        # print ("concordance_list", concordance_list)
        # fig = plt.figure()
        # plt.plot(concordance_list)
        # plt.xlabel("epochs", fontsize=16)
        # plt.title("GAIN training c-index", fontsize=16)
        # plt.savefig(os.path.join(figure_dir, 'gain_train_c_index.png'))

        print (imputed_array)
        print (imputed_array.shape)
        df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)   


    ###########################


    if imputation == "autoencoder":
        df_imputed.insert(loc=0, column='ergoid', value=NM_df['ergoid'].values)
        df_imputed.insert(loc=1, column='rs_cohort', value=NM_df['rs_cohort'].values)
        df_imputed.insert(loc=2, column='patient_name', value=NM_df['patient_name'].values)


        df_imputed["fstat"] = NM_df['fstat'].values
        df_imputed.insert(loc=len(df_imputed.columns.values.tolist())-1, column='lenfol', value=NM_df['lenfol'].values)
        df_imputed = df_imputed.sort_values('ergoid')
    else:
        df_imputed.insert(loc=0, column='ergoid', value=df_tte['ergoid'].values)
        df_imputed.insert(loc=1, column='rs_cohort', value=df_tte['rs_cohort'].values)
        df_imputed.insert(loc=2, column='patient_name', value=df_tte['patient_name'].values)

        df_imputed["fstat"] = df_tte['fstat'].values
        df_imputed.insert(loc=len(df_imputed.columns.values.tolist())-1, column='lenfol', value=df_tte['lenfol'].values)



        
    # print ("len(df_imputed)", len(df_imputed))    
    df_imputed = df_imputed.dropna(subset=['patient_name'])    
    print ("total num participants after the imputation", len(df_imputed)) 

    ## Exclude rows does not have segmented images
    id_segmented = os.listdir(segmented_img_dir) 
    # print (df_imputed['patient_name'])
    # print (id_segmented)
    df_imputed['patient_name'] = df_imputed['patient_name'].astype(int).astype(str)
    df_imputed = df_imputed[df_imputed['patient_name'].isin(id_segmented)]

    print ("total number of participants after excluding heart segmentation failure", len(df_imputed))

    df_imputed.to_csv(os.path.join(tabular_dir, 'RS_CT_subcohort_time2event_%s_%s_linked_%s_excluded.csv' %(outcome, imputation, category)), sep = ",", index = False)



if __name__ == '__main__':
    main()    




