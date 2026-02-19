
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import h5py
import argparse
from pathlib import Path

def generate_datafiles(hdf5_filename, x_df, y_df, len_df, calc_dir, fold ):
    hf = h5py.File(hdf5_filename, 'w')

    for index, id in enumerate(x_df["patient_name"].values):
        print ("id", id)


        img_filepath = os.path.join(calc_dir, "pericaridum_%s.npy" %id)
        img_filecheck = Path(img_filepath)
        if img_filecheck.is_file():

            img_array = np.load(img_filepath)

            table_array = x_df.loc[(x_df["patient_name"] == id)].drop(['ergoid', 'rs_cohort', 'patient_name'], axis=1).values
            table_array = table_array.ravel()

            lenfol = len_df.values[index][0]
            fstat = y_df.values[index][0]



            g1 = hf.create_group('%s' %id)
            g1.attrs['RID'] = id
            g1.attrs['VISCODE'] = "bl"
            g1.attrs['DX'] = "CN"
            g1.attrs['time'] = lenfol
            if fstat == 0:
                g1.attrs['event'] = "no"
            elif fstat == 1:
                g1.attrs['event'] = "yes"

            dset2 = g1.create_dataset("tabular", data = table_array)
            # print ("dset2.name", dset2.name)
            # dset3 = g1.create_dataset('Left-Hippocampus/vol_with_bg', data = img_array, shape = img_array.shape)
            
            
            g_img = g1.create_group('Left-Hippocampus')
            dset3 = g_img.create_dataset('vol_with_bg', data = img_array)
            # dset3 = g1.create_dataset('Left-Hippocampus/vol_with_bg', data = img_array)
            # print ("dset3.name", dset3.name)
        else:
            print ("img file missing")
            pass




    ## Add stats group
    table_df = x_df.drop(['ergoid', 'rs_cohort', 'patient_name'], axis=1)
    table_mean = table_df.mean().values
    # print ("table_mean", table_mean)

    table_std = table_df.std().values
    # print ("table_std", table_std)

    g2 = hf.create_group('stats')
    g3 = g2.create_group('Left-Hippocampus')
    g4 = g2.create_group('tabular')
    dset_col = g4.create_dataset('columns', data = table_df.columns.values)
    dset_mean = g4.create_dataset('mean', data = table_mean)
    dset_std = g4.create_dataset('stddev', data = table_std)
    g5 = g3.create_group('vol_with_bg')
    dset_imgmeta = g5.create_dataset('dummy_data', data = np.ones((img_array.shape[0],img_array.shape[1],img_array.shape[2]), dtype = "float16"))
    dset_imgshape = g5.create_dataset('shape_data', data = np.zeros((img_array.shape[0],img_array.shape[1],img_array.shape[2]), dtype = "float16"))

    # dset_dummy = g3.create_dataset('columns', data = table_df.columns.values)

    hf.close()



def generate_datafiles_test(hdf5_filename, x_df, y_df, len_df, calc_dir, fold ):
    hf = h5py.File(hdf5_filename, 'w')
    lenfol_df = pd.DataFrame([])
    lenfol_list =[]
    for index, id in enumerate(x_df["patient_name"].values):
        print ("id", id)


        img_filepath = os.path.join(calc_dir, "pericaridum_%s.npy" %id)
        img_filecheck = Path(img_filepath)
        if img_filecheck.is_file():

            img_array = np.load(img_filepath)

            table_array = x_df.loc[(x_df["patient_name"] == id)].drop(['ergoid', 'rs_cohort', 'patient_name'], axis=1).values
            table_array = table_array.ravel()

            lenfol = len_df.values[index][0]
            fstat = y_df.values[index][0]
            print ("lenfol", lenfol)
            # print ("fstat", fstat)
            lenfol_list.append(lenfol)


            g1 = hf.create_group('%s' %id)
            g1.attrs['RID'] = id
            g1.attrs['VISCODE'] = "bl"
            g1.attrs['DX'] = "CN"
            g1.attrs['time'] = lenfol
            if fstat == 0:
                g1.attrs['event'] = "no"
            elif fstat == 1:
                g1.attrs['event'] = "yes"

            dset2 = g1.create_dataset("tabular", data = table_array)
            # print ("dset2.name", dset2.name)
            # dset3 = g1.create_dataset('Left-Hippocampus/vol_with_bg', data = img_array, shape = img_array.shape)
            
            
            g_img = g1.create_group('Left-Hippocampus')
            dset3 = g_img.create_dataset('vol_with_bg', data = img_array)
            # dset3 = g1.create_dataset('Left-Hippocampus/vol_with_bg', data = img_array)
            # print ("dset3.name", dset3.name)
        else:
            print ("img file missing")
            pass

    lenfol_df["lenf"] = lenfol_list
    lenfol_df.to_csv("lenfol_check_%s.csv" %fold)


    ## Add stats group
    table_df = x_df.drop(['ergoid', 'rs_cohort', 'patient_name'], axis=1)
    table_mean = table_df.mean().values
    # print ("table_mean", table_mean)

    table_std = table_df.std().values
    # print ("table_std", table_std)

    g2 = hf.create_group('stats')
    g3 = g2.create_group('Left-Hippocampus')
    g4 = g2.create_group('tabular')
    dset_col = g4.create_dataset('columns', data = table_df.columns.values)
    dset_mean = g4.create_dataset('mean', data = table_mean)
    dset_std = g4.create_dataset('stddev', data = table_std)
    g5 = g3.create_group('vol_with_bg')
    dset_imgmeta = g5.create_dataset('dummy_data', data = np.ones((img_array.shape[0],img_array.shape[1],img_array.shape[2]), dtype = "float16"))
    dset_imgshape = g5.create_dataset('shape_data', data = np.zeros((img_array.shape[0],img_array.shape[1],img_array.shape[2]), dtype = "float16"))

    # dset_dummy = g3.create_dataset('columns', data = table_df.columns.values)

    hf.close()


def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data
    '''

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    '''

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data

def df_to_xy (df, cac):

    print ("len(df)", len(df))

    if cac:
        df = df.drop(['Ln_CACvol'], axis=1)
    else:
        df = df.drop(['Ln_CAC', 'Ln_CACvol'], axis=1)
                     


    DURATION_COL = "lenfol"
    EVENT_COL = "fstat"
    ergo_id = df['ergoid']
    rs_cohort = df['rs_cohort']
    patient_name = df['patient_name'].astype(int).astype(str)
    df = df.drop(['ergoid', 'rs_cohort', 'patient_name'], axis=1)
    y = df.pop(EVENT_COL).to_frame()

    ## Normalization
    duration_values = df[DURATION_COL].values
    df = df.drop([DURATION_COL], axis=1)

    norm_df_values, norm_parameters_n = normalization(df.values)
    X = pd.DataFrame(norm_df_values, columns = df.columns)   
 
    X.insert(loc=0, column='ergoid', value=ergo_id)
    X.insert(loc=0, column='rs_cohort', value=rs_cohort)  
    X.insert(loc=0, column='patient_name', value=patient_name)  

    X[DURATION_COL] = duration_values

    return df, X, y

## Access to the server and the ESS data folder
current_dir = os.path.dirname(os.path.abspath(__file__))
figure_dir = os.path.join(current_dir, 'Figure_thres')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

temp_image_dir = os.path.join(current_dir, 'temp_image') 
storage_dir = "/data/scratch/hmo"


tabular_dir = os.path.join(current_dir, 'Tabular')
if not os.path.exists(tabular_dir):
    os.makedirs(tabular_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cac', help="inclusion of cac variable", default=False, action="store_true")
    parser.add_argument('-crop', help="inclusion of crop variable", default=False, action="store_true")
    parser.add_argument('--outcome', help="outcome definition", default="CHD", type=str)
    parser.add_argument('--imputation', help="different methods for data imputataion", default="mice", type=str)
    parser.add_argument('-heartslices', type=int, help="number of slices for pericardium", default=30)
    parser.add_argument('-resize', type=int, help="zoom", default=128)

    parser.add_argument('--category', help="category", default="cacs0", type=str)


    # Predictors:
    # Traditional risk factors (TRF): Age, sex, prev_DM, systolic BP, non-HDL cholesterol, current smoking
    # Extended TRF: TRF, BMI, famHxMI, BP-lowering med, statin, aspirin, past smoking
    # Exteded lab: Ext TRF, proBNP, hsTnT, CRP, eGFR, triglycerides, LDL (Martin-Hopkins; i/o nonHDL)

    args = parser.parse_args()
    cac = args.cac
    crop = args.crop
    outcome = args.outcome
    resize = args.resize

    imputation = args.imputation
    num_heartslice = args.heartslices

    category = args.category

    if crop:
        heart_dir = os.path.join(storage_dir, 'heart_array_%s_%s_crop' %(num_heartslice, resize))
        calc_dir = os.path.join(storage_dir, 'calc_array_%s_%s_crop' %(num_heartslice, resize))
    else:
        heart_dir = os.path.join(storage_dir, 'heart_array_%s_%s' %(num_heartslice, resize))
        calc_dir = os.path.join(storage_dir, 'calc_array_%s_%s' %(num_heartslice, resize))


    print ("heart_dir", heart_dir)
    print ("calc_dir", calc_dir)

    ## Load tabular data and normalize
    tabular_filpath = os.path.join(tabular_dir, 'RS_CT_subcohort_time2event_%s_%s_linked_%s.csv' %(outcome, imputation, category))
    df = pd.read_csv(tabular_filpath, sep = ",")

    df, X, y = df_to_xy (df, cac)


    tabular_ex_filpath = os.path.join(tabular_dir, 'RS_CT_subcohort_time2event_%s_%s_linked_%s_excluded.csv' %(outcome, imputation, category))
    df_ex = pd.read_csv(tabular_ex_filpath, sep = ",")


    df_ex, X_ex, y_ex = df_to_xy (df_ex, cac)

    DURATION_COL = "lenfol"
    EVENT_COL = "fstat"

    ## Staratified split (tran,val,test)
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)


    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        print("len(train_index)", len(train_index))
        # print(f"  Test:  index={test_index}")
        print("len(test_index)", len(test_index))

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        X_train = pd.concat([X_train,X_ex])
        y_train = pd.concat([y_train,y_ex])

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,stratify=y_train, test_size=0.3, random_state=0)



        ## Exclude lenfol 
        len_train = X_train.pop(DURATION_COL).to_frame()
        len_val = X_val.pop(DURATION_COL).to_frame()
        len_test = X_test.pop(DURATION_COL).to_frame()

        ## Create and save hdf5 files for train, eval, test sepereately.
        print (X_train)
        print (X_val)
        print (X_test)
        print (len(X_train))
        print (len(X_val))
        print (len(X_test))
        if cac:
            generate_datafiles('DAFT/ergo_train_cac_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_train, y_train, len_train, heart_dir)
            generate_datafiles('DAFT/ergo_val_cac_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_val, y_val, len_val, heart_dir)
            generate_datafiles('DAFT/ergo_test_cac_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_test, y_test, len_test, heart_dir)
        else:
            generate_datafiles('DAFT/ergo_train_heart_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_train, y_train, len_train, heart_dir, i)
            generate_datafiles('DAFT/ergo_val_heart_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_val, y_val, len_val, heart_dir, i)
            generate_datafiles_test('DAFT/ergo_test_heart_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_test, y_test, len_test, heart_dir, i)



    ## Staratified split (tran,val,test)
    skf = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i+5}:")
        # print(f"  Train: index={train_index}")
        print("len(train_index)", len(train_index))
        # print(f"  Test:  index={test_index}")
        print("len(test_index)", len(test_index))

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train = pd.concat([X_train,X_ex])
        y_train = pd.concat([y_train,y_ex])

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.3, random_state=100)



        ## Exclude lenfol 
        len_train = X_train.pop(DURATION_COL).to_frame()
        len_val = X_val.pop(DURATION_COL).to_frame()
        len_test = X_test.pop(DURATION_COL).to_frame()

        ## Create and save hdf5 files for train, eval, test sepereately.
        print (X_train)
        print (X_val)
        print (X_test)
        print (len(X_train))
        print (len(X_val))
        print (len(X_test))

        if cac:
            generate_datafiles('DAFT/ergo_train_cac_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_train, y_train, len_train, heart_dir)
            generate_datafiles('DAFT/ergo_val_cac_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_val, y_val, len_val, heart_dir)
            generate_datafiles('DAFT/ergo_test_cac_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i), X_test, y_test, len_test, heart_dir)
        else:
            generate_datafiles('DAFT/ergo_train_heart_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i+5), X_train, y_train, len_train, heart_dir, i)
            generate_datafiles('DAFT/ergo_val_heart_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i+5), X_val, y_val, len_val, heart_dir, i)
            generate_datafiles_test('DAFT/ergo_test_heart_%s_%s_%s_excluded_%s.hdf5' %(outcome, resize, category, i+5), X_test, y_test, len_test, heart_dir, i)




if __name__ == '__main__':
    main()    