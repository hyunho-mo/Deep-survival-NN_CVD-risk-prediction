
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from datetime import datetime
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator



# disable chained assignments
pd.options.mode.chained_assignment = None 


## Access to the server and the ESS data folder
current_dir = os.path.dirname(os.path.abspath(__file__))



tabular_dir = os.path.join(current_dir, 'Tabular')
if not os.path.exists(tabular_dir):
    os.makedirs(tabular_dir)

figure_dir = os.path.join(current_dir, 'Figure_stats')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# template_csv_path =  os.path.join(current_dir, 'MyDigiTwin_RS_template.csv')
# study_popluation_csv_path =  os.path.join(tabular_dir, 'RS_CT_subcohort_population.csv')
study_popluation_csv_path =  os.path.join(tabular_dir, 'RS_CT_subcohort_population_linked.csv')


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
    plt.ylabel('%s survival function' %argument)
    plt.xlabel('Timeline (days)')

    plt.savefig(os.path.join(figure_dir, 'km_curve_%s.png' %argument))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ascvd', help="define ascvd outcomes", default=False, action="store_true")
    parser.add_argument('-hard_ascvd', help="define hard_ascvd outcomes", default=False, action="store_true")
    parser.add_argument('-chd', help="define chd outcomes", default=False, action="store_true")
    parser.add_argument('-hard_chd', help="define hard_chd outcomes", default=False, action="store_true")
    # parser.add_argument('-ext_ascvd',  help="", default="original")
    # parser.add_argument('-hard_chd', help="", default="original")
    # parser.add_argument('-hard_chd', help="", default="original")


    args = parser.parse_args()
    ascvd = args.ascvd
    hard_ascvd = args.hard_ascvd
    chd = args.chd
    hard_chd = args.hard_chd


    ## Load study population csv file to dataframe
    df = pd.read_csv(study_popluation_csv_path, sep = ",", na_values=' ')    
    print ("number of participants: ", len(df))


    # ####################### Outcome ##########################

    max_follow = datetime(2015, 1 , 1)

    if ascvd: # ASCVD (ASCVD mort, MI, stroke)
        argument = "ASCVD"

        df['fup_ASCVD_days'] = df['fup_ASCVD'] *365
        df['lenfol'] = (df['enddat_ASCVD'].astype('datetime64[ns]')  - df['scandate'].astype('datetime64[ns]') ).dt.days

        df["fstat"] = np.zeros(len(df))
        df["fstat"].loc[(df["inc_ASCVD"] == 1)] = 1

        ## Exclude negative length of follow up
        df = df.loc[(df['lenfol']>=0)]

        print (df["fstat"].values.ravel().sum())
        print (df["fstat"].values.ravel().sum()/len(df)*100)
        draw_km_curve(df['lenfol'], df["fstat"], argument)

        processed_csv_path =  os.path.join(tabular_dir, 'RS_CT_subcohort_%s_outcomes_linked.csv' %argument)



    if hard_ascvd: # hard ASCVD 
        argument = "hardASCVD"

        df['fup_hardASCVD_days'] = df['fup_hardASCVD'] *365
        df['lenfol'] = (df['enddat_hardASCVD'].astype('datetime64[ns]')  - df['scandate'].astype('datetime64[ns]') ).dt.days

        df["fstat"] = np.zeros(len(df))
        df["fstat"].loc[(df["inc_hardASCVD"] == 1)] = 1

        ## Exclude negative length of follow up
        df = df.loc[(df['lenfol']>=0)]

        print (df["fstat"].values.ravel().sum())
        print (df["fstat"].values.ravel().sum()/len(df)*100)


        draw_km_curve(df['lenfol'], df["fstat"], argument)

        processed_csv_path =  os.path.join(tabular_dir, 'RS_CT_subcohort_%s_outcomes_linked.csv' %argument)


    if chd: #  CHD
        argument = "CHD"

        df['fup_CHD_days'] = df['fup_CHD'] *365
        df['lenfol'] = (df['enddat_CHD'].astype('datetime64[ns]')  - df['scandate'].astype('datetime64[ns]') ).dt.days

        df["fstat"] = np.zeros(len(df))
        df["fstat"].loc[(df["inc_CHD"] == 1)] = 1

        ## Exclude negative length of follow up
        df = df.loc[(df['lenfol']>=0)]

        print (df["fstat"].values.ravel().sum())
        print (df["fstat"].values.ravel().sum()/len(df)*100)


        draw_km_curve(df['lenfol'], df["fstat"], argument)

        processed_csv_path =  os.path.join(tabular_dir, 'RS_CT_subcohort_%s_outcomes_linked.csv' %argument)


    if hard_chd: # hard CHD 
        argument = "hardCHD"

        df['fup_hardCHD_days'] = df['fup_hardCHD'] *365
        df['lenfol'] = (df['enddat_hardCHD'].astype('datetime64[ns]')  - df['scandate'].astype('datetime64[ns]') ).dt.days

        df["fstat"] = np.zeros(len(df))
        df["fstat"].loc[(df["inc_hardCHD1"] == 1)] = 1

        ## Exclude negative length of follow up
        df = df.loc[(df['lenfol']>=0)]

        print (df["fstat"].values.ravel().sum())
        print (df["fstat"].values.ravel().sum()/len(df)*100)


        draw_km_curve(df['lenfol'], df["fstat"], argument)

        processed_csv_path =  os.path.join(tabular_dir, 'RS_CT_subcohort_%s_outcomes_linked.csv' %argument)

    ## Save time-to-event data
    df.to_csv(processed_csv_path, sep = ",", index = False)



    ##################### Plot ########################
    fig, ax = plt.subplots()       
    df["fstat"].replace({1: 'Yes', 0: 'No'}, inplace=True) 
    categories = df["fstat"].unique()
    counts = df["fstat"].value_counts().values
    bar_colors = ['tab:red', 'tab:blue']
    bar_labels = categories
    # hbars = ax.bar(categories, counts, label=bar_labels, color=bar_colors)
    hbars = ax.bar(categories, counts, label=bar_labels, color=bar_colors)
    A_as_ticklabel = [f"{round(100*a/sum(counts),2)}%" for a in counts]
    ax.bar_label(hbars, fmt='%.2f%%', labels=A_as_ticklabel)
    ax.set_ylabel("count")
    plt.title("%s event" %argument)
    ax.set_title("%s event" %argument)
    # ax.legend()
    fig.savefig(os.path.join(figure_dir, "%s_histogram.png" %argument),  bbox_inches='tight', dpi = 300)


    fig, ax = plt.subplots()
    df['scanage'].hist(bins=50, alpha=0.5, legend = True)
    ax.legend()
    ax.set_ylabel("counts")
    plt.title("Age distribution")
    fig.savefig(os.path.join(figure_dir, "age_distribution.png" ))
    


if __name__ == '__main__':
    main()    




