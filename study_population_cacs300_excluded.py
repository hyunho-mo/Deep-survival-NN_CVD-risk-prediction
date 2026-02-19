
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from datetime import datetime

# disable chained assignments
pd.options.mode.chained_assignment = None 


## Access to the server and the ESS data folder
current_dir = os.path.dirname(os.path.abspath(__file__))

tabular_dir = os.path.join(current_dir, 'Tabular')
if not os.path.exists(tabular_dir):
    os.makedirs(tabular_dir)


# template_csv_path =  os.path.join(current_dir, 'MyDigiTwin_RS_template.csv')
template_csv_path =  os.path.join(tabular_dir, 'RS_MDCT_subcohort.csv')
id_link_path = os.path.join(tabular_dir, 'ergoid_to_ctdummyid.csv')
processed_csv_path =  os.path.join(tabular_dir, "RS_CT_subcohort_population_cacs300_excluded.csv")



figure_dir = os.path.join(current_dir, 'Figure')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

def main():

    ####################### Study population ##########################
    ## Read CSV file and assign to a dataframe
    template_df = pd.read_csv(template_csv_path, sep = ",", na_values=' ')
    print ("total number of participants: ", len(template_df))

    id_link_df = pd.read_csv(id_link_path, sep = "\t", na_values=' ')

    print (id_link_df)
    print (id_link_df['ergoid'])
    print (id_link_df['patient_name'])

    template_df = template_df.join(id_link_df.set_index('ergoid'), on=['ergoid'])
    print ("total number of participants after id linking: ", len(template_df))

    template_cols = template_df.columns
    print ("template_cols", template_cols)

    ## Include only participants who consent for follow up (Informed consent for follow-up data collection - DataWiki - 24.05.2018)
    template_df = template_df.loc[(template_df["ic_ok"]==1)] 
    print ("total number of participants allowing followup: ", len(template_df))
    ## Drop na rows
    template_df = template_df.dropna(axis=1, how='all')

    print ("total number of participants allowing followup any value: ", len(template_df))
    ## Dropna for date of birth (gebdatum) & Date of entering the Rotterdam Study (startdate)
    # template_df = template_df.dropna(axis=1, how='all')

    template_df = template_df.replace(r'          ', np.NaN)    
    # template_df = template_df.dropna(subset=['fp_startdate'])
    template_df = template_df.dropna(subset=['gebdatum'])
    ## Dropna for scandate
    template_df = template_df.dropna(subset=['scandate'])

    print ("total number of participants including birth and scan date: ", len(template_df))


    ## Align datetime format for all the data columns 
    for col in template_cols:
        if ("dat" in col) or ("date" in col) or ("datum" in col) or ("enddat" in col):
            ## Replace date string to NaN if it is later than 2024 (the current year) or earlier than 1700
            template_df[col].loc[(template_df[col].str[-4:].astype(float)>=2024.0)] = np.NaN
            template_df[col].loc[(template_df[col].str[-4:].astype(float)<=1700.0)] = np.NaN
            template_df[col] = pd.to_datetime(template_df[col], format='%m/%d/%Y')
            template_df[col] = pd.to_datetime(template_df[col], errors = 'coerce')
            # template_df[col] = pd.to_datetime(template_df[col], format='%m-%d-%Y')

    ## Add 'age column' which is calculated by 'fp_startdate' - 'gebdatum'
    # age_values = (template_df['fp_startdate'] -template_df['gebdatum']).dt.days / 365.0
    # age_values = template_df['scanage'] 
    # template_df.insert(loc=3, column='age', value=age_values.round(2))
    
    ### Exclusion
    ## Upper age limit (currently max 105 yrs; n~450 >85 yrs)? Yes, 85
    template_df = template_df.loc[(template_df["scanage"]<=85)] 

    ### Inclusion
    # template_df = template_df.loc[(template_df["CAC"]<=100)] 
    template_df = template_df.loc[(template_df["CAC"]>300)] 

    print ("total number of participants scan age less than 85 & cacs0: ", len(template_df))

    ## Exclude nitrate users (=proxy for angina)? Yes
    ## drop rows nitrate users and nan for this column
    # template_df = template_df.dropna(subset=['nitrates'])

    # print ("num of nan", template_df["prev_AAA_PAD_CEA_OK"].isna().sum()) 

    # ## No prev_MI, prev_PCI, prev_CABG, prevCVATIA, prev_AAA_PAD_CEA
    # intervention_history_cols = ["prev_MI", "prev_PCI", "prev_CABG"]
    # ## Only exclude if prevalent (‘1’ [and/or ‘2’ for MI]), no informed consent (‘7’), or no follow-up available (‘9’)
    # for hist_col in intervention_history_cols:
    #     template_df = template_df.loc[(template_df[hist_col]!=1)] 
    #     template_df = template_df.loc[(template_df[hist_col]!=2)] 
    #     template_df = template_df.loc[(template_df[hist_col]!=7)] 
    #     template_df = template_df.loc[(template_df[hist_col]!=9)] 

    # intervention_history_cols = ["prev_CVATIA", "prev_AAA_PAD_CEA_OK", "prev_dialysis", "nitrates"]
    # ## Only exclude if prevalent (‘1’)
    # for hist_col in intervention_history_cols:
    #     template_df = template_df.loc[(template_df[hist_col]!=1)] 


    ## Inclusion 
    template_df = template_df.loc[(template_df["MyDigiTwin_MDCT"]==0)] 

    print ("number of participants: ", len(template_df))

    template_df.to_csv(processed_csv_path, sep = ",", index = False)

    # ## Participants with prevalent CHD (MI, PCI or CABG)('8') or no follow-up (‘7’) or no informed consent (‘9’) should be excluded from the analyses
    # event_history_cols = ["inc_MI", "inc_PCI", "inc_CABG", "inc_CHD", "inc_hardCHD"]
    # for hist_col in event_history_cols:
    #     template_df = template_df.loc[(template_df[hist_col]!=7)] 
    #     template_df = template_df.loc[(template_df[hist_col]!=8)] 
    #     template_df = template_df.loc[(template_df[hist_col]!=9)] 

    # exit(0)






if __name__ == '__main__':
    main()    





