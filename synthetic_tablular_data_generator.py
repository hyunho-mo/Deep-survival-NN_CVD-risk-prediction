



import sys, os
import numpy as np
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
tabular_dir = os.path.join(current_dir, 'Tabular')
synth_dir = os.path.join(current_dir, 'Ashley_synth')
if not os.path.exists(synth_dir):
    os.makedirs(synth_dir)

calc_img_dir = os.path.join(synth_dir, 'calc_array_30_128_crop')
if not os.path.exists(calc_img_dir):
    os.makedirs(calc_img_dir)


# calc_array_30_128_crop


# Load original real data file
# RS_CT_subcohort_time2event_CHD_mice_linked.csv
tabular_filepath = os.path.join(tabular_dir, 'RS_CT_subcohort_time2event_CHD_mice_linked.csv')
ori_df = pd.read_csv(tabular_filepath, sep = ",")

var_cols = ori_df.columns
print ("var_cols", var_cols)
## Save to synthetic datafile

synth_df = pd.DataFrame([], columns = var_cols)

# synth_df.columns = var_cols

synth_df['ergoid'] = ori_df['ergoid']

for var in var_cols:

    if var =="ergoid":
        continue

    elif var =="patient_name":
        synth_df[var] = ori_df[var]
    
    else:
        temp_array = np.ones(len(ori_df))
        synth_df[var] = temp_array* ori_df[var].values[0]

synth_filepath = os.path.join(synth_dir, 'RS_CT_subcohort_time2event_CHD_mice_linked.csv')
synth_df.to_csv(synth_filepath, sep = ",", index = False)

## Check the shape of the original image files
#/data/scratch/hmo/calc_array_30_128_crop
#calc_1692.npy

ori_image = np.load ('/data/scratch/hmo/calc_array_30_128_crop/calc_1692.npy')
print (ori_image.shape)
print (ori_image)
## Generate the fake image files 
array_temp = np.random.choice([0, 1], size=ori_image.shape, p=[1./3, 2./3])
print (array_temp.shape)
print (array_temp)

name_list = synth_df['patient_name'].tolist()
for name in name_list:

    array_temp = np.random.choice([0, 1], size=ori_image.shape, p=[1./3, 2./3])
    # print (array_temp)
    array_temp_filepath =  os.path.join(calc_img_dir, 'calc_%s.npy' %name)
    np.save(array_temp_filepath, array_temp)