

import os,sys
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Specify the directory containing your CSV files
current_dir = os.path.dirname(os.path.abspath(__file__))
daft_dir = os.path.join(current_dir, 'daft')
calib_table_dir = os.path.join(daft_dir, 'calib_table')


category = "cacs0"

##########################################################


# # Optional: Only include files ending with .csv
# csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]


for i in range(10):

    discr_net = "concat5fc"
    category_dir = os.path.join(calib_table_dir, category)
    csv_folder = os.path.join(category_dir, discr_net)

    csv_files = "table_calib_%s_%s_%s.csv" %(category, discr_net, i)
    file_path = os.path.join(csv_folder, csv_files)
    merged_df = pd.read_csv(file_path)
    merged_df = merged_df[merged_df["time"]<=3650]



    nn_true_df = merged_df[merged_df["event"]==True]

    true_output = nn_true_df["output"].values
    nn_true_df["true_output_normalized"] = (true_output - true_output.min()) / (true_output.max() - true_output.min())



    plt.figure(figsize=(1.5,1.5))
    # matplotlib.use('Agg')
    sns.set_theme(style="ticks")
    if i==0:
        sns.scatterplot(x="time", y="true_output_normalized", data=nn_true_df, color="dodgerblue",edgecolor="black", alpha=0.5, label="Proposed", zorder=2)
    else:
        sns.scatterplot(x="time", y="true_output_normalized", data=nn_true_df, color="dodgerblue",edgecolor="black", alpha=0.5, zorder=2)
    ax = plt.gca() # Get a matplotlib's axes instance
    # plt.title("Agatston score correlation")
    plt.xlabel("Time-to-event (Days)")
    plt.ylabel("Normalized partial hazard")

    # The following code block adds the correlation line:
    m, b = np.polyfit(nn_true_df["time"], nn_true_df["true_output_normalized"], 1)
    # X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],100)
    X_plot = np.linspace(-100,ax.get_xlim()[1],100)
    print ("m", m)
    print ("b", b)
    plt.plot(X_plot, m*X_plot + b, 'black', linewidth=4, alpha=0.6)
    plt.plot(X_plot, m*X_plot + b, 'b-', linewidth=3, alpha=0.6)


    max_value =  max(max(nn_true_df["time"]), max(nn_true_df["true_output_normalized"]))
    # plt.ylim(-100, max_value+300)
    # plt.xlim(-100, max_value+300)

    print ("max_value", max_value)

    # plt.ylim(-100, max_value+300)
    plt.ylim(-0.1, 1.4)
    
    plt.xlim(-100, max_value+200)
    plt.yticks(np.arange(0, 1.2, 0.2)) 


    discr_net = "deepsurv"
    category_dir = os.path.join(calib_table_dir, category)
    csv_folder = os.path.join(category_dir, discr_net)

    csv_files = "table_calib_%s_%s_%s.csv" %(category, discr_net, i)
    file_path = os.path.join(csv_folder, csv_files)
    merged_df = pd.read_csv(file_path)

    merged_df = merged_df[merged_df["time"]<=3650]
    # merged_df = merged_df[merged_df["output"]<=10]

    cox_true_df = merged_df[merged_df["event"]==True]

    true_output = cox_true_df["output"].values
    cox_true_df["true_output_normalized"] = (true_output - true_output.min()) / (true_output.max() - true_output.min())


    sns.set_theme(style="ticks")
    if i==0:
        sns.scatterplot(x="time", y="true_output_normalized", data=cox_true_df, color="yellow",edgecolor="black", alpha=0.5, label="DeepSurv", zorder=1)
    else:
        sns.scatterplot(x="time", y="true_output_normalized", data=cox_true_df, color="yellow",edgecolor="black", alpha=0.5, zorder=1)
    ax = plt.gca() # Get a matplotlib's axes instance

    # The following code block adds the correlation line:
    m, b = np.polyfit(cox_true_df["time"], cox_true_df["true_output_normalized"], 1)
    X_plot = np.linspace(-100,ax.get_xlim()[1],100)

    plt.plot(X_plot, m*X_plot + b, 'black', linewidth=4, alpha=0.6)
    plt.plot(X_plot, m*X_plot + b, 'y-', linewidth=3, alpha=0.6)



    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    discr_net = "deepsurv"
    category_dir = os.path.join(calib_table_dir, category)
    csv_folder = os.path.join(category_dir, discr_net)

    if i==0:
        plt.legend(loc="upper right", bbox_to_anchor=(1, 1.4))

    plt.savefig(os.path.join(csv_folder, 'calib_plot_fold_merged_%s.png' %i), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(csv_folder, 'calib_plot_fold_merged_%s.eps' %i), bbox_inches='tight', dpi=300)
    plt.close()







# ################################################################
# discr_net = "cox"

# category_dir = os.path.join(calib_table_dir, category)
# csv_folder = os.path.join(category_dir, discr_net)



# for i in range(10):

#     csv_files = ["cox_calib_%s_%s.csv" %(category, i)]

#     # Create a list to store individual DataFrames
#     dataframes = []

#     # Loop through and read each CSV file
#     for file in csv_files:
#         file_path = os.path.join(csv_folder, file)
#         df = pd.read_csv(file_path)
#         dataframes.append(df)

#     # Merge all DataFrames into a single DataFrame
#     merged_df = pd.concat(dataframes, ignore_index=True)

#     # Display the result
#     print (merged_df)
#     print("Merged DataFrame shape:", merged_df.shape)
#     print(merged_df.head())


#     cox_true_df = merged_df[merged_df["event"]==True]


#     true_output = cox_true_df["output"].values
#     cox_true_df["true_output_normalized"] = (true_output - true_output.min()) / (true_output.max() - true_output.min())

#     print (cox_true_df)
#     print("Merged DataFrame shape:", cox_true_df.shape)
#     print(cox_true_df.head())


#     plt.figure(figsize=(3,3))
#     # matplotlib.use('Agg')
#     sns.set_theme(style="ticks")
#     sns.scatterplot(x="time", y="true_output_normalized", data=cox_true_df, color="yellow",edgecolor="black", alpha=0.5)
#     ax = plt.gca() # Get a matplotlib's axes instance
#     # plt.title("Agatston score correlation")
#     plt.xlabel("Time-to-event (Days)")
#     plt.ylabel("Normalized predictied partial hazard")


#     # plt.text(.55, .32, r"$r$"+"={:.3f}".format(r), transform=ax.transAxes)
#     # plt.text(.55, .25, r"$\rho$"+"={:.3f}".format(rho), transform=ax.transAxes)
#     # plt.text(.6, .25, "Kendall's ={:.3f}".format(tau), transform=ax.transAxes)


#     # The following code block adds the correlation line:
#     m, b = np.polyfit(cox_true_df["time"], cox_true_df["true_output_normalized"], 1)
#     X_plot = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],10)
#     print ("m", m)
#     print ("b", b)
#     plt.plot(X_plot, m*X_plot + b, 'r-')

#     plt.plot(X_plot, X_plot, linestyle = '--', color = 'lime')

#     max_value =  max(max(cox_true_df["time"]), max(cox_true_df["true_output_normalized"]))
#     # plt.ylim(-100, max_value+300)
#     # plt.xlim(-100, max_value+300)

#     print ("max_value", max_value)

#     # plt.ylim(-100, max_value+300)
#     plt.ylim(0, 1)
#     plt.xlim(-100, max_value+300)


#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     plt.savefig(os.path.join(csv_folder, 'calib_plot_fold_%s.png' %i), bbox_inches='tight', dpi=300)
#     plt.savefig(os.path.join(csv_folder, 'calib_plot_fold_%s.eps' %i), bbox_inches='tight', dpi=300)
#     plt.close()


