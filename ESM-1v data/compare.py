
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("/cluster/pixstor/xudong-lab/yuexu/D_PLM/ESM-1v data/results/"\
               "/ESM-1v_data_summary_start_end_seq_splm_esm2.csv")

plt.clf()
plt.cla()
# Set the plot size
plt.figure(figsize=(12, 6))

sns.scatterplot(x='Dataset_file', y='checkpoint_0000450.pth-mask-marginals', data=df, color='green', label='dplm-mask-marginals', s=40,alpha=0.6)

# Plot method1
sns.scatterplot(x='Dataset_file', y='ESM2-mask-marginals', data=df, color='purple', label='ESM2-mask-marginals', s=40,alpha=0.6)

# Plot method2
sns.scatterplot(x='Dataset_file', y='ESM2-wt-marginals', data=df, color='orange', label='ESM2-wt-marginals', s=30,alpha=0.6)

# Plot method3
sns.scatterplot(x='Dataset_file', y='checkpoint_run2_0020500.pth-mask-marginals', data=df, color='black', label='checkpoint_run2_0020500.pth-mask-marginals', s=60,alpha=0.6)
sns.scatterplot(x='Dataset_file', y='checkpoint_0280000.pth-mask-marginals', data=df, color='brown', label='checkpoint_0280000.pth-mask-marginals', s=50,alpha=0.6)
sns.scatterplot(x='Dataset_file', y='freezepretrain_MLM_separatetable_0043000_on_280000.pth-mask-marginals', data=df, color='red', label='freezepretrain_MLM_separatetable_0043000_on_280000', s=50,alpha=0.6)

#sns.scatterplot(x='Dataset_file', y='checkpoint_0280000.pth-wt-marginals', data=df, color='lightbrown', label='checkpoint_0280000.pth-wt-marginals', s=100)

# Draw vertical lines at each x-tick
for tick in plt.gca().get_xticks():
    plt.axvline(x=tick, color='gray', linestyle='--', alpha=0.7)

# Set the labels and title
plt.xlabel('Datasets')
plt.ylabel('Spearmans')
plt.title('Comparison of all the methods')

# Show legend
plt.legend()
plt.xticks(rotation=90)  # Change the angle as needed
plt.subplots_adjust(bottom=0.5)  # Adjust the value as needed
# Show the plot
plt.show()

#################compare RLA-euclidean_distance

csv_name='ESM-1v_data_summary_start_end_seq_splm_esm2'
#colname='s-plm-wt-mt-RLA-euclidean_distance'
colname='checkpoint_0001000.pth-wt-mt-RLA-euclidean_distance'
colname='checkpoint_0002000.pth-wt-mt-RLA-euclidean_distance'

#df=pd.read_csv(f"/data/duolin/splm_gvpgit/mutation_usecases/ESM-1v data/results/ESM-1v_data_summary_start_end_seq_splm_esm2.csv")
df=pd.read_csv(f"/cluster/pixstor/xudong-lab/yuexu/D_PLM/ESM_1v_data/results/{csv_name}.csv")

plt.clf()
plt.cla()
# Set the plot size
plt.figure(figsize=(12, 6))

# Plot method1
sns.scatterplot(x='Dataset_file', y='ESM2-wt-mt-RLA-euclidean_distance', data=df, color='purple', label='ESM2-RLA-euclidean_distance', s=60,alpha=0.6)

# Plot method2
#sns.scatterplot(x='Dataset_file', y='s-plm-wt-mt-RLA-euclidean_distance', data=df, color='orange', label='s-plm-RLA-euclidean_distance', s=60,alpha=0.6)
sns.scatterplot(x='Dataset_file', y=colname, data=df, color='orange', label=colname, s=60,alpha=0.6)

# Draw vertical lines at each x-tick
for tick in plt.gca().get_xticks():
    plt.axvline(x=tick, color='gray', linestyle='--', alpha=0.7)

# Set the labels and title
plt.xlabel('Datasets')
plt.ylabel('Spearmans')
plt.title('Comparison of all the methods')
# Show legend
plt.legend()
plt.xticks(rotation=90)  # Change the angle as needed
plt.subplots_adjust(bottom=0.5)  # Adjust the value as needed
# Show the plot
#plt.show()
plt.savefig(f"/cluster/pixstor/xudong-lab/yuexu/D_PLM/ESM_1v_data/results/{csv_name}_{colname}.png")
###########################


################################################bar plots

df=pd.read_csv("/data/duolin/splm_gvpgit/mutation_usecases/ESM-1v data/results/"\
               "/ESM-1v_data_summary_start_end_seq_splm_esm2.csv")

# List of column keys to plot
column_keys = [
    'ESM2-mask-marginals', 
    'ESM2-wt-marginals', 
    'checkpoint_run2_0020500.pth-mask-marginals', 
    'checkpoint_0280000.pth-mask-marginals',
    'freezepretrain_MLM_separatetable_0043000_on_280000.pth-mask-marginals',
    'freezepretrain_MLM_separatetable_0069000_on_280000.pth-mask-marginals',
    'ESM2-wt-mt-RLA-euclidean_distance',
    "s-plm-wt-mt-RLA-euclidean_distance",
]

# Custom labels for x-axis
custom_labels = [
    'ESM2 MASK Marginals', 
    'ESM2 WT Marginals', 
    '0020500 Mask Marginals',
    '0280000 Mask Marginals',
    'train_MLM Mask Marginals', 
    'train_MLM (more steps) Mask Marginals',
    'ESM2-RLA-euclidean_distance',
    's-plm-RLA-euclidean_distance',
]

# Clear any previous plots
plt.clf()
plt.cla()

# Filter DataFrame for the selected columns
df_filtered = df[column_keys]

# Calculate mean and SEM for the selected columns
mean_values = df_filtered.mean()
sem_values = df_filtered.sem()

# Prepare data for plotting
mean_sem_df = pd.DataFrame({
    'Mean': mean_values,
    'SEM': sem_values
})

# Set the plot size
plt.figure(figsize=(14, 8))

# Plot mean with error bars in the specified order
bars = mean_sem_df['Mean'].plot(kind='bar', yerr=mean_sem_df['SEM'], capsize=4, color=['purple', 'orange','black', 'brown','red','red','purple','red'])

# Set the labels and title
plt.xlabel('methods')
plt.ylabel('Values')
plt.title('Mean and SEM for selected methods')

# Rotate x-axis labels
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.5)  # Adjust the value as needed
bars.set_xticklabels(custom_labels, rotation=90)

# Show the plot
plt.show()