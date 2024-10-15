import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

dir_head = 'Data'

sns.set_palette("Set2")
sns.set_theme('paper', "whitegrid")

class EvaluateReconstructions:
    def __init__(self, results_dir, num_genes, by):
        self.num_genes = num_genes
        self.results_dir = results_dir
        self.reconstructed_data_dir = f"{self.results_dir}/evaluation/"
        self.load_data(by)
        self.plot_age_reconstruction()
        self.get_age_mae()
        self.create_tsne()

    def get_train_val_fnames(self, by):
        if by != 'all':
            train_fname = f"{dir_head}/top_genes/by_{by}/train_top_{self.num_genes}_genes_{by}.npy"
            val_fname = f"{dir_head}/top_genes/by_{by}/val_top_{self.num_genes}_genes_{by}.npy"
        else:
            train_fname = f"{dir_head}/full_dataset/train_scaled.npy"
            val_fname = f"{dir_head}/full_dataset/val_scaled.npy"
        return train_fname, val_fname

    def load_data(self, by):
        train_fname, val_fname = self.get_train_val_fnames(by)
        self.train = np.load(train_fname, allow_pickle=True).astype(np.float32)
        self.val = np.load(val_fname, allow_pickle=True).astype(np.float32)
        self.reconstructed_train = np.load(self.reconstructed_data_dir + "reconstruction_train.npy")
        self.reconstructed_val = np.load(self.reconstructed_data_dir + "reconstruction_val.npy")
        self.tsne_train = np.load(self.reconstructed_data_dir + "train_tsne.npy")
        self.tsne_val = np.load(self.reconstructed_data_dir + "val_tsne.npy")
    
    def compare_true_and_reconstructed(self, col_index, dataset, ax):
        true = self.train if dataset == 'Training' else self.val
        reconstructed = self.reconstructed_train if dataset == 'Training' else self.reconstructed_val
        a = sns.histplot(true[:, col_index], label='True', element='step', ax=ax)
        b = sns.histplot(reconstructed[:, col_index], label='Reconstructed', element='step', ax=ax)
        ax.set(xlim=(0., 1.), yscale='log', xlabel='Age', ylabel='Count', title=dataset)
        return ax.get_legend_handles_labels()

    def plot_age_reconstruction(self):
        col_index = -1
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='tight')
        h, l = self.compare_true_and_reconstructed(col_index, 'Training', ax[0])
        _, _ = self.compare_true_and_reconstructed(col_index, 'Validation', ax[1])
        plt.suptitle(f"Distributions of True vs. Reconstructed Ages")
        fig.legend(handles=h, labels=l, ncols=2)
        plt.savefig(f"{self.results_dir}/evaluation/age_reconstruction.svg", dpi=300)
        plt.close()
    
    def get_age_mae(self):
        results = {
            'train accuracy': mean_absolute_error(self.train[:, -1], self.reconstructed_train[:, -1]),
            'val accuracy': mean_absolute_error(self.val[:, -1], self.reconstructed_val[:, -1])
        }
        results_df = pd.DataFrame([results])
        results_df.to_csv(self.reconstructed_data_dir + "age_mae.csv")

    def prep_clinical_data(self, dataset):
        clinical_data = pd.read_csv(f"{dir_head}/full_dataset/clinical/{dataset}_clinical.csv").drop(columns=['Unnamed: 0'])
        tsne = self.tsne_train if dataset == 'train' else self.tsne_val
        clinical_data['x'] = tsne[:, 0]
        clinical_data['y'] = tsne[:, 1]
        clinical_data['dataset'] = dataset
        return clinical_data

    def plot_latent_space(self, data, variable, sort=None):
        grid = sns.relplot(data=data, kind='scatter', x='x', y='y', hue=variable, col='dataset', col_wrap=2, hue_order=sort)
        grid.savefig(f"{self.results_dir}/evaluation/latent_space_{variable}.svg", dpi=300)
        plt.close()

    def create_tsne(self):
        clinical_train = self.prep_clinical_data('train')
        clinical_val = self.prep_clinical_data('val')
        all_data = pd.concat([clinical_train, clinical_val])
        project_ids = sorted(clinical_train['Project ID'].unique())
        for variable in ['Project ID', 'age_at_index', 'gender']:
            self.plot_latent_space(all_data, variable, project_ids if variable == 'Project ID' else None)
