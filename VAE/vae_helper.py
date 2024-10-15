import vae
import analysevae


import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme('paper', "whitegrid", palette='Set2')


class SaveResults:
    """Class to save model and training history."""
    def __init__(self, model, results_dir, model_name, history):
        self.model = model
        self.results_dir = results_dir
        self.model_name = model_name
        self.history = pd.DataFrame(history.history)

    def save_model_weights(self):
        """Save the model weights to a file."""
        fname = f"{self.results_dir}/model_weights/{self.model_name}.keras"
        self.model.save(fname)
        print(f"Saved model to {fname}.")

    def save_history(self):
        """Save the training history to a CSV file."""
        history_path = f'{self.results_dir}/training_history.csv'
        self.history.to_csv(history_path, index=False)
        print(f"Saved history to '{history_path}'.")

    def plot_metric(self, metric, ax):
        """Plot a specified metric on the provided axes."""
        ax.plot(self.history[metric], label='Training')
        ax.plot(self.history[f"val_{metric}"], label='Validation')
        ax.set_xlabel('Epoch')
        metric_label = metric.replace('_', ' ').title()
        if metric_label == 'Elbo':
            metric_label = metric_label.upper()
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} Over Epochs")
        ax.legend()

    def plot_cyclical_annealing(self, ax):
        """Plot KL Annealing over epochs."""
        ax.plot(self.history['annealing'], label='Annealing')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Annealing')
        ax.set_title('KL Annealing Over Epochs')
        ax.legend()

    def plot_metrics(self):
        """Plot training metrics and save the figure."""
        fig, ax = plt.subplots(1, 4, figsize=(20, 5), layout='tight')

        self.plot_metric('ELBO', ax[0])
        self.plot_metric('kl_loss', ax[1])
        self.plot_metric('reconstruction_loss', ax[2])
        self.plot_cyclical_annealing(ax[3])

        plt.suptitle("Training History")

        figure_name = f'{self.results_dir}/plots.png'
        plt.savefig(figure_name, dpi=300)
        print(f"Successfully saved figure to {figure_name}")

        plt.close()

    def save_model_and_logs(self):
        self.save_history()
        self.plot_metrics()

class EvaluateModel:
    """Class to evaluate the model."""

    def __init__(self, model, model_name, train, val):
        self.model = model
        self.model_name = model_name
 
        self.train = train
        _, _, self.latent_train = model.encoder(train, training=False)
        self.reconstructed_train = model.decoder(self.latent_train, training=False)

        self.val = val
        _, _, self.latent_val = model.encoder(val, training=False)
        self.reconstructed_val = model.decoder(self.latent_val, training=False)

        self.results_dir = f'results/{model_name}/evaluation'
        os.makedirs(self.results_dir, exist_ok=True)

    def save_data(self, data, name):
        """Save the given data as a .npy file."""
        np.save(f'{self.results_dir}/{name}.npy', data.numpy())
        print(f"Saved {name} to '{self.results_dir}/{name}.npy'")

    def get_tsne(self):
        """Generate and save t-SNE representations for latent variables."""
        tsne = TSNE(n_components=2, random_state=0, init='pca')
        train_tsne = tsne.fit_transform(self.latent_train)
        val_tsne = tsne.fit_transform(self.latent_val)

        np.save(f'{self.results_dir}/train_tsne.npy', train_tsne)
        np.save(f'{self.results_dir}/val_tsne.npy', val_tsne)
        print("Saved t-SNE representations.")

    def evaluate(self):
        """Evaluate the model by saving latent representations, reconstructions, and t-SNE plots."""
        self.save_data(self.latent_train, 'latent_train')
        self.save_data(self.latent_val, 'latent_val')
        print("Saved latent representations.")

        self.save_data(self.reconstructed_train, 'reconstruction_train')
        self.save_data(self.reconstructed_val, 'reconstruction_val')
        print("Saved reconstructions.")
        self.get_tsne()
        print("Evaluation complete.")


def get_model_name(by, original_dim, latent):
    """Returns a unique model name based on the current time"""
    return f"vae_{by}_{original_dim}_{latent}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

def create_model_directory(by, original_dim, latent_dim):
    model_name = get_model_name(by, original_dim, latent_dim)
    if os.path.exists(f"results/{model_name}"):
        create_model_directory(model_name)
    else:
        print(f"Model name: {model_name}")
        results_dir = f'results/{model_name}'
        os.makedirs(results_dir, exist_ok=True)
        return model_name, results_dir

def load_masking_genes(by, n_genes):
    fname = f'{'top_genes_by_' if by != 'random' else ''}{by}{'_genes' if by == 'random' else ''}.csv'
    genes = pd.read_csv(f"Data/{fname}").head(n_genes)['index'].values
    return genes

def mask_gene_index(data, by, n_genes, masking_value=-1.):
    masked_data = data.copy()
    masked_data[:, load_masking_genes(by, n_genes)] = masking_value
    return masked_data

def mask_age_index(data):
    masked_data = data.copy()
    masked_data[:, -1] = -1.
    return masked_data

def set_up_masking(train, val, mask_age=False, mask_genes=False, mask_by=None, n_genes=None):
    if not mask_age and not mask_genes:
        return train, val
    if mask_age:
        masked_train = mask_age_index(train)
        masked_val = mask_age_index(val)
        return masked_train, masked_val

    if mask_genes:
        masked_train = mask_gene_index(masked_train, mask_by, n_genes)
        masked_val = mask_gene_index(masked_train, mask_by, n_genes)

def save_model_info(results_dir, vae_info, data_shapes, training_info):
    def write_section(title, info):
        f.write(f"\n{title} ===================================\n")
        for key, value in info.items():
            f.write(f"\t{key}: {value}\n")
    
    with open(f'{results_dir}/model_info.txt', 'x') as f:
        write_section("VAE INFO", vae_info)
        write_section("DATASET INFO", data_shapes)
        write_section("TRAINING INFO", training_info)

    print(f"Saved model info to {results_dir}/model_info.txt")

def build_vae(
        original_dim=61124, 
        latent_dim=64, 
        masking_value=-1., 
        dropout_rate=0.5, 
        loss=tf.keras.losses.MeanAbsoluteError()
):
    model = vae.VariationalAutoEncoder(original_dim, latent_dim, masking_value, dropout_rate, loss=loss)
    model.build()
    return model

def train_model(train, val, by, mask_age=True, mask_genes=True, num_masked_genes=128, mask_by='kl',
                original_dim=61124, latent_dim=64, batch_size=128, 
                num_epochs=200, dropout_rate=0., learning_rate=1e-5, 
                loss=tf.keras.losses.BinaryCrossentropy(), 
                kl_annealing_cycle_length=100, kl_annealing_ratio=0.9):

    train_masked, val_masked = set_up_masking(train, val, mask_age, mask_genes, mask_by, num_masked_genes)
    print(train_masked.shape, val_masked.shape)

    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
    model = build_vae(original_dim, latent_dim, -1., dropout_rate, loss)
    model.compile(optimizer=opt, loss=loss)

    # Save model info
    model_name, results_dir = create_model_directory(by, original_dim, latent_dim)

    vae_info = {
        "Model name": model_name,
        "Original Dimension": original_dim,
        "Latent Dimension": latent_dim,
        "Masking Age": mask_age
    }

    data_shapes = {
        "By": [by, mask_by],
        "Train": train.shape,
        "Val": val.shape
    }

    training_info = {
        "Batch Size": batch_size,
        "Num Epochs": num_epochs,
        "Learning Rate": learning_rate,
        "Loss": loss.__class__.__name__,
        "Optimizer": opt.__class__.__name__,
        "KL Annealing Ratio": kl_annealing_ratio,
        "KL Annealing n Cycles": kl_annealing_cycle_length
    }

    save_model_info(results_dir, vae_info, data_shapes, training_info)

    kl_annealing = vae.CyclicalAnnealingCallback(num_epochs, kl_annealing_cycle_length, kl_annealing_ratio)
    callbacks = [kl_annealing]

    history = model.fit(
        train_masked, train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(val_masked, val),
        verbose=2,
        callbacks=callbacks if kl_annealing_ratio > 0 else None
    )
    
    SaveResults(model, results_dir, model_name, history).save_model_and_logs()
    EvaluateModel(model, model_name, train, val).evaluate()
    analysevae.EvaluateReconstructions(results_dir, original_dim, by)
