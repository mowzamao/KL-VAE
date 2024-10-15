# importing libraries
import pandas as pd
import numpy as np
from scipy.special import kl_div
from collections import defaultdict
from matplotlib import pyplot as plt
import os

import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')

# VARIABLES
cancer_names = ['Acute Lymphocytic Leukemia', 'Acute Monocytic Leukemia','Acute Myeloid Leukemia', 'Adenocarcinoma of the Lung and Bronchus', 'Chronic Lymphocytic Leukemia', 'Chronic Myeloid Leukemia', 'Hepatic Flexure', 'Hodgkin - Extranodal', 'Hodgkin - Nodal', 'Hodgkin Lymphoma', 'Kaposi Sarcoma (9140)', 'Lymphocytic Leukemia', 'Lymphoma', 'Melanoma of the Skin', 'Mesothelioma (9050-9055)', 'Myeloid and Monocytic Leukemia', 'Myeloma', 'NHL - Extranodal', 'NHL - Nodal', 'Neuroblastoma (9490-9509)', 'Non-Hodgkin Lymphoma', 'Other Acute Leukemia', 'Other Leukemia', 'Other Lymphocytic Leukemia', 'Other Myeloid Monocytic Leukemia']
age_groups = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]


# FUNCTIONS
def get_genecount_data(infile, outfile):
    """
    Converts a GTEx .gct file to a parquet.

    Params
    ------
    infile (str): the file to read
    outfile (str): name for the parquet
    """
    # Reading the Gene Count data, dropping unnecessary 'id' and 'description' columns
    types = defaultdict(int, Name='str',  Description='str')
    df = pd.read_table(infile, engine='pyarrow', header=2, index_col="Name", dtype=types)
    df.to_parquet(outfile)

    return df

def normalise_expression(infile, outfile):
    """
    Brief: 
    - Reads in raw gene counts from a parquet and normalises based on housekeeping genes. 
    - Saves the resulting dataframe to a new parquet.

    PARAMS:
    - infile (str or pandas DataFrame): either filepath (str) of a parquet or a pandas DataFrame containing raw gene count data
    - outfile (str): path for saving normalised data as a parquet

    RETURNS:
    - pandas DataFrame: normalised gene count data
    """
    if type(infile) == str:
        gtex = pd.read_parquet(infile)
    else:
        gtex = infile

    gtex.drop(columns=["Description", "id"], inplace=True)
    housekeeping_genes = ['ENSG00000111640.14', 'ENSG00000075624.13', 'ENSG00000134644.15', 'ENSG00000150991.14']

    # Data is normalised by the mean of the housekeeping genes per individual (column)
    for col in gtex.columns:
        sm: float = 0
        cnt = 0
        for gene in housekeeping_genes:
            if gene not in gtex.index:
                continue
            sm += gtex.at[gene, col]
            cnt += 1
        mean = sm / cnt
        
        gtex[col] = gtex[col].astype(float)
        gtex.loc[:, col] = gtex.loc[:, col] / mean

    gtex = np.log(gtex+1)

    gtex.to_parquet(outfile)
    return gtex

def split_sex(infile, outfile_female, outfile_male):
    """
    Function for splitting normalised GTEx data by sex.
    - Reads a parquet and uses the subject ID as a key for merging the GTEx gene expression data with the GTEx phenotypes data.
    - Splits the dataframe into male and female based on subject sex.
    - Saves female and male DataFrames as pickles.

    Params:
    - infile (str or pandas DataFrame): either a filepath to a parquet or a DataFrame containing normalised gtex data
    - outfile_female (str): a path for saving the female data
    - outfile_male (str): a path for saving the male data

    Returns:
    - List [DataFrame, DataFrame]: index 0 contains the female dataframe, index 1 contains the male dataframe
    """
    phenotypes = pd.read_csv("data/raw_data/GTEx/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep='\t', index_col='SUBJID')
    phenotypes = phenotypes.reset_index()

    if type(infile) == str:
        test_data = pd.read_parquet(infile)
    else:
        test_data = infile
        
    df_reset = test_data.T.reset_index()
    df_reset['key'] = df_reset['index'].str.split('-').str[:2].str.join('-')

    gene_pheno_joined = df_reset.merge(phenotypes[['SUBJID', 'SEX', 'AGE']], left_on='key', right_on='SUBJID', how='left')
    gene_pheno_joined = gene_pheno_joined.sort_values(by=['AGE'])

    female_gtex = gene_pheno_joined[gene_pheno_joined['SEX'] == 2].drop(columns = ['key', 'SUBJID', 'SEX'])
    male_gtex = gene_pheno_joined[gene_pheno_joined['SEX'] == 1].drop(columns = ['key', 'SUBJID', 'SEX'])

    female_gtex.to_pickle(outfile_female)
    male_gtex.to_pickle(outfile_male)

    return [female_gtex, male_gtex]

def get_averages(df):
    """
    Gets the average normalised gene expression per age group from a dataframe.
    
    Params:
    - df (DataFrame): the normalised gene expression data frame

    Returns:
    - pandas DataFrame: A dataframe containing the average gene expression for each gene and for each age group   
    """
    ages = sorted(set(df['AGE'].values))
    mean_dict = {}
    for age in ages:
        mean_dict[age] = df[df['AGE'] == age].mean(axis=0, numeric_only=True)
    
    return mean_dict

def get_median(df):
    """
    Gets the median normalised gene expression per age group from a dataframe.
    
    Params
    - df (DataFrame): the normalised gene expression data frame

    Returns
    - DataFrame: A dataframe containing the median gene expression for each gene and for each age group   
    """
    ages = sorted(set(df['AGE'].values))
    median = {}
    for age in ages:
        median[age] = df[df['AGE'] == age].median(axis=0, numeric_only=True)
    return median

def scale_gtex(gtex):
    """
    Scales GTEx data
    
    Params:
    - gtex (DataFrame)
    
    Returns:
    DataFrame: scaled GTEx Data
    """
    gene_description = pd.read_csv('data/gtex_gene_names.csv', index_col='Name', usecols=['Name', 'Description'])

    scaled_data = {}
    for col in gtex.columns:
        x = gtex[col].values
        s = np.sum(x)
        description = gene_description.at[col, 'Description'] # this is just so i can get the gene name as well as the gene id
        if s > 0:
            x /= s
            scaled_data[col] = [description, x]
        else:
            scaled_data[col] = [description, 0]
    scaled_data = pd.DataFrame.from_dict(scaled_data, orient='index', columns=['Description', 'Scaled Data'])
    return scaled_data
    
def aggregate_GTEx(gtex, sex, tissue):
    """
    - Gets average and median gene expression for each age group for each gene from some GTEx data.
    - Saves the results to a pickle
    
    Params:
    - gtex (DataFrame): Normalised GTEx Data
    - sex (str): either 'female' or 'male'
    - tissue (str): the tissue corresponding to the GTEx data  
    """
    averages = pd.DataFrame(get_averages(gtex)).T
    medians = pd.DataFrame(get_median(gtex)).T

    scaled_averages = scale_gtex(averages)
    averages_fname = "data/GTEx/" + sex + "/scaled_averages/" +str(tissue)+ ".pkl"
    scaled_averages.to_pickle(averages_fname)

    scaled_medians = scale_gtex(medians)
    medians_fname = "data/GTEx/" + sex + "/scaled_medians/" +str(tissue)+ ".pkl"
    scaled_medians.to_pickle(medians_fname)
    
def get_gtex(gtex_samples):
    """
    - reads raw gene count data, normalises it, splits the data based on sex, and then aggregates and scales it, and saves the results

    Params:
    gtex_samples (List[string]): A list of strings corresponding to the GTEx tissue names
    """
    for tissue in gtex_samples:
        # setting file paths
        raw_gtex_path='data/raw_data/GTEx/gene_reads_' + tissue + '.gct'
        results_folder = 'data/GTEx/'
        raw_gtex_parquet = results_folder +'/raw_parquets/'+ tissue + '.parquet'
        normalised_gtex = results_folder+'/normalised_data/' + tissue + '.parquet'
        female_gtex = results_folder + 'female/' + tissue + '.pkl'
        male_gtex = results_folder + 'male/' + tissue + '.pkl'

        # read raw data
        gene_counts = get_genecount_data(raw_gtex_path, raw_gtex_parquet)
        
        # normalise raw data with max's code
        normalised = normalise_expression(gene_counts, normalised_gtex)

        # split data into male and female
        female, male = split_sex(infile=normalised, outfile_female=female_gtex, outfile_male=male_gtex)

        # get scaled averages and medians for each sex
        aggregate_GTEx(female, 'female', tissue)
        aggregate_GTEx(male, 'male', tissue)


def plot_genes(sex, results_path, tissue, cancer, seer, avg, med, rank_start=1, rank_end=10, inverse=False):
    """
    Plots the scaled incidence for the genes with the best KL divergence results. Saves the results to a plots folder in the Seer cancer directory.

    Params:
    - sex (str): 'female' or 'male'
    - results_path (str): path for saving the plots
    - tissue (str): the GTEx tissue name
    - cancer (str): the SEER cancer name
    - seer (NumPy array): the scaled SEER cancer incidence data corresponding with 'cancer'
    - avg (DataFrame): the results from calculating KL divergence from the GTEx average gene expression data
    - med (DataFrame): the results from calculating KL divergence from the GTEx median gene expression data
    - rank_start (int): the number from where to begin the ranking
    - rank_end (int): the number from where to end the ranking
    - inverse (Bool): if True, the scaled seer cancer incidence has been reversed.    
    """
    # set save directory
    images_folder = os.path.join(results_path, 'plots')
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)
    
    # set age groups
    ages = age_groups
    if sex == 'male':
        match tissue:
            case 'bladder': ages = age_groups[0:5]
            case 'kidney_medulla': ages = [age_groups[i] for i in [0, 3]]
            case _: ages = age_groups
    else: 
        match tissue:
            case 'bladder': ages = age_groups[0:4]
            case 'kidney_cortex': ages = age_groups[1:5]
            case 'cervix_endocervix': ages = [age_groups[i] for i in [0, 2, 3, 4]]
            case 'kidney_medulla': ages = age_groups[4]
            case 'cervix_ectocervix': ages = age_groups[0:5]
            case 'minor_salivary_gland': ages = age_groups[0:5]
            case 'fallopian_tube': ages = age_groups[0:5]
            case _: ages = age_groups
        
    # get data and save plots
    for name, df in {'Average': avg, 'Median': med}.items():
        range_genes_data = df[(df['KL Rank'] >= rank_start) & (df['KL Rank'] <= rank_end)]
        
        plt.figure(figsize=(12, 8))
        for i, row in range_genes_data.iterrows():
            gene_exp = row['Scaled Data']
            gene = row['Description']
            kl_div = row['KL Divergence']
            rank_num = row['KL Rank']
            sns.lineplot(x=ages, y=gene_exp, label=f'{gene} (Rank {rank_num}, KL {kl_div:.2f})')
        sns.lineplot(x=ages, y=seer, label='SEER', color='black', linestyle='--')

        if cancer not in cancer_names: 
            cancer = cancer+" Cancer" # so our title says 'breast cancer' instead of just breast
        if inverse == True: 
            plt.title(f'{name} {str.capitalize(sex)} {tissue.replace("_", " ").title()} Gene Expression Over Age vs Inversed SEER {cancer} Incidence Over Age for Best KL Divergence (Rank {rank_start} - Rank {rank_end})')
        else: 
            plt.title(f'{name} {str.capitalize(sex)} {tissue.replace("_", " ").title()} Gene Expression Over Age vs SEER {cancer} Incidence Over Age for Best KL Divergence (Rank {rank_start} - Rank {rank_end})')
        plt.xlabel('Age Group')
        plt.ylabel('Scaled Gene Expression / Incidence')
        plt.legend()

        if inverse == True: 
            fname = str(images_folder) + '/inverse_' + name + '_'+tissue+'.png'
        else: 
            fname = str(images_folder) + '/' + name + '_'+tissue+'.png'
        
        plt.savefig(fname, dpi=300)
        plt.close()

    print("Images saved to ", images_folder)


def get_scaled_seer(sex, cancer, tissue):
    """
    Info
    ----
    Read seer data from .csv, scale it and return the scaled data.

    Params:
    - cancer (str): SEER cancer name
    - tissue (str): GTEx tissue sample name (used to determine which age classes to use from SEER)

    Returns:
    - NumPy Array: list of scaled SEER values in ascending order of age group
    """
    seer = pd.read_csv(f"../data/kl_div/SEER/{cancer}.csv", index_col=0)

    seer_values = seer[str.capitalize(sex)].values

    if sex == 'male':
        match tissue:
            case 'kidney_medulla': seer = seer_values[[0, 3]]
            case 'bladder': seer = seer_values[0:5]
            case _: seer = seer_values
    else:            
        match tissue:
            case 'bladder': seer = seer_values[0:4]
            case 'kidney_cortex': seer = seer_values[1:5]
            case 'cervix_endocervix': seer = seer_values[[0, 2, 3, 4]]
            case 'cervix_ectocervix': seer = seer_values[0:5]
            case 'minor_salivary_gland': seer = seer_values[0:5]
            case 'fallopian_tube': seer = seer_values[0:5]
            case _: seer = seer_values

    seer /= sum(seer)
    return seer

def calc_kl_divergence(df, y):
    """
    Calculates KL divergence for each column (gene) in the data with the SEER incidence distribution
    and adds ranking (ascending).
    Params
    ------
    df (Pandas DataFrame): containing average / median scaled incidence over age data for each GTEx gene
    y (Numpy Array: containing SEER scaled cancer incidence data

    Returns
    -------
    pd.DataFrame: DataFrame with new columns for the KL divergence results and the KL divergence rank for each gene.
    """

    def kl_divergence(expression):
        return np.sum(kl_div(expression, y))

    df['KL Divergence'] = df['Scaled Data'].apply(kl_divergence)

    df = df.sort_values(by='KL Divergence')
    df['KL Rank'] = np.arange(1, len(df) + 1)

    return df


def get_kl_div(sex, seer_cancer_list, gtex_samples, plot_results=True, inverse=False):
    """
    Goes through each seer and gtex dataset, 
    loads scaled average and median normalised gene expression per age group data, 
    calculates KL divergence for each gene, saves it to kl-div/result,
    plots genes if plot_results is set to True
    
    Params
    ------
    sex (str): if 'female', uses female data sets, otherwise uses male data sets
    seer_cancer_list (Array of str): list of SEER cancers to get KL divergence for
    gtex_samples (Array of str): list of GTEx tissues to get KL divergence for    
    plot_results (Bool): if True, plots top 10 kl diveregence results
    inverse (Bool): if True, also gets KL diveregence for the gtex tissue x inversed seer cancer incidence
    """
    for tissue in gtex_samples:
        print(f"Beginning analysis for {tissue}")
        scaled_averages = pd.read_pickle(filepath_or_buffer=f"../data/kl_div/GTEx/{sex}/scaled_averages/{tissue}.pkl")
        
        scaled_medians = pd.read_pickle(filepath_or_buffer=f"../data/kl_div/GTEx/{sex}/scaled_averages/{tissue}.pkl")

        for cancer in seer_cancer_list:
            results_path = os.path.join("../data/kl-div-results", cancer, sex, tissue)
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            # get SEER
            if tissue in ['bladder', 'kidney_medulla', 'cervix_endocervix', 'kidney_cortex', 'cervix_ectocervix', 'minor_salivary_gland', 'fallopian_tube']:
                seer = get_scaled_seer(sex, cancer, tissue)
            else:
                seer = np.load(f"../data/kl_div/SEER/{sex}_scaled/{cancer}.npy", allow_pickle=True)
            if inverse == True:
                inv_seer = np.flip(seer)
            
            # get KL divergence
            kl_div_avg = calc_kl_divergence(scaled_averages, seer)
            kl_div_median = calc_kl_divergence(scaled_medians, seer)

            if inverse == True:
                inv_kl_div_avg = calc_kl_divergence(scaled_averages, inv_seer)
                inv_kl_div_median = calc_kl_divergence(scaled_medians, inv_seer)
            
            # save KL divergence
            avg_fname = results_path +'/'+ tissue + '_kl_avg_'+sex+'.csv'
            med_fname = results_path +'/'+ tissue + '_kl_med_'+sex+'.csv'
            kl_div_avg.to_csv(avg_fname)
            kl_div_median.to_csv(med_fname)

            if inverse == True:
                avg_fname = results_path +'/inverse_'+ tissue + '_kl_avg_'+sex+'.csv'
                med_fname = results_path +'/inverse_'+ tissue + '_kl_med_'+sex+'.csv'
                inv_kl_div_avg.to_csv(avg_fname)
                inv_kl_div_median.to_csv(med_fname)

            if plot_results == True:
                plot_genes(sex, results_path, tissue, cancer, seer, kl_div_avg, kl_div_median)
                if inverse == True:
                    plot_genes(sex, results_path, tissue, cancer, inv_seer, inv_kl_div_avg, inv_kl_div_median, inverse=True)