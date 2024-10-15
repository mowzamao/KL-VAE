# Top KL Div Genes
Sophie Mowe (23197128)
August 2024

Each excel spreadsheet in this folder contains the 10 genes with the lowest KL divergence for each SEER Cancer Incidence and GTEx Tissue sample dataset.

## Data Dictionary
- Column1: Index (int)
- Cancer: The SEER Cancer Incidence Dataset Used (string)
- Tissue: The GTEx Tissue sample dataset (string)
- Gene: The gene name (string)
- KL Divergence: KL Divergence calculated from the cancer and tissue data (float)

## Female Datasets
Each spreasdheet contains the 10 genes with the lowest KL divergence for the female GTEx tissue sample dataset, where:
- Only tissue samples from female donors were used from the GTEx dataset.
- Only the female SEER cancer incidence per age was used.

### female_avg_inverse_top_genes.xlsx:
- Gene expression for each age group was aggregated through calculating the average expression for each 10-year age group.
- The SEER Cancer Incidence curve was inversed prior to calculating KL Divergence, so that genes which are downregulated with age at the same rate as cancer incidence increases per age group have the lowest KL divergence.

### female_avg_top_genes.xlsx:
Contains the 10 genes with the lowest KL divergence for the female GTEx tissue sample dataset, where:
- Gene expression for each age group was aggregated through calculating the average expression for each 10-year age group.

### female_med_inverse_top_genes.xlsx:
Contains the 10 genes with the lowest KL divergence for the female GTEx tissue sample dataset, where:
- Gene expression for each age group was aggregated through calculating the median expression for each 10-year age group.
- The SEER Cancer Incidence curve was inversed prior to calculating KL Divergence, so that genes which are downregulated with age at the same rate as cancer incidence increases per age group have the lowest KL divergence.

### female_med_top_genes.xlsx:
- Gene expression for each age group was aggregated through calculating the median expression for each 10-year age group.

## Male Datasets
- Only tissue samples from male donors were used from the GTEx dataset.
- Only male cancer incidence over age was used from the SEER dataset.

### male_avg_inverse_top_genes.xlsx:
- Gene expression for each age group was aggregated through calculating the average expression for each 10-year age group.
- The SEER Cancer Incidence curve was inversed prior to calculating KL Divergence, so that genes which are downregulated with age at the same rate as cancer incidence increases per age group have the lowest KL divergence.

### male_avg_top_genes.xlsx:
- Gene expression for each age group was aggregated through calculating the average expression for each 10-year age group.

### male_med_inverse_top_genes.xlsx:
- Gene expression for each age group was aggregated through calculating the median expression for each 10-year age group.
- The SEER Cancer Incidence curve was inversed prior to calculating KL Divergence, so that genes which are downregulated with age at the same rate as cancer incidence increases per age group have the lowest KL divergence.

### female_med_top_genes.xlsx:
- Gene expression for each age group was aggregated through calculating the median expression for each 10-year age group.

## NOTES
- SEER Neuroblastoma has 0 incidence per 100,000 for female and male individuals in the GTEx Age Range
- SEER Pleura has almost 0 incidence per 100,000 for individuals in the GTEx Age Range: Genes that are silenced are the best match
- GTEx kidney_medulla only has one sample from a female donor, so a distribution for gene expression cannot be created. Thus, there is no KL divergence data for female kidney medulla.