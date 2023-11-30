# IRF_ML_SARSCoV2_pub
Classical machine learning to classify SARs-CoV-2 vs. Mock in NHP

When referencing this work, please cite:

Chu, W. T., Castro, M. A., Reza, S., Cooper, T. K., Bartlinski, S., Bradley, D., Anthony, S. M., Worwa, G., Finch, C. L., Kuhn, J. H., Crozier, I., & Solomon, J. (2023). Novel machine-learning analysis of SARS-CoV-2 infection in a subclinical nonhuman primate model using radiomics and blood biomarkers. Scientific Reports, 13(1), 19607. https://doi.org/10.1038/s41598-023-46694-9


## Project Summary

### Goal
- Determine features most relevant to prediction of SARS-CoV-2 vs. Mock
- Build a ML model for automatic classification of SARS-CoV-2 vs. Mock
    - Build a foundation for future work in severity classification & translation to humans

### Data
- 12 SARs-CoV-2 & 8 Mock Cynomolgus monkeys
- 4 time points: BL, 2, 4, & 6 days post-infection
- Radiomics measures calculated off of CT scan of lung & whole body
- Clinical pathology and immunology measures calculated off of blood sample analyses

### Approach & Notebook Organization
1) Preprocessing\
    a) Reshape & reformat radiomics data\
        - Calculate change from baseline\
    b) Reshape & reformat clinical pathology data\
        - Calculate change from baseline\
    c) Reshape & reformat immunology data\
        - Calculate change from baseline\
    d) Merge radiomics and clinical pathology data\
2) Run exploratory analyses
    - Classic statistics
    - Data Visualization
3) Feature Selection
    - Relevance threshold (f-stat, MI, chi2)
    - Minimum redundancy, maximum relevance (mRMR)
4) Machine Learning
    - Models
    - Evaluation of performance
    - Effect of confounding variables
5) Comparison of model performance

### Citations

#### mRMR-Permute uses mRMR by Peng et al., please see the below citation for more information:
- Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), 1226–1238. https://doi.org/10.1109/TPAMI.2005.159
