# Brain-Connectomics
This repository contains pipelines to build brain connectivity matrices using Freesurfer's recon-all metrics.

This work is an effort to replicate and reproduce work from [Morphological Brain Age Prediction using Multi-View Brain Networks Derived from Cortical Morphology in Healthy and Disordered Participants ]([url](https://www.nature.com/articles/s41598-019-46145-4#Abs1)) and [Unsupervised Manifold Learning Using High-Order Morphological Brain Networks Derived From T1-w MRI for Autism Diagnosis]([url](https://www.frontiersin.org/articles/10.3389/fninf.2018.00070/full#B61)https://www.frontiersin.org/articles/10.3389/fninf.2018.00070/full#B61).
In this repository, you will find scripts for building the referenced lower-order and higher-order matrices referenced in the paper, as well as the connectivity (used in the 1st paper) and averaged / concatednated matrices used in the second paper.

First, you will need to run [Freesurfer's recon-all]([url](https://andysbrainbook.readthedocs.io/en/latest/FreeSurfer/FS_ShortCourse/FS_03_ReconAll.html)https://andysbrainbook.readthedocs.io/en/latest/FreeSurfer/FS_ShortCourse/FS_03_ReconAll.html) command on all your subjects. We are using the Desikan-Killiany atlas in the scripts. If you need to use the Destrieux atlas, please modify the variable in the config file accordingly:
```
parcellation: "DKTatlas" #"a2009s" to use the Destrieux atlas
```
Side note: I noticed a fault with recon-all output, some subjects get less ROIs parcellated, some get more. This may be due to the fact we are standardising the subject's scan. A better approach would be to register the standard space to the native space instead. This is a cool approach and can be investigated in [Kong et al. 's work]([url](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Kong2019_MSHBM)https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Kong2019_MSHBM). This is not applied in this piepeline.

### MorphBrainNetwork.py
This script build the subjects' lower-order, higher-order and the collective connectivity matrix. It takes 5 arguments, 3 of which are optional.
```
python MorphBrainNetwork.py --sub_path Data/Subjects/recon-all/ --config_file config.yaml
```
* ```sub_path```: The path to where subject directories are. These directories should include the recon-all output. For instance: _sub_path_/sub-01/stats/rh.aparc.DKTatlas.stats
*  ```sub_df```: Dataframe containing ID, AGE, and Dataset columns.
* ```config_file```: The path to the file containing all the variables needed for the script to run. This includes the hemispheres, the frontal views for the lower order matrix,the number of ROIs, the ROIS, and the parcellation.
* ```lo``` : Construct the lower-order matrix for every subject, it takes y/n. The default is y.
* ```ho``` : Construct the lower-order matrix for every subject, it takes y/n. The default is y.
* ```fe``` : Performs feature extraction to construct the connectivity matrix for all subjects, it takes y/n. The default is y.

### MBN_BrainAge_njobs.py
This script trains various models to predict age based on the averaged and concatenated matrices generated using MorphBrainNetwork.py. This script 5 arguments: 
* ```config_file```: The path to the file containing all the variables needed for the script to run. This includes the main directory,the name of the subjects directory (ex: if your data is under Main_directory/CohortA, then sub_dir would take _CohortA_), the filename of the subjects of interest with their names and age, the model name, and max percentage of features needed by one of the methods of feature selection. Please note that the main directory is where the _Models_ folder will be created to save your models. 
* ```mode```: Mode should be **avg** or **con**. This is to indicate if the model should used the averaged or concatenated views.
* ```hem```: Hemisphere **rh** or **lh**
* ```method```: The method to use to select the features:  **autFeat**, **percentage_max**, **multiple_models**, **RFR**
* ```n_jobs```: Number of cores used for cross_val function to avoid loky errors. Default is 4 CPUs.

### SVM_SIMLR.py
This script uses output from the package [SIMLR]([url](https://github.com/BatzoglouLabSU/SIMLR/tree/SIMLR/MATLAB)https://github.com/BatzoglouLabSU/SIMLR/tree/SIMLR/MATLAB). To perform clustering of data using SIMLR, you need to run the script using the connectivity matrix generated using MorphBrainNetwork.py. Once you have the output, save the results of the clustering in a csv, which will be used in SVM_SIMLR.py
In this stup, we will train multiple SVM's on classifying our patient / healthy subjects. To do this: 
1. Run SIMLR on your two groups separately. You will end up with c number of clusters per group
1. Group the sub-clusters from your two cohorts. For example: (Group1_clusterA, Group2_clusterA), (Group1_clusterB, Group2_clusterA), (Group1_clusterA, Group2_clusterC) etc. You should end up with c² combinations, as explained in the [paper]([url](https://www.frontiersin.org/articles/10.3389/fninf.2018.00070/full#B61)https://www.frontiersin.org/articles/10.3389/fninf.2018.00070/full#B61)
1. Save your pairs and their corresponding matrices in a pkl file.
2. Train a SVM on every pair
3. Use ensemble learning by applying every SVM on an **unseen** datapoint (from another pair) to classify the observation.


