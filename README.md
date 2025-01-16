cMIL: Deep learning-based bone marrow cytology analysis in metastasis detection and prognosis prediction of tumor

#### Abstract
Bone marrow (BM) is the most common site of metastatic disease at diagnosis and a frequent site of relapse in various adult and pediatric cancers, such as breast cancer, prostate cancer, and neuroblastoma. We developed a deep learning-based image analysis from Wright-Giemsa-stained BM cytology images for metastasis detection and prognosis prediction. We implied a weakly-supervised multiple instance learning (MIL) model, with features extracted using CNN architecture followed a quality supervision process, referred to as cMIL. Additionally, we constructed a risk stratification model to make individualized predictions of survival outcomes based on BM cytological features. The proposed cMIL system demonstrates significant potential in computational cytology, offering a promising tool for improving BM metastasis detection, risk stratification, and personalized treatment planning for tumor patients in clinical practice, highlighting its potential to streamline medical research and improve predictive accuracy.

#### Introduction
The complexity and heterogeneous nature of the BM, with its diverse cell lineages in various stages of differentiation, make it a challenging organ for diagnostics. To improve the quality and efficiency of metastasis identification in BM smears, we have developed a DL-based image analysis using Wright-Giemsa-stained BM cytology images. We integrated a weakly-supervised MIL model with CNNs architecture to extract features from BM cytological WSIs. This model, referred to as cMIL, incorporates a crucial quality supervision process that selects qualified patches from BM cytological images before feature extraction, ensuring the accuracy and reliability of the downstream analysis. Additionally, we constructed a risk stratification model to predict individualized survival outcomes based on BM cytological features.

#### System Architecture
cMIL architecture is designed to support Wright-Giemsa-stained BM cytology images for metastasis detection and prognosis prediction. The pipeline consists of several key components:

1.Data Source: Wright-Giemsa-stained BM cytology images.

2.Quality Supervision: Evaluate the quality of BM digital cytological images before feature extraction, which was pre-trained on both high-quality and low-quality patches of BM cytological images.

3.Feature Extraction Tools: Utilizes deep learning models like Resnet50 to extract relevant features from digital cytology images.

4.Model Training and Evaluation Suite: Provides a range of machine learning and deep learning frameworks to train, validate, and optimize predictive models.

5.Visualization and Interpretation: Offers robust tools for visualizing data, model outputs, and interpretability metrics, aiding in the understanding of model results.

6.Prognosis prediction: A risk stratification model to make individualized predictions of survival outcomes based on BM cytological features.

#### Key Features
1.Data Collection: Harmonizing clinical and cytology (WSI) data.

2.Deep Feature Extraction: Utilizing transfer learning with deep learning models (ResNet50) to extract features from largest ROI sections or entire WSI regions for disease characterization and prediction.
$ \text{Feature Extraction with CNN:} \quad f_i = \text{CNN}_{k}(ROI_i) $

3.Deep Learning for cytology: Training deep learning models for classification tasks using WSI patches, and integrating deep features for comprehensive analysis.
$ \text{Deep Learning Classification:} \quad P(C|X) = \frac{\exp(W \cdot X)}{\sum_{j=1}^{K} \exp(W_j \cdot X)} $

4.Feature Selection: Employing techniques such as correlation analysis (e.g., spearman, pearson), and Lasso for dimensionality reduction and feature importance.
$ \text{Lasso Regression:} \quad \min_{\beta} \left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - x_i \beta)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right) $

5.Model Building: Integrating selected features into machine learning algorithms like logistic regression (LR), tree-based models such as random forests and extremely randomized trees (ExtraTrees), extreme gradient boosting (XGBoost), light gradient boosting machine (LightGBM), and multilayer perceptron (MLP) to construct predictive models.
$ \text{COX Regression:} \quad h(t|X) = h_0(t) \exp(X \beta) $
$ \text{Logistic Regression:} \quad \text{logit}(P(Y=1|X)) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p $

6.Performance Evaluation: Comparing models using metrics like accuracy, AUC, sensitivity, specificity, F1-score, and visualizing performance through ROC, DCA, calibration curves, and confusion matrices.
$ \text{AUC:} \quad \text{AUC} = \int_{0}^{1} TPR(FPR^{-1}(x)) , dx $

7.Model Interpretation: Employing Grad-CAM for deep learning model interpretability.

8.Nomogram Construction: construct a risk stratification model to make individualized predictions of survival outcomes based on BM cytological features and clinical data.

9.Survival Analysis: Conducting COX regression and KM analysis with log-rank tests.
$ \text{KM Estimator:} \quad \hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right) $

#### Details of Deep Learning Training Process
Data preprocessing

All BM smears were well stained using the Wright-Giemsa method, and the preparation quality met the requirements set by the International Committee for Standardization in Hematology. We scanned the stained BM smears at 40× magnification (Plan N 40 ×/0.65 FN22, resolution 0.42 mm) for WSI imaging. The WSIs were then segmented into non-overlapping 512 × 512 pixel patches, and the Vahadane method was applied to normalize the color of these patches. 
During cell classification and metastasis detection, metastatic clusters in the BM were annotated by independent experienced hematopathologist.

Deep learning identifies cytological features in BM cytology image 

This model employed a multi-task learning strategy that combined the filtered patches from the quality supervision module with features extracted by the signature extraction module to perform clinical tasks.
Quality supervision: we proposed a quality supervision module to evaluate the quality of BM digital cytological images before feature extraction, which was pre-trained on both high-quality and low-quality patches of BM cytological images, resulting in a dataset of 10000 patches. Based on the evaluation results, the system proceeded with further analysis by including only the high-quality patches for feature extraction. The computer-aided filter module comprises five initial convolutional blocks, an adaptive average pooling layer in the middle, and a final multilayer perceptron. Thereinto, each convolutional block includes a convolution layer, batch normalization, and a rectified linear unit activation layer. The hyperparameters were configured as follows: 100 epochs, a minibatch size of 32, decoupled weight decay of 0.01, and an initial learning rate of 0. 0001.The optimizer used was Adam, and the loss function was cross-entropy loss. 
Multi-Instance Learning-Based Feature Aggregation: we used DL models to extract BM cytological features and constructed predictive models using AI algorithms. For WSI classification, these DL approaches extracted features from individual patches and aggregated them into slide-level representations for final prediction.

1.Patch-level prediction: Each patch was analyzed using the deep learning model to derive probabilities and labels, denoted as Patchprob and Patchpred. we employed a residual network 50 layers (ResNet50) architecture to extract BM cytological features and generate label predictions along with their respective probabilities for qualified patches. We utilized SGD (Stochastic Gradient Descent) as the optimizer, and softmax cross entropy as the loss function. We adjusted the learning rate using a cosine decay strategy, the number of iteration epochs was 40 and an initial learning rate of 0.01, and a batch size of 64.

2.WSI-level predictions: we used a weakly-supervised MIL algorithm to aggregate dispersed patch-level features to WSI-level features. The Predict Likelihood Histograms (PLH) that map out the predictive probabilities and labels for each slice, offering a probabilistic summary of the prediction landscape. The Bag of Words (BoW) approach, segmenting each image into slices and extracting data to compile seven predictive results per sample, using the Term Frequency-Inverse Document Frequency (TF-IDF) method for analysis. During MIL-based WSI fusion, we combined the PLH and BoW pipelines to represent the collective BM cytological information from a patient's BM aspirate specimen. The final WSI-level prediction served as the representations of the participant for subsequent analytical operations.
Histogram Feature Aggregation: Distinct numbers were treated as "bins" to count occurrences across types. Frequencies of Patchprob and Patchpred in each bin were tallied and normalized using min-max normalization, resulting in Histoprob and Histopred.
Bag of Words (BoW) Feature Aggregation: A dictionary was constructed from unique elements in Patchprob and Patchpred. Each slice was represented as a vector noting the frequency of each dictionary element, with a TF-IDF transformation applied to emphasize informative features. This resulted in a BoW feature representation for each slice, encapsulating both the presence and significance of features, denoted as Bowprob and Bowpred.
Multi Instance Learning Feature Aggregation: We integrated Histoprob, Histopred. Bowprob and Bowpred using a feature concatenation method, combining these into a single comprehensive feature vector: $$ feature_{fusion} = Histo_{prob} \oplus Histo_{pred} \oplus Bow_{prob} \oplus Bow_{pred} $$

3.Feature selection process: we employed the LASSO (Least Absolute Shrinkage and Selection Operator) method to screen features and identify the final WSI-level features from BM cytology. The selected features were then used to develop an AI model using various machine learning methods.

Model Construction and Evaluation

Clinical characteristics of participants were compared using independent sample t-tests for continuous variables and chi-squared (χ²) test for categorical variables. A p-value of less than 0.05 was considered statistically significant.
For metastasis detection, we utilized machine learning algorithms to predict the metastasis status in the BM smears. These algorithms included, logistic regression (LR), tree-based models such as random forests and extremely randomized trees (ExtraTrees), light gradient boosting machine (LightGBM), and multilayer perceptron (MLP). We employed the area under the curve (AUC) as the primary performance metric and evaluated the model's sensitivity, specificity, accuracy, F1 score, positive predictive value (PPV), and negative predictive value (NPV) across various classification thresholds. Furthermore, confusion metrics and decision curve analysis (DCA) were performed to evaluate the reliability and clinical utility of the model. Gradient-weighted Class Activation Mapping (Grad-CAM) was employed for model interpretability. The model that demonstrated the best performance on the test cohort was selected as the optimal model. Data preprocessing and model development were conducted using Python (version 3.7.12) and the deep learning platform Anaconda (version 3) based on the OnekeyAI platform.

For prognosis prediction, overall survival (OS) was defined as the time from the diagnosis of tumor to death from any cause, or to the follow-up cutoff time if no event occurred. We constructed a prognostic model incorporating BM cytological signatures and clinical characteristics. Cox proportional hazards regression analysis was used to identify independent prognostic factors, and the final model for the nomogram was selected using a backward step-down selection process. Nomograms for predicting OS were constructed based on the identified independent prognostic factors. The AUC was used to assess the discrimination ability of the nomogram. Calibration curves were created to visualize the agreement between predicted probabilities and observed survival. The nomogram underwent 500 bootstrap resamples for internal validation. DCA was performed to evaluate the reliability and clinical utility of the nomogram. Kaplan-Meier survival analysis with a log-rank test was performed to compare OS between groups. Prognostic analysis was performed using R software (version 4.2). 

#### Conclusion
The proposed cMIL system demonstrated significant performance in identifying diagnostic metastasis and performing risk stratification from BM cytology images. The workﬂow we developed can streamline BM cytology workflows, reducing turnaround time and minimizing interpersonal variability. In addition, the architecture of this model could be extended to other rare cancers, where BM cytological provides valuable diagnostic insights. This versatile approach could contribute to the broader field of computational pathology, offering new avenues for AI-driven diagnosis and prognostication in a variety of oncological settings.

#### References
### References

1.**Lin, K. S. et al**. Minimal residual disease in high-risk neuroblastoma shows a dynamic and disease burden-dependent correlation between bone marrow and peripheral blood. Transl Oncol 14, 101019 (2021). https://doi.org:10.1016/j.tranon.2021.101019.

2.**Pelizzo, G. et al**. Microenvironment in neuroblastoma: isolation and characterization of tumor-derived mesenchymal stromal cells. BMC Cancer 18, 1176 (2018). https://doi.org:10.1186/s12885-018-5082-2

3.**Hochheuser, C. et al**. The Metastatic Bone Marrow Niche in Neuroblastoma: Altered Phenotype and Function of Mesenchymal Stromal Cells. Cancers (Basel) 12 (2020). https://doi.org:10.3390/cancers12113231

4.**Sai, B. & Xiang, J**. Disseminated tumour cells in bone marrow are the source of cancer relapse after therapy. J Cell Mol Med 22, 5776-5786 (2018). https://doi.org:10.1111/jcmm.13867

5.**Bandyopadhyay, S. et al**. Mapping the cellular biogeography of human bone marrow niches using single-cell transcriptomics and proteomic imaging. Cell 187, 3120-3140.e3129 (2024). https://doi.org:10.1016/j.cell.2024.04.013

6.**Kumar, B. et al**. Acute myeloid leukemia transforms the bone marrow niche into a leukemia-permissive microenvironment through exosome secretion. Leukemia 32, 575-587 (2018). https://doi.org:10.1038/leu.2017.259

7.**HaDuong, J. H. et al**. Interaction between bone marrow stromal cells and neuroblastoma cells leads to a VEGFA-mediated osteoblastogenesis. Int J Cancer 137, 797-809 (2015). https://doi.org:10.1002/ijc.29465

8.**van Eekelen, L., Litjens, G. & Hebeda, K. M**. Artificial Intelligence in Bone Marrow Histological Diagnostics: Potential Applications and Challenges. Pathobiology 91, 8-17 (2024). https://doi.org:10.1159/000529701

9.**Lewis, J. E. & Pozdnyakova, O**. Digital assessment of peripheral blood and bone marrow aspirate smears. International Journal of Laboratory Hematology 45, 50-58 (2023). https://doi.org:10.1111/ijlh.14082

10.**Xu, H. et al**. A whole-slide foundation model for digital pathology from real-world data. Nature 630, 181-188 (2024). https://doi.org:10.1038/s41586-024-07441-w

11.**Ramesh, S. et al**. Artificial intelligence-based morphologic classification and molecular characterization of neuroblastic tumors from digital histopathology. npj Precision Oncology 8 (2024). https://doi.org:10.1038/s41698-024-00745-0

12.**Cai, X., Zhang, H., Wang, Y., Zhang, J. & Li, T**. Digital pathology-based artificial intelligence models for differential diagnosis and prognosis of sporadic odontogenic keratocysts. International Journal of Oral Science 16 (2024). https://doi.org:10.1038/s41368-024-00287-y

13.**Dolezal, J. M. et al**. Slideflow: deep learning for digital histopathology with real-time whole-slide visualization. BMC Bioinformatics 25 (2024). https://doi.org:10.1186/s12859-024-05758-x

14.**Weng, Z. et al**. GrandQC: A comprehensive solution to quality control problem in digital pathology. Nature Communications 15 (2024). https://doi.org:10.1038/s41467-024-54769-y

15.**Zhang, Y. et al**. Histopathology images-based deep learning prediction of prognosis and therapeutic response in small cell lung cancer. npj Digital Medicine 7 (2024). https://doi.org:10.1038/s41746-024-01003-0

16.**Niazi, M. K. K., Parwani, A. V. & Gurcan, M. N**. Digital pathology and artificial intelligence. The Lancet Oncology 20, e253-e261 (2019). https://doi.org:10.1016/s1470-2045(19)30154-8

17.**Verghese, G. et al**. Computational pathology in cancer diagnosis, prognosis, and prediction – present day and prospects. The Journal of Pathology 260, 551-563 (2023). https://doi.org:10.1002/path.6163

18.**Zheng, K. et al**. Deep learning model with pathological knowledge for detection of colorectal neuroendocrine tumor. Cell Reports Medicine 5 (2024). https://doi.org:10.1016/j.xcrm.2024.101785

19.**Iacucci, M. et al**. Artificial Intelligence Enabled Histological Prediction of Remission or Activity and Clinical Outcomes in Ulcerative Colitis. Gastroenterology 164, 1180-1188.e1182 (2023). https://doi.org:10.1053/j.gastro.2023.02.031

20.**Shi, J.-Y. et al**. Exploring prognostic indicators in the pathological images of hepatocellular carcinoma based on deep learning. Gut 70, 951-961 (2021). https://doi.org:10.1136/gutjnl-2020-320930

21.**Jayaraman, P., Desman, J., Sabounchi, M., Nadkarni, G. N.,Sakhuja, A**. A Primer on Reinforcement Learning in Medicine for Clinicians. npj Digital Medicine 7 (2024). https://doi.org:10.1038/s41746-024-01316-0

22.**Wang, X. et al**. A pathology foundation model for cancer diagnosis and prognosis prediction. Nature 634, 970-978 (2024). https://doi.org:10.1038/s41586-024-07894-z

23.**Jiang, Y. et al**. Biology-guided deep learning predicts prognosis and cancer immunotherapy response. Nature Communications 14 (2023). https://doi.org:10.1038/s41467-023-40890-x

24.**Kong, J. et al**. Computer-aided evaluation of neuroblastoma on whole-slide histology images: Classifying grade of neuroblastic differentiation. Pattern Recognition 42, 1080-1092 (2009). https://doi.org:10.1016/j.patcog.2008.10.035

25.**Liu, Y. et al**. Pathological prognosis classification of patients with neuroblastoma using computational pathology analysis. Computers in Biology and Medicine 149 (2022). https://doi.org:10.1016/j.compbiomed.2022.105980

26.**Gheisari, S., Catchpoole, D. R., Charlton, A. & Kennedy, P. J**. Convolutional Deep Belief Network with Feature Encoding for Classification of Neuroblastoma Histological Images. Journal of Pathology Informatics 9 (2018). https://doi.org:10.4103/jpi.jpi_73_17

27.**Shouval, R., Fein, J. A., Savani, B., Mohty, M. & Nagler, A**. Machine learning and artificial intelligence in haematology. British Journal of Haematology 192, 239-250 (2020). https://doi.org:10.1111/bjh.16915

28.**Chen, P. et al**. Detection of Metastatic Tumor Cells in the Bone Marrow Aspirate Smears by Artificial Intelligence (AI)-Based Morphogo System. Frontiers in Oncology 11 (2021). https://doi.org:10.3389/fonc.2021.742395

29.**Claveau, J.-S. et al**. Value of bone marrow examination in determining response to therapy in patients with multiple myeloma in the context of mass spectrometry-based M-protein assessment. Leukemia 37, 1-4 (2022). https://doi.org:10.1038/s41375-022-01779-8

30.**Fu, X., Sahai, E. & Wilkins, A**. Application of digital pathology‐based advanced analytics of tumour microenvironment organisation to predict prognosis and therapeutic response. The Journal of Pathology 260, 578-591 (2023). https://doi.org:10.1002/path.6153

31.**Elsayed, B. et al**. Deep learning enhances acute lymphoblastic leukemia diagnosis and classification using bone marrow images. Frontiers in Oncology 13 (2023). https://doi.org:10.3389/fonc.2023.1330977

32.**Hazra, D., Byun, Y.-C. & Kim, W. J**. Enhancing classification of cells procured from bone marrow aspirate smears using generative adversarial networks and sequential convolutional neural network. Computer Methods and Programs in Biomedicine 224 (2022). https://doi.org:10.1016/j.cmpb.2022.107019

33.**Wu, Y.-Y. et al**. A Hematologist-Level Deep Learning Algorithm (BMSNet) for Assessing the Morphologies of Single Nuclear Balls in Bone Marrow Smears: Algorithm Development. JMIR Medical Informatics 8 (2020). https://doi.org:10.2196/15963
