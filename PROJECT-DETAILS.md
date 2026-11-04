# Machine Learning & AI Projects Roadmap (33 Projects)

A categorized roadmap of 33 projects, from classical machine learning to
deep learning, Transformers, generative AI, reinforcement learning, and
MLOps.

> **Difficulty is an estimate of implementation depth, not a
> guarantee.** Reproducing a tutorial is easier than building a robust
> system, running controlled experiments, and improving a baseline.
> Dataset access, licensing, compute requirements, and project scope can
> change.

## How to use this roadmap

For each project: 1. Read the problem statement and define the
prediction or generation target. 2. Download the dataset from the linked
source and read its licence/terms. 3. Build a simple baseline before
using complex models. 4. Use leakage-safe train/validation/test splits.
5. Evaluate with metrics appropriate to the task. 6. Document errors,
limitations, reproducibility, and next experiments.

## Difficulty levels

  ---------------------------------------------------------------------------
  Level                   Meaning                 Typical work
  ----------------------- ----------------------- ---------------------------
  **1 --- Beginner**      Classical ML            EDA, cleaning, feature
                          foundations             engineering,
                                                  regression/classification

  **2 --- Intermediate**  Broader ML methods and  Clustering, NLP, audio
                          unstructured data       features, introductory RL

  **3 --- Advanced**      Deep learning and       CNNs, Transformers,
                          applied ML systems      fine-tuning, PyTorch,
                                                  distributed data

  **4 --- Expert**        Complex architectures   ASR, GANs, multimodal
                          or production/research  recommenders, model-based
                          workflows               RL, MLOps

  **5 --- Specialized /   Modern                  LLM adaptation, medical
  Frontier**              foundation/generative   multimodal models,
                          models or substantial   diffusion, production
                          systems work            deployment
  ---------------------------------------------------------------------------

## Dataset portals

These portals are useful when an exact dataset is not linked: - [Kaggle
Datasets](https://www.kaggle.com/datasets) --- competitions and
downloadable datasets; account may be required. - [UCI Machine Learning
Repository](https://archive.ics.uci.edu/datasets) --- curated classical
ML datasets. - [Hugging Face Datasets](https://huggingface.co/datasets)
--- NLP, audio, image, and multimodal datasets. - [Google Research
Datasets](https://research.google/resources/datasets/) --- research
datasets. - [OpenML](https://www.openml.org/search?type=data) ---
machine-learning datasets and benchmark tasks. -
[GitHub](https://github.com/) --- project-specific datasets and scripts;
verify provenance and licences. - [Roboflow
Universe](https://universe.roboflow.com/) --- computer-vision datasets;
check each dataset's licence. - [Mozilla Common
Voice](https://commonvoice.mozilla.org/en/datasets) ---
community-contributed speech datasets.

------------------------------------------------------------------------

# Level 1 --- Beginner: Classical Machine Learning (Projects 1--6)

**Prerequisites:** Python, NumPy, Pandas, Matplotlib/Seaborn, basic
statistics, scikit-learn.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------
                 \# Project       Category / task  Main concepts     Dataset source
  ----------------- ------------- ---------------- ----------------- ---------------------------------------------------------------------------------------
                  1 Predict       Regression /     Linear models,    [UCI Appliances Energy
                    energy        forecasting      tree models, time Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction);
                    consumption                    features,         [Kaggle energy datasets](https://www.kaggle.com/datasets?search=energy+consumption)
                                                   temporal          
                                                   validation        

                  2 Predict       Tabular          EDA, categorical  [Kaggle Medical Cost Personal Dataset
                    insurance     regression       encoding,         search](https://www.kaggle.com/datasets?search=medical+cost+personal+dataset)
                    charges                        scaling,          
                                                   regression        
                                                   metrics           

                  3 Predict       Binary           Missing-value     [UCI Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval)
                    credit-card   classification   handling,         
                    approvals                      encoding,         
                                                   scaling,          
                                                   imbalance,        
                                                   GridSearchCV      

                  4 Wine quality  Classification / Logistic          [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
                    prediction    regression       regression,       
                                                   random forest,    
                                                   class imbalance,  
                                                   MAE/RMSE or F1    

                  5 Predictive    Classification / Crop              [Kaggle Crop Recommendation
                    modelling for feature          recommendation,   search](https://www.kaggle.com/datasets?search=crop+recommendation); [UCI
                    agriculture   selection        feature ranking,  datasets](https://archive.ics.uci.edu/datasets)
                                                   model comparison  

                  6 Store sales   Time-series      Date features,    [Kaggle Store Sales --- Time Series
                    forecasting   forecasting      lags, rolling     Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
                                                   means,            
                                                   backtesting,      
                                                   ensembles         
  ----------------------------------------------------------------------------------------------------------------------------------------------------------

**Move to Level 2 when:** you can build reusable preprocessing
pipelines, avoid data leakage, select appropriate metrics, compare
baselines, and explain residuals/errors.

# Level 2 --- Intermediate: Unsupervised ML, NLP, Audio and Introductory RL (Projects 7--14)

**Prerequisites:** Level 1 skills, probability/statistics, feature
engineering, basic linear algebra.

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                 \# Project        Category / task             Main concepts           Dataset source
  ----------------- -------------- --------------------------- ----------------------- ------------------------------------------------------------------------------------------------------
                  7 Reveal         NLP + clustering            Text cleaning, TF-IDF,  [Google Play Store reviews datasets](https://www.kaggle.com/datasets?search=google+play+store+reviews)
                    categories                                 K-means, cluster        
                    found in data                              interpretation          

                  8 Word frequency NLP / text mining           Web/text acquisition,   [Project Gutenberg --- Moby-Dick](https://www.gutenberg.org/ebooks/2701)
                    in *Moby-Dick*                             tokenization, stop      
                                                               words, NLTK, frequency  
                                                               analysis                

                  9 Facial         Computer vision /           Image features,         [LFW
                    recognition    classification              classification,         dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html);
                    with                                       identity-vs-detection   [Kaggle face datasets](https://www.kaggle.com/datasets?search=face+recognition)
                    supervised                                 distinction             
                    learning                                                           

                 10 What makes a   Tabular + text              Mixed feature types,    [Goodreads datasets search](https://www.kaggle.com/datasets?search=goodreads+books); [UCSD Book
                    good book?     classification/regression   text vectorization,     Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
                                                               pipelines, model tuning 

                 11 Cluster        Unsupervised learning       K-means, scaling,       [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/); [R datasets
                    Antarctic                                  elbow/silhouette, PCA   repository](https://github.com/allisonhorst/palmerpenguins)
                    penguin                                                            
                    species                                                            

                 12 Taxi route     Reinforcement learning      Gymnasium Taxi-v3,      [Gymnasium Taxi tutorial](https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/);
                    optimization                               Q-learning,             [Taxi-v3 environment](https://gymnasium.farama.org/environments/toy_text/taxi/)
                                                               epsilon-greedy, reward  
                                                               curves                  

                 13 Breast cancer  Medical tabular             Logistic regression,    [UCI Breast Cancer Wisconsin
                    detection      classification              SVM, trees, recall,     (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic);
                                                               ROC-AUC, calibration    [scikit-learn
                                                                                       dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

                 14 Speech emotion Audio classification        Librosa, MFCCs,         [RAVDESS](https://zenodo.org/records/1188976);
                    recognition                                spectral features, MLP, [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
                                                               speaker-independent     
                                                               splits                  
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Move to Level 3 when:** you can explain TF-IDF and embeddings,
validate clustering quality, extract meaningful audio features, and
implement tabular Q-learning with a measured learning curve.

# Level 3 --- Advanced: Deep Learning, Transformers and Applied ML Systems (Projects 15--21)

**Prerequisites:** PyTorch or TensorFlow/Keras, neural-network
fundamentals, backpropagation, optimization, GPU basics, validation
discipline.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------
                 \# Project          Category / task  Main concepts              Dataset source
  ----------------- ---------------- ---------------- -------------------------- ----------------------------------------------------------------------------------
                 15 Service desk     NLP + deep       Tokenization, embeddings,  [Kaggle customer-support ticket
                    ticket           learning         PyTorch classifier,        datasets](https://www.kaggle.com/datasets?search=customer+support+ticket);
                    classification                    weighted loss, F1          [Hugging Face datasets](https://huggingface.co/datasets?search=support%20tickets)

                 16 Rick Sanchez bot Conversational   Dialogue preprocessing,    [Rick and Morty dialogue dataset
                                     NLP / generative DialoGPT, Transformer      search](https://www.kaggle.com/datasets?search=rick+and+morty+dialogue); [Hugging
                                     model            fine-tuning, generation,   Face datasets search](https://huggingface.co/datasets?search=rick%20and%20morty);
                                                      Gradio                     [DialoGPT model](https://huggingface.co/microsoft/DialoGPT-medium)

                 17 E-commerce       Computer vision  CNN, augmentation,         [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist); [Kaggle fashion
                    clothing         / image          transfer learning,         product images](https://www.kaggle.com/datasets?search=fashion+product+images)
                    classifier       classification   confusion matrix           

                 18 Detect traffic   Computer vision  CNN, image preprocessing,  [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html); [Kaggle traffic-sign
                    signs                             augmentation,              datasets](https://www.kaggle.com/datasets?search=traffic+sign+recognition)
                                                      classification/detection   
                                                      distinction                

                 19 Demand           Distributed ML / PySpark, lag features,     [Kaggle Store
                    forecasting      forecasting      rolling statistics, time   Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting);
                                                      split, MAE/RMSE/MAPE       [M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

                 20 Predicting       Time-series ML + Time-aware splits,         [UK Met Office data](https://www.metoffice.gov.uk/research/climate/maps-and-data);
                    temperature in   experiment       lag/rolling features,      [Kaggle London weather
                    London           tracking         MLflow, model comparison   search](https://www.kaggle.com/datasets?search=london+weather)

                 21 Reinforcement    Game-playing RL  DQN/PPO, reward design,    [Kaggle Connect X](https://www.kaggle.com/competitions/connectx)
                    learning for                      self-play/competition      
                    Connect X                         evaluation                 
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------

**Rick Sanchez bot note:** this is a character-style dialogue-generation
exercise. It does not create a reasoning agent or a new foundation
model. The exact dialogue dataset must be checked for completeness,
provenance, and usage rights. A small fine-tuning run is a useful
learning exercise; evaluate coherence, repetition, safety, and held-out
dialogue performance.

**Move to Level 4 when:** you can train and debug neural models, explain
overfitting and optimization, use reproducible experiments, and report
more than accuracy alone.

# Level 4 --- Expert: Research Architectures and Production Workflows (Projects 22--28)

**Prerequisites:** strong deep learning, experiment design, advanced
Python, GPU training, evaluation methodology, and relevant domain
fundamentals.

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                 \# Project           Category /    Main concepts       Dataset source
                                      task                              
  ----------------- ----------------- ------------- ------------------- ----------------------------------------------------------------------------------------------------
                 22 Multilingual ASR  Speech /      XLS-R/Wav2Vec2,     [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets);
                    with Transformers sequence      audio feature       [CoVoST](https://ai.meta.com/tools/covost/); [Hugging Face ASR
                                      modelling     extraction, CTC,    datasets](https://huggingface.co/datasets?task=automatic-speech-recognition)
                                                    WER, fine-tuning    

                 23 One-shot face     Generative    StyleGAN, GAN       [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset);
                    stylization       vision / GAN  inversion, paired   [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html); [JoJoGAN
                                                    data, perceptual    repository](https://github.com/mchong6/JoJoGAN)
                                                    quality             

                 24 H&M personalized  Recommender   Collaborative       [H&M Personalized Fashion
                    fashion           systems +     filtering, ranking, Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
                    recommendations   multimodal ML text/image          
                                                    features, retrieval 

                 25 Customer support  Speech + NLP  ASR, diarization,   [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/); [Mozilla Common
                    call analysis     pipeline      sentiment, NER,     Voice](https://commonvoice.mozilla.org/en/datasets); [Hugging Face audio
                                                    embeddings,         datasets](https://huggingface.co/datasets?task=automatic-speech-recognition)
                                                    semantic search     

                 26 Monitoring a      ML monitoring Precision-recall,   [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud);
                    financial fraud   / anomaly     PR-AUC,             [OpenML](https://www.openml.org/search?type=data)
                    model             detection     calibration,        
                                                    data/concept drift, 
                                                    explainability      

                 27 Reinforcement     Advanced RL / MuZero, learned     [Arcade Learning Environment](https://ale.farama.org/);
                    learning agent    model-based   dynamics, search,   [MinAtar](https://github.com/kenjyoung/MinAtar); [Atari
                    for Atari 2600    planning      replay, distributed datasets/search](https://www.google.com/search?q=Atari+2600+reinforcement+learning+datasets)
                                                    training            

                 28 End-to-end MLOps  ML platform   Training pipeline,  [TensorFlow Datasets](https://www.tensorflow.org/datasets); [Google Cloud ML
                                      engineering   Docker, Kubernetes, quickstarts](https://cloud.google.com/architecture); [Databricks MLOps
                                                    CI/CD, model        quickstart](https://github.com/databricks-solutions/mlops-quickstart)
                                                    registry,           
                                                    monitoring          
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Research warning:** MuZero is a substantial research implementation.
Reproducing the paper faithfully requires considerably more than
training a basic DQN agent. Start with a small environment before
scaling to Atari.

# Level 5 --- Specialized / Frontier: Foundation Models and Production AI (Projects 29--33)

**Prerequisites:** strong DL, Transformers, GPU memory management, model
evaluation, fine-tuning, and software engineering. Some projects require
rented GPU compute.

  ---------------------------------------------------------------------------------------------------------------------------------------
                 \# Project       Category /    Main concepts     Dataset / model source
                                  task                            
  ----------------- ------------- ------------- ----------------- -----------------------------------------------------------------------
                 29 Fine-tuning   LLM           Tokenization,     [OpenAI GPT-OSS models](https://huggingface.co/openai); [Hugging Face
                    GPT-OSS       adaptation    chat templates,   datasets](https://huggingface.co/datasets?search=medical%20qa);
                                                supervised        [MedQuAD](https://github.com/abachaa/MedQuAD)
                                                fine-tuning,      
                                                evaluation        

                 30 Fine-tuning   Medical       Vision-language   [MedGemma
                    MedGemma on   multimodal    models, image     documentation](https://deepmind.google/models/gemma/medgemma/); [BraTS
                    brain MRI     learning      preprocessing,    challenge datasets](https://www.med.upenn.edu/cbica/brats/); [Medical
                                                fine-tuning,      Segmentation Decathlon](http://medicaldecathlon.com/)
                                                ROC-AUC, clinical 
                                                validation        

                 31 Fine-tuning   Generative    Diffusers,        [Hugging Face image
                    SDXL with     vision /      DreamBooth, LoRA, datasets](https://huggingface.co/datasets?task=image-generation);
                    DreamBooth    diffusion     VRAM              [Diffusers DreamBooth
                    and LoRA                    optimization,     docs](https://huggingface.co/docs/diffusers/training/dreambooth)
                                                image evaluation  

                 32 Song          Generative    Latent diffusion, [Hugging Face audio
                    generation    audio         conditioning,     datasets](https://huggingface.co/datasets?task=audio-generation);
                    with latent                 audio generation, [MusicBench](https://huggingface.co/datasets?search=MusicBench);
                    diffusion                   chunked decoding  [Google Research datasets](https://research.google/resources/datasets/)

                 33 Deploying an  MLOps /       GitHub Actions,   [TensorFlow Datasets](https://www.tensorflow.org/datasets); [UCI
                    ML            software      tests, data       datasets](https://archive.ics.uci.edu/datasets); [Databricks MLOps
                    application   systems       validation, model quickstart](https://github.com/databricks-solutions/mlops-quickstart)
                    to production               versioning,       
                                                automated         
                                                deployment        
  ---------------------------------------------------------------------------------------------------------------------------------------

**Important data and safety notes:** - **Medical imaging:** use
appropriately licensed, de-identified research datasets. Do not treat a
prototype as a diagnostic device. Brain MRI labels, segmentation masks,
and classification labels are different tasks; confirm the dataset
matches the intended target. - **Medical Q&A:** verify source quality
and licensing. A medical chatbot is not a substitute for clinical
advice. - **Face datasets:** check consent, privacy, and permitted use
before training or publishing. - **Music/audio:** check copyright and
dataset terms before training or distributing generated outputs. -
**Personal-photo fine-tuning:** obtain consent from the people depicted.

------------------------------------------------------------------------

# Recommended learning order

This is a dependency-based sequence; it is not necessary to complete
every project.

  -----------------------------------------------------------------------
  Phase                   Recommended projects    Exit requirement
  ----------------------- ----------------------- -----------------------
  A --- Classical ML      1, 3, 4, 5              Clean pipelines,
                                                  validation, feature
                                                  engineering, metrics

  B --- Time series and   6, 7, 11                Backtesting,
  unsupervised ML                                 clustering, PCA,
                                                  interpretation

  C --- NLP and audio     8, 10, 14               Text processing,
  foundations                                     vectorization, audio
                                                  feature extraction

  D --- Deep learning     15, 17, 18              PyTorch/Keras training,
                                                  regularization,
                                                  transfer learning

  E --- Transformers and  16, 22, 25              Attention,
  speech                                          tokenization,
                                                  fine-tuning, WER and
                                                  inference

  F --- Specialized       21, 23, 24, 27, 29, 30, Architecture
  modelling               31, 32                  understanding,
                                                  controlled experiments,
                                                  domain evaluation

  G --- Production        19, 20, 26, 28, 33      Tracking,
  systems                                         reproducibility, CI/CD,
                                                  monitoring, deployment
  -----------------------------------------------------------------------

## Suggested portfolio strategy

Build **4--6 deep projects**, rather than 33 shallow tutorial copies:

1.  **Classical ML:** one strong tabular project with a clean sklearn
    pipeline and explainability.
2.  **NLP:** a ticket classifier or dialogue model with error analysis.
3.  **Computer vision:** a CNN/transfer-learning project with a
    carefully designed evaluation.
4.  **LLM:** a fine-tuning or retrieval-augmented project with baseline
    comparisons.
5.  **Research-oriented:** reproduce a paper or test a well-defined
    hypothesis with ablations.
6.  **MLOps:** deploy one model with tests, experiment tracking,
    monitoring, and documented rollback.

For each portfolio project, publish: - Problem statement and
motivation - Dataset source, licence, and preprocessing - Baseline and
proposed approach - Metrics and evaluation protocol - Ablation studies
or error analysis - Reproducible setup and inference instructions -
Limitations and future work

## Dataset acquisition checklist

Before downloading or using a dataset, verify:

-   [ ] The source is the original publisher or a trusted mirror.
-   [ ] The licence permits your intended research, academic, or
    commercial use.
-   [ ] The train/test split and labels match the task.
-   [ ] There is no target leakage or duplicate contamination.
-   [ ] Personal or sensitive information is handled appropriately.
-   [ ] Dataset version and download date are recorded.
-   [ ] Large datasets fit your storage and compute budget.

## Suggested repository structure

``` text
ml-projects/
├── README.md
├── 01-classical-ml/
│   ├── energy-consumption/
│   ├── credit-approval/
│   └── ...
├── 02-intermediate-ml/
├── 03-deep-learning/
├── 04-research-projects/
├── 05-generative-ai/
└── 06-mlops/
```

## Final advice

A project becomes advanced through the quality of the work, not just the
model name. A well-designed experiment with a strong baseline,
leakage-free evaluation, ablations, and reproducible results is more
valuable than simply fine-tuning a large model and reporting one
accuracy score.
