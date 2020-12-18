# Evaluating and Visualizing the properties of Dropout for GRU

The folder contains the code used to produce graphs and do all the experimentations for the COMPSCI-689
final project. Code was run on google colab as it requires GPU for processing. It will not work on CPU.

First load the code folder in google drive correctly. To train the model, we use different notebooks. In the 
notebook `Visualize Haste Sentiment Model.ipynb` you can run the below command. It should already be present in an
existing cell.

```
!python Sentiment_haste.py --model_file=gru_sentiment_haste_again_d0.0_z0_h128.pt --hidden_units=128 --dropout=0 --zoneout=0
```
Arguments can be changed accordingly to train different models. It also saved the model in the directory.

To produce visualizations for a saved model, run the following command. Plots are saved in the directory.
```
!python Visualize_sentiment_haste.py --model_file=gru_sentiment_haste_d0_z0.pt --visualize_file_suffix=haste_d0_z0
```

To perform hyperparameter tuning for dropconnect, run the following command:
```
!python Sentiment_haste.py --model_file=gru_sentiment_haste_dx_zx.pt --tune_dropconnect=True
```

The `dataset` folder should have the training, test and validation data in csv format. To test on adversarial 
test input, replace the `test.csv` with your adversarial input file.

The list of files which we wrote for this project:
1. Sentiment_haste.py
1. Sentiment.py
1. Visualize_sentiment_haste.py
1. Visualize_sentiment.py
1. SentimentAnalysis/GRUHaste.py
1. SentimentAnalysis/GRUSentiment.py
1. SentimentAnalysis/Explorer.py
1. All ipython notebooks in this folder

Other files were part of an existing project.

### Report
The project can be found under `report/Evaluating and Visualizing the properties of Dropoutfor GRU.pdf`.