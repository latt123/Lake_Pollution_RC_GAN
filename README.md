# Lake_Pollution_RC_GAN

Notebook and scripts of an RCGAN model created for a project in predicting lake pollution.
Project completed as part of INRAE-MISTEA Internship by Lucy Attwood May to August 2023.

To run model and explanations of RCGAN structure, see **run_model** notebook.  

So the notebook can be shared, it has been changed to use a different data set (unrelated to lake pollution) that is public. The data set is [DJIA 30 Stock Time Series](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231/)

Project uses **Julia**

##  Structure of the repository
* saved_models - generator and discriminator saved during training
* saved_models_gif - generator and discriminator saved during training for creating GIF
* Close_forecast_50_epochs.gif - GIF showing evolution of forecast by generator over 50 epochs
* GAN_general_drawio - diagram of 'classic' GAN structure
* Manifest.toml
* Project.toml
* data_set.csv - data set used in notebook, **NOT** the lake pollution data set used in project
* funcions_GAN.csv - functions to create and train the GAN model (Julia)
* functions_forecasting - functions to create forecasts using the generators (Julia)
* **run_model.ipynb** - explanation of the RCGAN model, training the model using the data set and creating a GIF* (Julia)
