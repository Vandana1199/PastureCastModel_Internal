# PastureCastModel_Internal
Backend_Modeling code files


Maintian the below Structure for future model upgrades

Step 1: Column names
features = ['MeanHeight(mm)', 'NDVI_mean', 'GNDVI_mean', "SAVI_mean", "EVI_mean","NDRE_mean","CLRE_mean","SRre_mean", "JulianDate"]
target = 'Biomass(kg/ha)'

Step 2: preprocessing Pipeline
Add any preprocessing step into the pipeline before passing it to the model

Step3: Structured file 
Maintaint the file structure and do not keep any code cells after model implementation, example no learning curves, no feature importances etc.,
