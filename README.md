# ASL
RandomForest Classifier for ASL

Step 1. collect individual landmarks into separate .csv files using 01.collect.py. Do this for each label you wish to collect
Step 2a. Create a folder gesture_csvs and move all the individual .csv into this folder
Step 2b. run the 02.clean_and_merge.py script to merge all .csv into a single merged_gestures_balanced.csv file
Step 3. run the 03.train.py, this will create gesture_model.pkl file
Step 4. try realtime inference using 04.infer.py

