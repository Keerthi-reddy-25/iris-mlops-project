IRIS NEURAL NETWORK + MLOPS PROJECT (EPOCH VERSION)

Requirements:
- Python 3.10 or later recommended
- pandas, scikit-learn, joblib
- TensorFlow is NOT required

SETUP:
1. Extract this ZIP file.
2. Open the extracted iris_mlops_all_files folder in VS Code.
3. Open Terminal in that folder.
4. Install dependencies:
   python -m pip install pandas scikit-learn joblib

RUN THE WORKFLOW:
1. Train a neural network for 100 epochs:
   python train.py
   Output: iris_model.pkl

2. Test the saved model on new data:
   python test_new_data.py
   Input: new_iris_data.csv
   Output: prediction_results.csv

3. Combine old and new labeled data and retrain:
   python retrain.py
   Output: iris_model_updated.pkl

FILES:
- iris_dataset.csv: 150-row Iris dataset
- new_iris_data.csv: 15 labeled example rows for testing/retraining
- train.py: train/test split, scaling, neural network training for 100 epochs, evaluation and save
- test_new_data.py: predictions and accuracy when actual labels are present
- retrain.py: combine datasets and train an updated model for 100 epochs
- iris_mlops_epochs.py: standalone copy of the epoch-based training script
- README.txt: these instructions

EXAM PRACTICE:
- Remove some rows or edit values in iris_dataset.csv, then run train.py.
- Use test_new_data.py to check predictions on new_iris_data.csv.
- For retraining, add correctly labeled new examples and run retrain.py.
- For fair model comparison, evaluate old and updated models on the same separate validation dataset.
- Accuracy is not guaranteed to be 100%.
- Only calculate accuracy when correct actual labels are available.
