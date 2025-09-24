This is an end-to-end PyTorch project for predicting sales based on advertising budgets.
We use a synthetic dataset (TV, Radio, Online ad budgets) to train a linear regression model.

The project demonstrates:
Dataset creation with NumPy & Pandas
Exploratory Data Analysis (EDA)
Data preprocessing (scaling, train-test split)
Model building in PyTorch
Training, evaluation, and visualization

⚡ Features

🔹 Synthetic Dataset: 5000 samples, 3 features, 1 target

🔹 Preprocessing: Standardization (mean=0, std=1)

🔹 Model: Simple linear regression (nn.Linear)

🔹 Loss Function: Mean Squared Error (MSE)

🔹 Optimizer: Adam

🔹 Evaluation: Test MSE, RMSE, and scatter plot of predicted vs actual sales

🔹 Save/Load Model: Store the trained model for future use

🛠 Project Steps
Dataset Creation
Features: TV_Ads, Radio_Ads, Online_Ads

Target: Sales
Added some random noise to simulate real-world data
Exploratory Data Analysis (EDA)

Checked dataset shape, statistics, missing values
Visualized feature distributions and correlations
Preprocessing
Split into train (80%) and test (20%) sets
Standardized features and target using StandardScaler
Convert to PyTorch Tensors
Created TensorDataset and DataLoader for batch training
Model Building
Defined a SalesModel class with nn.Linear(3,1)
Training
Used MSE loss and Adam optimizer
Trained for 100 epochs with batch size 64
Evaluation & Visualization
Predicted on test data
Calculated MSE & RMSE
Plotted predicted vs actual sales
Save & Load Model
Saved trained model using torch.save
Reloaded model for inference using torch.load

📊 Example Results
Test MSE: 931,574

Test RMSE: ~965 units

Scatter plot shows predicted sales are closely aligned with actual sales. ✅
