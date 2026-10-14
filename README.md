# Data_Leakage_Autopsy
Data Leakage Autopsy

A hands-on notebook that shows how data leakage inflates a model's score, and how to catch it. It starts with a churn model that scores 97% and works down to an honest 80%, one leak at a time.

Built for the live webinar Data Leakage Autopsy by Girish, The Unlox Academy.

What is data leakage?

Data leakage is when information that will not exist at prediction time gets into training. The model is not predicting the future. It is reading the answer sheet. Nothing crashes and the accuracy looks great, so you only find out after deployment.

Before you use any column, ask: will I have this value at the moment I make the prediction?

The five leaks
#	Leak	Analogy	Effect on the churn model
1	Preprocessing (imputer and scaler) fit on all data before the split	Curving the exam using marks from the exam itself	About 97.0 → 96.9. Small here, because random forests ignore scale
2	A proxy column (cancellation_ticket_opened) that only exists after the customer decides to leave	Reading the answer from the answer sheet	96.9 → 89.7 after dropping it
3	The same customer appears in both train and test	The same question in the study guide and the exam	89.7 → 85.2 with GroupKFold
4	A random split on time-ordered data	Studying next year's paper	85.2 → 80.0 with a time split (train on months 1-24, test on months 25-36)
5	Feature selection done before cross-validation	Picking the questions after seeing the marks	Noise data: 90% leaky vs 50% honest

The honest score is about 80%. The majority-class baseline is about 63%.

Three tests to catch leakage
Single-feature AUC scan. Flags suspicious columns. The prediction-time question decides whether they are really leaks.
Adversarial validation. Train a classifier to tell train rows from test rows. An AUC near 0.5 means they look alike. A high AUC means the split is not representative (about 0.84 for the time split here).
Label-shuffle test. Shuffle the labels and re-run the pipeline. A good procedure falls to chance. This catches procedure leaks only, not leaky columns or repeated customers.
Repository contents
File	What it is
D_L_A.ipynb	The notebook. Run it top to bottom
churn_data.csv	Synthetic churn dataset: 5,570 rows, 560 customers, 36 months
make_data.py	Generates the dataset (seed 42). Also provides make_noise() for the noise demo
Data Leakage Webinar Deck.pptx	The webinar slides (headings only)
The dataset

The data is synthetic. It has one row per customer per month, and the target is churned_next_8m.

Column	Meaning
customer_id, snapshot_month	Customer and month of the snapshot (1-36)
customer_age, tenure_months	Customer profile
monthly_spend, logins_last_30d, support_tickets_last_30d	Activity (spend and logins have missing values)
plan_type, payment_method, region	Categorical features
discount_flag, uses_new_dashboard	Offers and product usage
cancellation_ticket_opened	The proxy leak. Do not use it as a feature
churned_next_8m	Target: did the customer churn in the next 8 months?

The churn rate is about 37%.

How to run

Google Colab

Upload D_L_A.ipynb, churn_data.csv and make_data.py to the same Colab folder.
Run all cells.

Locally

bash
pip install numpy pandas scikit-learn
jupyter notebook D_L_A.ipynb

To regenerate the data: python make_data.py.

Scores may shift by about 1 point depending on your scikit-learn version.

Notebook walkthrough
Imports and loading the data
Preparing features, groups and months
The stacked leaky model (all leaks together, about 97%)
The noise demo (feature selection before cross-validation)
Adversarial validation
The label-shuffle test
Tools

Python, pandas, NumPy, scikit-learn (Pipeline, GroupKFold, KFold, SelectKBest, RandomForestClassifier, LogisticRegression).

Note on the numbers

The noise demo uses seed 36, which is a clear example. Across 40 seeds the leaky score averages about 85% against 50% for the honest pipeline. Because the data is synthetic, treat the scores as a demonstration of the effect, not as real-world benchmarks.

Author

Girish, Lead Mentor, The Unlox Academy, Bengaluru.
