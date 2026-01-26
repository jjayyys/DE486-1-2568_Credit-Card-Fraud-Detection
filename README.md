# Credit Card Fraud Detection Pipeline
This project is the semester project of "DE486: Data Engineering Integrations (1/2568)" in B.Sc. in Data Engineering at Srinakharinwirot University.

# Project Description
This project is an end-to-end ELT (Extract, Load, Transform) data pipeline designed to automate the processing of financial transaction data for fraud detection analysis.

It orchestrates the entire lifecycle of data—from fetching raw datasets from Kaggle to storing them in a data warehouse (MySQL), performing feature engineering, and generating an automated data quality dashboard.

# Background & Problem Statement
**The Pain Points**
- Dirty Data: Financial datasets often contain duplicate entries and inconsistent formats that ruin model accuracy.
- Manual Effort: Manually downloading CSVs, cleaning them, and loading them into databases is repetitive and error-prone.
- Unscaled Features: Machine learning models (like Logistic Regression or Neural Networks) struggle with unscaled data, such as "Transaction Amount" ranging from $0 to $10,000+.
- Lack of Visibility: It is difficult to verify if the data distribution (Fraud vs. Normal) remains consistent after processing without visualization tools.

**The Solution**
A containerized automated pipeline that:
1. Ingests data directly from the source (Kaggle API).
2. Cleans duplicates and Transforms features automatically.
3. Validates the output with a generated health-check dashboard.
4. Isolates dependencies using Docker for reproducibility.

# Tech Stack
- **Orchestration:** Apache Airflow (v2.8.1)

- **Containerization:** Docker & Docker Compose

- **Data Warehouse:** MySQL (v8.0)

- **Language:** Python 3

- **Libraries:**

    - **Data Manipulation:** Pandas, SQLAlchemy, PyMySQL

    - **Visualization:** Matplotlib, Seaborn

    - **Machine Learning:** Scikit-learn (StandardScaler)

    - **API:** Kaggle API

# How It Works (Pipeline Architecture)
The pipeline is defined in `dags/fraud_pipeline.py` and executes the following steps sequentially:

**Step 1: Extract & Load (EL)**
- **Trigger:** The Airflow DAG `fraud_detection_docker_pipeline_kaggle` runs on a daily schedule.
- Action:
    - Authenticates with Kaggle using `kaggle.json`.
    - Downloads the `mlg-ulb/creditcardfraud` dataset.
    - Loads raw data into the MySQL table `raw_transactions`.

**Step 2: Transform (T)**
- Cleaning: Removes duplicate records (approx. 1,000+ duplicates in this dataset).
- Feature Engineering:
    - Time: Converts raw seconds into `day` and `hour` features to help models capture temporal fraud patterns.
    - Scaling: Applies `StandardScaler` to the `Amount` column to normalize the distribution.
    - ID Generation: Creates a unique `transaction_id`.
- Storage: Saves processed data into two specialized tables:
    - `transactions_processed`: Full cleaned dataset.
    - `transaction_features`: Feature-ready dataset for ML training.

**Step 3: Visualization & Validation**
- Dashboarding: Automatically generates a 2x2 visualization grid (`data_comparison.png`) to verify:
    - Data volume integrity (Raw vs. Cleaned).
    - Fraud patterns by hour.
    - Class imbalance (Fraud vs. Normal).
    - Scaled Amount distribution.

# How to Setup & Use
**Prerequisites**
- Docker Desktop installed.
- A Kaggle account and an API Token (kaggle.json).

**Installation Steps**
1. Clone the Repository
```
    git clone <your-repo-url>
    cd <project-folder>
```

2. Add Kaggle Credentials
- Place your kaggle.json file inside the data/ folder.
- Note: The pipeline looks for this file at /opt/airflow/data/kaggle.json inside the container.

3. Start the Services Run the following command to build the Airflow image and start MySQL:
```
    docker-compose up -d --build
```
This initializes the database and creates a default admin user.

4. Access Airflow UI
- Open your browser and go to `http://localhost:8080`.
- Username: `admin`
- Password: `admin`

5. Run the Pipeline
- Find the DAG named `fraud_detection_docker_pipeline_kaggle`.

- Toggle the switch to **"Unpause"**.

- Click the **"Trigger DAG"** button (Play icon).

6. Check Results
- Once finished, check the `data/` folder for the output dashboard: `data_comparison.png`.
- You can also connect to the MySQL database on port `3307` to query the data.

# Future Improvements
- **Model Training:** Add a task to train a Random Forest or XGBoost model on the `transaction_features` table.

- **Alerting:** Configure Airflow email/Slack alerts if the data volume drops or if the "Fraud" class disappears.

- **Cloud Deployment:** Move the infrastructure to AWS (EC2/RDS) or Google Cloud (Composer/Cloud SQL) for production readiness.
