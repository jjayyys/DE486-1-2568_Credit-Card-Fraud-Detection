from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import zipfile

# --- 1. Configuration ---

DB_CONNECTION_STR = 'mysql+pymysql://root:root@mysql_db:3306/transaction_db'
DATA_PATH = '/opt/airflow/data'

# ชื่อ Dataset ใน Kaggle (username/dataset-slug)
KAGGLE_DATASET = 'mlg-ulb/creditcardfraud'
CSV_FILENAME = 'creditcard.csv' # ชื่อไฟล์จริงใน Dataset ของ Kaggle ชื่อนี้
CSV_FILE_PATH = os.path.join(DATA_PATH, CSV_FILENAME)
VIZ_FILE = os.path.join(DATA_PATH, 'data_comparison.png')

# --- 2. ELT Functions ---

def extract_and_load_raw():
    """
    Step 1: Extract (Kaggle API) & Load
    ดึงข้อมูลจาก Kaggle -> Unzip -> โหลดลง MySQL
    """
    print(f"🚀 Starting Step 1: Extract from Kaggle & Load")
    
    # 1. ตั้งค่า Kaggle Config (เพื่อให้หาไฟล์ kaggle.json เจอในโฟลเดอร์ data)
    os.environ['KAGGLE_CONFIG_DIR'] = DATA_PATH
    
    # ตรวจสอบว่ามีไฟล์ kaggle.json หรือไม่
    if not os.path.exists(os.path.join(DATA_PATH, 'kaggle.json')):
        raise FileNotFoundError(f"❌ ไม่พบ 'kaggle.json' ใน {DATA_PATH} กรุณาวางไฟล์ Token ก่อน")

    # 2. Download จาก Kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    try:
        print("Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset '{KAGGLE_DATASET}'...")
        # โหลดมาไว้ที่ DATA_PATH
        api.dataset_download_files(KAGGLE_DATASET, path=DATA_PATH, unzip=True)
        print("✅ Download and Unzip complete.")
        
    except Exception as e:
        print(f"❌ Kaggle API Error: {e}")
        raise e

    # 3. ตรวจสอบไฟล์ CSV (Kaggle dataset นี้ไฟล์ชื่อ creditcard.csv)
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ CSV ที่คาดหวัง: {CSV_FILE_PATH}")

    # 4. อ่าน CSV และโหลดลง DB
    print(f"Reading CSV from {CSV_FILE_PATH}...")
    df = pd.read_csv(CSV_FILE_PATH)
    
    # เปลี่ยนชื่อคอลัมน์ให้เหมือนโค้ดเดิม (เผื่อไฟล์ต้นฉบับชื่อต่างกัน)
    # แต่ Dataset นี้โครงสร้างเหมือนเดิม เป๊ะ
    print(f"✅ Read successfully. Raw Shape: {df.shape}")

    try:
        engine = sqlalchemy.create_engine(DB_CONNECTION_STR)
        print("Uploading to MySQL table 'raw_transactions'...")
        df.to_sql('raw_transactions', engine, if_exists='replace', index=False, chunksize=5000)
        print("✅ Data loaded to 'raw_transactions' successfully.")
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        raise e

def transform_in_db():
    """Step 2: Transform"""
    print("🚀 Starting Step 2: Transform")
    engine = sqlalchemy.create_engine(DB_CONNECTION_STR)

    query = "SELECT * FROM raw_transactions"
    df = pd.read_sql(query, engine)
    print(f"Fetched {len(df)} rows from DB.")
    
    df.columns = df.columns.str.strip()

    # Feature Engineering
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {original_len - len(df)} duplicate rows.")

    df["Time"] = df["Time"].astype(int)
    df["day"] = df["Time"] // (3600 * 24)
    df["hour"] = (df["Time"] // 3600) % 24

    scaler = StandardScaler()
    df['Amount'] = df['Amount'].astype(float)
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df['transaction_id'] = range(1, len(df) + 1)

    cols_meta = ['transaction_id', 'Time', 'day', 'hour', 'Amount', 'Amount_Scaled', 'Class']
    v_columns = [f'V{i}' for i in range(1, 29)]
    cols_features = ['transaction_id'] + v_columns

    df_transactions = df[cols_meta]
    df_features = df[cols_features]

    print("Saving processed tables...")
    df_transactions.to_sql('transactions_processed', engine, if_exists='replace', index=False, chunksize=5000)
    df_features.to_sql('transaction_features', engine, if_exists='replace', index=False, chunksize=5000)
    print("✅ Transformation Completed.")

def visualize_data():
    """
    Step 3: Advanced Visualization for Data Verification
    สร้าง Dashboard ตรวจสอบข้อมูลแบบละเอียด (4 กราฟ)
    """
    print("🚀 Starting Step 3: Advanced Visualization")
    engine = sqlalchemy.create_engine(DB_CONNECTION_STR)

    # 1. Fetch Data
    with engine.connect() as conn:
        raw_count = conn.execute(text("SELECT COUNT(*) FROM raw_transactions")).scalar()
        processed_count = conn.execute(text("SELECT COUNT(*) FROM transactions_processed")).scalar()

    # ดึงข้อมูลมาวิเคราะห์
    df = pd.read_sql("SELECT * FROM transactions_processed", engine)
    
    # [FIX] แปลง Class เป็น String เพื่อให้ Seaborn จัดการสีได้ง่ายและถูกต้อง
    df['Class'] = df['Class'].astype(str)

    # 2. Setup Dashboard (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Data Pipeline Verification Dashboard', fontsize=16)

    # --- Plot 1: Data Integrity Check (Row Counts) ---
    axes[0, 0].bar(['Raw', 'Cleaned'], [raw_count, processed_count], color=['gray', '#2ecc71'])
    diff = raw_count - processed_count
    axes[0, 0].set_title(f'Data Volume Integrity\n(Removed {diff} duplicates)', fontsize=12)
    axes[0, 0].text(0, raw_count, f'{raw_count}', ha='center', va='bottom')
    axes[0, 0].text(1, processed_count, f'{processed_count}', ha='center', va='bottom')

    # --- Plot 2: Business Logic Verification (Fraud Time Pattern) ---
    fraud_data = df[df['Class'] == '1'] # ใช้ String '1'
    if not fraud_data.empty:
        sns.histplot(data=fraud_data, x='hour', bins=24, color='#e74c3c', kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Verification: Fraud Pattern by Hour\n(Expect peak at late night)', fontsize=12)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Fraud Data (Check Pipeline!)', ha='center', color='red')

    # --- Plot 3: Feature Engineering Verification (Amount Distribution) ---
    # [FIX] ใช้ hue='Class' และกำหนด palette เป็น String Key
    sns.boxplot(x='Class', y='Amount_Scaled', hue='Class', data=df, ax=axes[1, 0], 
                palette={'0': "#3498db", '1': "#e74c3c"}, legend=False)
    axes[1, 0].set_title('Verification: Amount Distribution (Scaled)\n(Fraud vs Normal)', fontsize=12)
    axes[1, 0].set_ylim(-2, 10) 

    # --- Plot 4: Target Class Distribution ---
    class_counts = df['Class'].value_counts()
    # ตรวจสอบว่ามีทั้ง 0 และ 1 หรือไม่ เพื่อกำหนดสีให้ถูกลำดับ
    colors = ['#3498db', '#e74c3c'] if '0' in class_counts and '1' in class_counts else None
    
    axes[1, 1].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=colors, explode=[0.1]*len(class_counts))
    axes[1, 1].set_title(f'Verification: Class Imbalance', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    # Save
    plt.savefig(VIZ_FILE)
    print(f"✅ Advanced Verification Dashboard saved at: {VIZ_FILE}")

# --- 3. DAG Definition ---

default_args = {
    'owner': 'DE486_1-68',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='fraud_detection_docker_pipeline_kaggle',
    default_args=default_args,
    description='ELT Pipeline fetching data from Kaggle API',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['docker', 'fraud-detection', 'kaggle'],
) as dag:

    t1_load_raw = PythonOperator(
        task_id='1_extract_kaggle_and_load',
        python_callable=extract_and_load_raw
    )

    t2_transform = PythonOperator(
        task_id='2_transform_in_db',
        python_callable=transform_in_db
    )

    t3_visualize = PythonOperator(
        task_id='3_generate_visualization',
        python_callable=visualize_data
    )

    t1_load_raw >> t2_transform >> t3_visualize