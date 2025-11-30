from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import os
import matplotlib
# ตั้งค่า Backend เป็น Agg เพื่อให้รันกราฟใน Docker (Headless) ได้โดยไม่ Error
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration (ปรับเพื่อ Docker) ---

# ใช้ชื่อ Service 'mysql_db' แทน localhost เพราะ Container คุยกันเองผ่าน Docker Network
# รูปแบบ: mysql+pymysql://user:password@service_name:port/db_name
DB_CONNECTION_STR = 'mysql+pymysql://root:root@mysql_db:3306/transaction_db'

# Path นี้ต้องตรงกับที่ Mount volume ไว้ใน docker-compose.yaml
# เรา Mount ./data ไว้ที่ /opt/airflow/data
DATA_PATH = '/opt/airflow/data'
CSV_FILE = os.path.join(DATA_PATH, 'transaction.csv')
VIZ_FILE = os.path.join(DATA_PATH, 'data_comparison.png')

# --- 2. ELT Functions ---

def extract_and_load_raw():
    """
    Step 1: Extract & Load
    อ่านไฟล์ CSV และโหลดลง MySQL ตาราง 'raw_transactions' ทันที
    """
    print(f"🚀 Starting Step 1: Extract & Load")
    
    # ตรวจสอบว่ามีไฟล์อยู่จริงไหม
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ที่: {CSV_FILE} กรุณาตรวจสอบว่าวางไฟล์ในโฟลเดอร์ data/ หรือยัง")

    # อ่าน CSV
    print(f"Reading CSV from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Read successfully. Raw Shape: {df.shape}")

    # เชื่อมต่อ Database และโหลดข้อมูล
    try:
        engine = sqlalchemy.create_engine(DB_CONNECTION_STR)
        with engine.connect() as conn:
            # ทดสอบการเชื่อมต่อ
            print("Connected to Database successfully.")
            
        # เขียนลง SQL (chunksize ช่วยลดการใช้ Memory)
        print("Uploading to MySQL table 'raw_transactions'...")
        df.to_sql('raw_transactions', engine, if_exists='replace', index=False, chunksize=5000)
        print("✅ Data loaded to 'raw_transactions' successfully.")
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        raise e

def transform_in_db():
    """
    Step 2: Transform
    อ่านจาก DB -> Clean/Feature Eng -> แยกตาราง -> Save กลับลง DB
    """
    print("🚀 Starting Step 2: Transform")
    engine = sqlalchemy.create_engine(DB_CONNECTION_STR)

    # 2.1 อ่านข้อมูลดิบ
    query = "SELECT * FROM raw_transactions"
    df = pd.read_sql(query, engine)
    print(f"Fetched {len(df)} rows from DB.")
    
    # Clean Column Names
    df.columns = df.columns.str.strip()

    # --- Feature Engineering ---
    # ลบข้อมูลซ้ำ
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {original_len - len(df)} duplicate rows.")

    # Time Engineering
    df["Time"] = df["Time"].astype(int)
    df["day"] = df["Time"] // (3600 * 24)
    df["hour"] = (df["Time"] // 3600) % 24

    # Scaling Amount
    scaler = StandardScaler()
    df['Amount'] = df['Amount'].astype(float)
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])

    # สร้าง ID
    df['transaction_id'] = range(1, len(df) + 1)

    # --- Splitting Tables (Normalization) ---
    cols_meta = ['transaction_id', 'Time', 'day', 'hour', 'Amount', 'Amount_Scaled', 'Class']
    v_columns = [f'V{i}' for i in range(1, 29)]
    cols_features = ['transaction_id'] + v_columns

    df_transactions = df[cols_meta]
    df_features = df[cols_features]

    # 2.2 Save กลับลง DB
    print("Saving processed tables...")
    df_transactions.to_sql('transactions_processed', engine, if_exists='replace', index=False, chunksize=5000)
    df_features.to_sql('transaction_features', engine, if_exists='replace', index=False, chunksize=5000)
    
    print("✅ Transformation Completed.")

def visualize_data():
    """
    Step 3: Visualization
    สร้างกราฟและบันทึกเป็นไฟล์ PNG กลับไปที่โฟลเดอร์ data
    """
    print("🚀 Starting Step 3: Visualization")
    engine = sqlalchemy.create_engine(DB_CONNECTION_STR)

    # ดึงข้อมูลจำนวนแถวมาเปรียบเทียบ
    with engine.connect() as conn:
        raw_count = conn.execute(text("SELECT COUNT(*) FROM raw_transactions")).scalar()
        processed_count = conn.execute(text("SELECT COUNT(*) FROM transactions_processed")).scalar()

    # ดึงข้อมูลเพื่อพลอตกราฟ Fraud
    df_clean = pd.read_sql("SELECT hour, Class FROM transactions_processed", engine)

    # เริ่มวาดกราฟ
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # กราฟ 1: Data Loss
    axes[0].bar(['Raw Data', 'Cleaned Data'], [raw_count, processed_count], color=['gray', 'green'])
    axes[0].set_title(f'Data Count Comparison\n(Lost {raw_count - processed_count} duplicates)')
    axes[0].set_ylabel('Number of Rows')

    # กราฟ 2: Fraud Pattern
    fraud_data = df_clean[df_clean['Class'] == 1]
    if not fraud_data.empty:
        sns.histplot(data=fraud_data, x='hour', bins=24, color='red', kde=True, ax=axes[1])
        axes[1].set_title('Fraud Transactions by Hour (Cleaned Data)')
        axes[1].set_xlabel('Hour of Day')
    else:
        axes[1].text(0.5, 0.5, 'No Fraud Data Found', ha='center')

    plt.tight_layout()
    
    # บันทึกไฟล์
    plt.savefig(VIZ_FILE)
    print(f"✅ Visualization saved at: {VIZ_FILE}")

# --- 3. DAG Definition ---

default_args = {
    'owner': 'somprat',  # เปลี่ยนชื่อเจ้าของได้ตามต้องการ
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='fraud_detection_docker_pipeline',  # ชื่อที่จะขึ้นในหน้าเว็บ Airflow
    default_args=default_args,
    description='ELT Pipeline for Fraud Detection on Docker',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['docker', 'fraud-detection'],
) as dag:

    t1_load_raw = PythonOperator(
        task_id='1_extract_and_load_raw',
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

    # กำหนดลำดับการทำงาน
    t1_load_raw >> t2_transform >> t3_visualize
