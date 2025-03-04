1. Cloning the Repository
git clone https://github.com/Hasan-Kaymet/KAYMET_TXT_TO_SQL_Hasan_Ar-.git

cd KAYMET_TXT_TO_SQL_Hasan_Ar-.git


2. Installing Dependencies
a) Python + pip (Local Setup)

Use the requirements.txt file:

pip install -r requirements.txt


3. Environment Variables
cp .env.example .env  #For api key

4. Running the App
uvicorn app.main:app --host 0.0.0.0 --port 8000

By default, visit http://localhost:8000/docs in your browser to see the interactive API docs (Swagger UI).