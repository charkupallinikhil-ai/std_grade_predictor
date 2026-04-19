import zipfile
import os

def create_zip():
    files_to_zip = [
        'app.py',
        'train_model.py',
        'requirements.txt',
        'model.pkl',
        'student_dataset.xlsx'
    ]
    
    zip_name = 'predictive_analytics_app.zip'
    
    print(f"Creating {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file}")
            else:
                print(f"Warning: {file} not found. Skipping.")
                
    print(f"Successfully created {zip_name} in the current directory.")

if __name__ == "__main__":
    create_zip()
