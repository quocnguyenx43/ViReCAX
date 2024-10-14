# !pip install google-api-python-client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
import os
import argparse as arg

def authenticate(service_account_file):
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    return creds

def get_file_id_by_name(file_name, service, folder_id):
    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None

def upload_file(service_account_file, folder_id, file_path):
    PARENT_FOLDER_ID = folder_id
    creds = authenticate(service_account_file)
    service = build('drive', 'v3', credentials=creds)

    # Check if the file already exists in Google Drive
    existing_file_id = get_file_id_by_name(file_path, service, PARENT_FOLDER_ID)
    if existing_file_id:
        # If the file exists, update its content
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().update(
            fileId=existing_file_id,
            media_body=media,
        ).execute()
        print('File updated. File ID: ', existing_file_id)
        return existing_file_id
    else:
        # If the file doesn't exist, create a new file
        file_metadata = {
            'name': file_path,
            'parents': [PARENT_FOLDER_ID]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
        ).execute()
        file_id = file['id']
        print('File uploaded. File ID: ', file_id)
        return file_id
    
parser = arg.ArgumentParser(description="Params")
parser.add_argument("--service_account_file", type=str)
parser.add_argument("--folder_id", type=str)
parser.add_argument("--local_folder", type=str)
args = parser.parse_args()
args = vars(args)

service_account_file = args['service_account_file']
folder_id = args['folder_id']
local_folder = args['local_folder']
files = os.listdir(local_folder)

for file in files:
    print(f'upload {file}')
    upload_file(service_account_file, folder_id, local_folder + '/' + file)
    
# !python upload_file_to_drive.py --service_account_file "/kaggle/input/new-service-account/service_account.json" --folder_id "1H6GRQN3oQ4n2FmTrelQXbfV-wAEreMNM" --local_folder "outputs/"