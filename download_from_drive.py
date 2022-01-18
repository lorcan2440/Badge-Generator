import io
import pickle
import os
import subprocess

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build, Resource
    from googleapiclient.http import MediaIoBaseDownload
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
except ImportError:
    subprocess.check_call(['pip', 'install', 'google', 'google-auth-oauthlib', 'google-api-python-client'])
finally:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build, Resource
    from googleapiclient.http import MediaIoBaseDownload
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError


def authenticate_access(client_secret_file: str = 'client_secret.json', api_name: str = 'drive',
        api_version: str = 'v3', scopes: list[str] = ['https://www.googleapis.com/auth/drive']):

    print(f'''Creating service for Google Drive access:
    Client Secrets file: {client_secret_file}
    API Name: {api_name}
    API Version: {api_version}
    Scopes: {scopes}''')

    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = scopes

    cred = None
    pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()

        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(f'Created service succesfully: {API_SERVICE_NAME}')
        return service
    except RefreshError as e:
        print(e)
        print('Retrying with original credentials...')
        os.remove(pickle_file)
        return authenticate_access(client_secret_file, api_name, api_version, scopes)


def download_file(filename: str, output_name: str, service: Resource = None, output_dir: str = None,
        **auth_kwargs):

    if service is None:
        print('Requres manual authentication when running for the first time.')
        service = authenticate_access(**auth_kwargs)
 
    file_id = filename.split('?id=')[-1]

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        output_dir = ''

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fd=fh, request=request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f'Downloading {filename} as {output_name}: {status.progress() * 100}%')

    fh.seek(0)
    with open(os.path.join(output_dir, output_name), 'wb') as f:
        f.write(fh.read())
