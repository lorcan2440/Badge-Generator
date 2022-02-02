import io
import pickle
import os
import subprocess

from PIL import Image

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build, Resource
    from googleapiclient.http import MediaIoBaseDownload
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
except ImportError:
    subprocess.check_call(['pip', 'install',
        'google', 'google-auth-oauthlib', 'google-api-python-client'])
finally:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build, Resource
    from googleapiclient.http import MediaIoBaseDownload
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError

from badge_settings import CLIENT_SECRET_JSON


def authenticate_access(client_secret_file: str = CLIENT_SECRET_JSON, api_name: str = 'drive',
        api_version: str = 'v3', scopes: list[str] = ['https://www.googleapis.com/auth/drive'],
        use_pickle_file: bool = True) -> Resource:
    '''
    Authenticates a user for accessing a service on Google Cloud API.
    
    #### Arguments
    
    `client_secret_file` (str, default = CLIENT_SECRET_JSON): path to credentials file
    `api_name` (str, default = 'drive'): name of the API to be accessed
    `api_version` (str, default = 'v3'): version of the API to be accessed
    `scopes` (list[str], default = ['https://www.googleapis.com/auth/drive']): scope of app
    `use_pickle_file` (bool, default = True): if True, load the credentials
        from a pickle file instead of a plaintext file
    
    #### Returns
    
    `googleapiclient.discovery.Resource`: a valid API service instance to be used later
    '''    

    print(f'''Creating service for Google Drive access:
    Client Secrets file: {client_secret_file}
    API Name: {api_name}
    API Version: {api_version}
    Scopes: {scopes}''')

    cred = None
    pickle_file = f'token_{api_name}_{api_version}.pickle'

    if os.path.exists(pickle_file) and use_pickle_file:
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            try:
                cred.refresh(Request())
            except RefreshError as e:
                print(e)
                print(f'Common fix: delete {pickle_file} and run again, authenticating once manually.')
                print('Trying again without using pickle file...')
                authenticate_access(client_secret_file, api_name, api_version, scopes,
                    use_pickle_file=False)
                
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes)
            cred = flow.run_local_server()

        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    service = build(api_name, api_version, credentials=cred)
    print(f'Created service succesfully: {api_name}')
    return service



def download_file(filename: str, output_name: str, service: Resource = None, output_dir: str = None,
        **auth_kwargs) -> None:
    '''
    Downloads a file stored in Google Drive to the local file system.
    
    #### Arguments
    
    `filename` (str): full filename of the file stored on Drive
    `output_name` (str): filename to store locally as
    `service` (Resource, default = None): if set, use a pre-authenticated service,
        otherwise authenticate again
    `output_dir` (str, default = None): if set, place the file in the given directory,
        otherwise place in the current working directory

    #### Returns

    `None`
    '''    

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


def has_transparency(img: Image.Image) -> bool:
    '''
    Determines whether an image contains transparent pixels.

    Source:
    https://stackoverflow.com/a/58567453/8747480

    #### Arguments

    `img` (PIL.Image.Image): image to check for transparency

    #### Returns

    `bool`: True if transparent, False if not transparent.
    '''

    if img.info.get('transparency', None) is not None:
        return True

    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True

    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True

    return False
