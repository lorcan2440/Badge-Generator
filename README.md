# Badge Generator

Utils for quickly making badges for an event from a spreadsheet with info from e.g. Google Forms.

(All example photos are of real people but personal information is fake.)

### Usage

You will need:
- a spreadsheet of personal info (example: [SkyHigh Networkers (Responses).xlsx](SkyHigh Networkers (Responses).xlsx))
- for people who did not submit a photo in the form, a folder of all their manually submitted photos, linked in the spreadsheet. Downloading images directly from Google Drive requires a [Google app with OAuth consent](https://developers.google.com/drive/api/v3/manage-downloads) and its credentials file (named client_secret.json, not provided here)
- a logo (example: [logo.jpg](logo.jpg))
- the generator: [Badge_Generator.py](Badge_Generator.py)

Edit `Badge_Generator.py` with some config properties (optional) and run.
A pdf of the badges will be made.

*Important*: check that all photos have been cut correctly. Errors will be shown if no faces were detected and then they must be done manually.

Third-party requirements are (see [requirements.txt](requirements.txt)):
- [PIL](https://github.com/python-pillow/Pillow): for creating the badges as images
- [fpdf](https://github.com/mstamy2/PyPDF2): for generating the PDF
- [pandas](https://github.com/pandas-dev/pandas) and [openpyxl](https://foss.heptapod.net/openpyxl/openpyxl): for reading the excel spreadsheet
- [cv2](https://github.com/opencv/opencv-python): for detecting faces in the images to generate profile photos
- [mediapipe](https://google.github.io/mediapipe/): for removing backgrounds from submitted photos
