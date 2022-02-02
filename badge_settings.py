from PIL import ImageFont


######################################
########## -- Parameters -- ##########
######################################

# sources
EXCEL_SPREADSHEET = ('assets/SkyHigh Networkers (Responses).xlsx', 'Form Responses 1')
PHOTOS_DIR = 'Submitted Images'
LOGOS_DIR = 'assets/logos'
DNN_MODEL_FILE = r'C:\OpenCV\models-sources\face-detection\opencv_face_detector_uint8.pb'
DNN_CONFIG_FILE = r'C:\OpenCV\models-sources\face-detection\opencv_face_detector.pbtxt'

# authentication
CLIENT_SECRET_JSON = 'assets/client_secret.json'

# destinations
BADGES_DIR = 'Badges Produced'
PDF_NAME = 'Badges.pdf'

# badge dimensions
PDF_ROWS = 4
PDF_COLUMNS = 2
BADGE_WIDTH = 1340
BADGE_HEIGHT = 1000
BORDER = 0.025
A4_WIDTH = 210
A4_HEIGHT = 297

# colours - ##BBGGRR
BG_COL = '#a2d1de'  # light blue
PHOTO_BG_COL = '#d0e5f7'  # very light blue
MAIN_TEXT_COL = '#000000'  # black
TITLE_CARD_COL = '#f42a41'  # red
TITLE_TEXT_COL = '#ffffff'  # white

# fonts - find font filenames by going to C:/Windows/Fonts.
# choose a family and font and view properties.
TITLE_FONT_SIZE = ImageFont.truetype('BOD_BI.TTF', 90)
PRIMARY_KEY_FONT_SIZE = ImageFont.truetype('arialbd.ttf', 60)
PRIMARY_VALUE_FONT_SIZE = ImageFont.truetype('arial.ttf', 60)
SECONDARY_KEY_FONT_SIZE = ImageFont.truetype('arialbi.ttf', 60)
SECONDARY_VALUE_FONT_SIZE = ImageFont.truetype('ariali.ttf', 60)

# title card
TITLE_CARD_HEIGHT_RATIO = 0.18
TITLE_TEXT = 'SkyHigh Networkers'

# photo
PHOTO_HEIGHT_RATIO = 0.4725
PHOTO_WIDTH_RATIO = 0.30
PHOTO_BG_REMOVE_THRESHOLD = 0.8
PHOTO_ASPECT_RATIO = (PHOTO_HEIGHT_RATIO * BADGE_HEIGHT) / (PHOTO_WIDTH_RATIO * BADGE_WIDTH)

# logo
LOGO_WIDTH_RATIO = 0.29
LOGO_HEIGHT_RATIO = 0.41

# information
PRIMARY_KEYS = 'Name:\nRole:\nAt'
SECONDARY_KEYS = 'Pronouns:\nID No.:'
UNSPECIFIED_VALUE = '(unspecified)'

# info dimensions and relations
INFO_PRIMARY_SPACING = 0.015
INFO_SECONDARY_SPACING = 9 * INFO_PRIMARY_SPACING
INFO_SECONDARY_VERTICAL_OFFSET = 0.7
INFO_KEY_POS = {
    PRIMARY_KEYS: (2 * BORDER * BADGE_WIDTH,
        (4 * BORDER + TITLE_CARD_HEIGHT_RATIO + PHOTO_HEIGHT_RATIO) * BADGE_HEIGHT, True),
    SECONDARY_KEYS: ((2 * BORDER + PHOTO_WIDTH_RATIO) * BADGE_WIDTH,
        (3 * BORDER + TITLE_CARD_HEIGHT_RATIO) * BADGE_HEIGHT, False),
}