# built-in libraries
import os
from typing import Union
import subprocess

# third-party libraries
from mediapipe.python.solutions import selfie_segmentation
from PIL import Image, ImageDraw
from fpdf import FPDF
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import scale

# local imports
from badge_utils import *
from badge_settings import *


######################################
########## -- Functions -- ###########
######################################


def remove_background_from_photo(img: np.ndarray, fill_bg_col: tuple = PHOTO_BG_COL,
        threshold: float = PHOTO_BG_REMOVE_THRESHOLD, model: int = 0) -> np.ndarray:
    '''
    Removes the background of a person's photo, leaving only the person.
    
    #### Arguments
    
    `img` (np.ndarray): input photo containing a person
    `fill_bg_col` (tuple, default = PHOTO_BG_COL): solid color to replace the background with,
        either in form (R, G, B) (values from 0 to 255) or '#RRGGBB' (hex values)
    `threshold` (float, default = PHOTO_BG_REMOVE_THRESHOLD): higher makes it
        more likely to remove background
    `model` (int, default = 0): selfie segmentation model
    
    #### Returns
    
    np.ndarray: image without the background
    '''    
    seg = selfie_segmentation.SelfieSegmentation(model_selection=model)
    if isinstance(img, str):
        img = cv2.imread(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = seg.process(imgRGB)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > threshold
    if isinstance(fill_bg_col, tuple):
        fill_bg_col_bgr = (fill_bg_col[2], fill_bg_col[1], fill_bg_col[0])
        _img_bg = np.zeros(img.shape, dtype=np.uint8)
        _img_bg[:] = fill_bg_col_bgr
        imgOut = np.where(condition, img, _img_bg)
    else:
        bg_tuple = tuple(int(fill_bg_col.lstrip('#').upper()[i:i + 2], 16) for i in (0, 2, 4))
        return remove_background_from_photo(img, bg_tuple, threshold, model)
    return imgOut


def get_profile_photo(img: Union[np.ndarray, str], scale_factor: float = 1.0,
        size: tuple[int] = (300, 300), bgr_mean: list[int] = [104, 117, 123],
        swapRB: bool = False, ddepth: int = cv2.CV_32F, conf_threshold: float = 0.85,
        dnn_type: str = 'TensorFlow', dnn_model_file: str = DNN_MODEL_FILE,
        dnn_config_file: str = DNN_CONFIG_FILE, expand_dim_first: str = 'x',
        expand_x: float = 0.75, expand_y: float = 1.125,
        end_aspect_ratio: float = PHOTO_ASPECT_RATIO,
        allow_multiple_detections: bool = False, output_dir: str = None,
        output_filename: str = None, show_img: bool = False,
        must_full_fit: bool = False, on_fail_decr_thresh: bool = True) -> np.ndarray:
    '''
    Detects a face in an image and extracts the headshot photo.
    
    #### Arguments
    
    `img` (Union[np.ndarray, str]): input photo containing a face, in NumPy array form
    `scale_factor` (float, default = 1.0): preprocess input image by multiplying on a scale factor
    `size` (tuple[int], default = (300, 300)): spatial size for output image
    `bgr_mean` (list[int], default = [104, 117, 123]): mean BGR (or RGB if `swapRB`) of image
    `swapRB` (bool, default = False): set to True if `img` is an RGB image instead of BGR
    `conf_threshold` (float, default = 0.85): minimum confidence to count as a face
    `dnn_type` (str, default = 'TensorFlow'): either 'TensorFlow' or 'Caffe' as the detection method
    `dnn_model_file` (str, default = DNN_MODEL_FILE): path to face detector model file
    `dnn_config_file` (str, default = DNN_CONFIG_FILE): path to face detector config file
    `expand_dim_first` (str, default = 'x'): whether to extend bbox in the 'x' or 'y' direction.
        Only relevant when `end_aspect_ratio` is set.
    `expand_x` (float, default = 0.75): extend bbox by proportion in x-direction
    `expand_y` (float, default = 1.125): extend bbox by proportion in y-direction
    `end_aspect_ratio` (float, default = _PHOTO_ASPECT_RATIO): ensure a given aspect ratio
        of the end photo by scaling (opposite direction to `expand_dim_first`) after extending.
    `allow_multiple_detections` (bool, default = False): if True and multiple faces are found,
        returns multiple photos, otherwise raises ValueError
    `output_dir` (str, default = None): if set, save the photo(s) to this directory
    `output_filename` (str, default = None): if set, save the photo(s) as this filename.
        If multiple photos found, the names will be enumerated in decreasing order of confidence
    `show_img` (bool, default = False): if True, show the detected faces and photo boxes in a window
    `must_full_fit` (bool, default = False): if True, raises ValueError if the face is too close to
        the image border i.e. the photo bounding box lies partly outside the image
    `on_fail_decr_thresh` (bool, default = True): if no faces are found, decrease `conf_threshold`
        by 0.1 and try again. This will only be done once.

    #### Returns
    
    np.ndarray: photo images, in NumPy array format. If `allow_multiple_detections` and
        multiple faces found, returns a list[np.ndarray] of photo images, most confident first.

    Returns None if no faces found above the `conf_threshold`.
    
    #### Raises
    
    `RuntimeError`: if multiple faces are found and `allow_multiple_detections` is False
    `ValueError`: if `dnn_type` is not either 'TensorFlow' or 'Caffe'
    `ValueError`: if `must_full_fit` is True and the face is too close to the image border
        i.e. the photo bounding box lies partly outside the image
    `OSError`: if either of the DNN model/config files could not be found
    '''
    
    if dnn_type.lower().startswith('c'):
        if os.path.isfile(dnn_config_file) and os.path.isfile(dnn_model_file):
            net = cv2.dnn.readNetFromCaffe(dnn_config_file, dnn_model_file)
        else:
            raise OSError('Could not find Caffe config and/or Model file(s)')
    elif dnn_type.lower().startswith('t'):
        if os.path.isfile(dnn_config_file) and os.path.isfile(dnn_model_file):
            net = cv2.dnn.readNetFromTensorflow(dnn_model_file, dnn_config_file)
        else:
            raise OSError('Could not find Tensorflow config and/or Model file(s)')
    else:
        raise ValueError('dnn_type must be either "tensorflow" or "caffe".')

    if isinstance(img, str):
        img = cv2.imread(img)

    img_height, img_width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, scale_factor, size, bgr_mean,
        swapRB=swapRB, crop=False, ddepth=ddepth)
    net.setInput(blob)
    detections = net.forward()

    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= conf_threshold:
            x1 = int(detections[0, 0, i, 3] * img_width)
            y1 = int(detections[0, 0, i, 4] * img_height)
            x2 = int(detections[0, 0, i, 5] * img_width)
            y2 = int(detections[0, 0, i, 6] * img_height)
            bboxes.append((x1, y1, x2, y2, confidence))

    bboxes.sort(key=lambda bbox: bbox[4], reverse=True)
    
    if len(bboxes) == 0:
        if on_fail_decr_thresh:
            return get_profile_photo(img, scale_factor=scale_factor, size=size, bgr_mean=bgr_mean,
                swapRB=swapRB, ddepth=ddepth, conf_threshold=conf_threshold - 0.1,
                dnn_type=dnn_type, dnn_model_file=dnn_model_file,
                dnn_config_file=dnn_config_file, expand_dim_first=expand_dim_first,
                expand_x=expand_x, expand_y=expand_y, end_aspect_ratio=end_aspect_ratio,
                allow_multiple_detections=allow_multiple_detections, output_dir=output_dir,
                output_filename=output_filename, show_img=show_img, must_full_fit=must_full_fit,
                on_fail_decr_thresh=False)
        else:
            raise RuntimeError(f'Found no faces in this image, '
            f'with confidence level {conf_threshold}.')
    elif len(bboxes) > 1:
        photos = []
        if allow_multiple_detections:
            for i, bbox in enumerate(bboxes, start=1):
                if output_filename is not None:
                    filename, fileext = os.path.splitext(output_filename)
                    output_filename = f'{filename}-{i}{fileext}'
                photo = photo_from_bbox(img, bbox, expand_dim_first=expand_dim_first,
                    expand_x=expand_x, expand_y=expand_y, end_aspect_ratio=end_aspect_ratio,
                    output_dir=output_dir, output_filename=output_filename,
                    show_img=show_img, must_full_fit=must_full_fit)
                photos.append(photo)
            return photos
        else:
            raise RuntimeError(f'Found {len(bboxes)} faces in this image. '
            'If only one face was expected, try increasing `conf_threshold`. '
            'To allow all detections, set `allow_multiple_detections=True`.')
    else:
        photo = photo_from_bbox(img, bboxes[0], expand_dim_first=expand_dim_first,
                    expand_x=expand_x, expand_y=expand_y, end_aspect_ratio=end_aspect_ratio,
                    output_dir=output_dir, output_filename=output_filename,
                    show_img=show_img, must_full_fit=must_full_fit)
        return photo


def photo_from_bbox(img: np.ndarray, bbox: tuple[int], expand_dim_first: str = 'x',
        expand_x: float = 0.3, expand_y: float = 0.45,
        end_aspect_ratio: float = 1.18, output_dir: str = None,
        output_filename: str = None, show_img: bool = False,
        must_full_fit: bool = False) -> np.ndarray:
    '''
    Extends a bounding box from the face detection to a specific size and saves if required.
    
    #### Arguments
    
    `img` (np.ndarray): full image containing face(s)
    `bbox` (tuple[int]): (x1, y1, x2, y2, *_) descriptor of bounding box for a detected face
    `expand_dim_first` (str, default = 'x'): whether to extend bbox in the 'x' or 'y' direction.
        Only relevant when `end_aspect_ratio` is set.
    `expand_x` (float, default = 0.3): extend bbox by proportion in x-direction
    `expand_y` (float, default = 0.45): extend bbox by proportion in y-direction
    `end_aspect_ratio` (float, default = None): ensure a given aspect ratio (height / width)
        of the end photo by scaling in the opposite of the `expand_dim_first` direction after extending.
    `output_dir` (str, default = None): if set, save the photo(s) to this directory
    `output_filename` (str, default = None): if set, save the photo(s) as this filename
    `show_img` (bool, default = False): if True, show the detected faces and photo boxes in a window
    `must_full_fit` (bool, default = False): if True, raises ValueError if the face is too close to
        the image border i.e. the photo bounding box lies partly outside the image

    #### Returns
    
    np.ndarray: the photo for this bounding box in the image

    #### Raises

    `ValueError`: if `must_full_fit` is True and the face is too close to the image border
        i.e. the photo bounding box lies partly outside the image
    '''

    x1, y1, x2, y2, *_ = bbox
    orig_height = y2 - y1
    orig_width = x2 - x1
    img_height, img_width, _ = img.shape
    centre = ((x1 + x2) // 2, (y1 + y2) // 2)

    if expand_dim_first == 'x':
        x1_exp = max([0, x1 - int(expand_x * orig_width)])
        x2_exp = min([img_width, x2 + int(expand_x * orig_width)])
        new_width = x2_exp - x1_exp
        if end_aspect_ratio is not None:
            new_height = new_width * end_aspect_ratio
            y1_new = max([0, int(centre[1] - new_height / 2)])
            y2_new = min([img_height, int(centre[1] + new_height / 2)])
        else:
            y1_new = max([0, y1 - int(expand_y * orig_height)])
            y2_new = min([img_height, y2 + int(expand_y * orig_height)])
        photo = img[y1_new:y2_new, x1_exp:x2_exp, :]
        photo_bbox = (x1_exp, y1_new, x2_exp, y2_new)

    elif expand_dim_first == 'y':
        y1_exp = max([0, y1 - int(expand_y * orig_height)])
        y2_exp = min([img_height, y2 + int(expand_y * orig_height)])
        new_height = y2_exp - y1_exp
        if end_aspect_ratio is not None:
            new_width = new_height / end_aspect_ratio
            x1_new = max([0, int(centre[0] - new_width / 2)])
            x2_new = min([img_width, int(centre[0] + new_width / 2)])
        else:
            x1_new = max([0, x1 - int(expand_x * orig_width)])
            x2_new = min([img_width, x2 + int(expand_x * orig_width)])
        photo = img[y1_exp:y2_exp, x1_new:x2_new, :]
        photo_bbox = (x1_new, y1_exp, x2_new, y2_exp)
        
    if output_dir is not None:
        cv2.imwrite(os.path.join(output_dir, output_filename))

    if must_full_fit and any([photo_bbox[0] == 0, photo_bbox[1] == 0,
            photo_bbox[2] == img_width, photo_bbox[3] == img_height]):
        if photo_bbox[0] == 0: close_edge = 'left'
        elif photo_bbox[1] == 0: close_edge = 'top'
        elif photo_bbox[2] == img_width: close_edge = 'right'
        elif photo_bbox[3] == img_height: close_edge = 'bottom'
        raise ValueError(f'Face bounding box is too close to the {close_edge} edge of the '
        'image so a sufficiently large photo could not be made without distortion. Try reducing '
        f'{"expand_x" if close_edge in ["left", "right"] else "expand_y"} or using an image where '
        'the face is closer to the centre.')

    if show_img:
        img_marked = cv2.rectangle(img.copy(), photo_bbox[:2], photo_bbox[2:], (255, 0, 0),
            thickness=img_width // 400)
        img_marked = cv2.rectangle(img_marked, bbox[:2], bbox[2:4], (0, 0, 255),
            thickness=img_width // 400)
        cv2.namedWindow('Found Boxes', cv2.WINDOW_NORMAL)
        cv2.imshow('Found Boxes', img_marked)
        cv2.waitKey(0)

    return photo


def resize_keep_aspect_ratio(image: np.ndarray, width: int = None, height: int = None,
        interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    '''
    Resize an image while maintaining aspect ratio.
    
    #### Arguments
    
    `image` (np.ndarray): input image
    `width` (int, default = None): the width to resize to, or None if specifying height
    `height` (int, default = None): the height to resize to, or None if specifying width
    `interpolation` (int, default = cv2.INTER_AREA): interpolation method for resizing
    
    #### Returns
    
    np.ndarray: resized image
    '''    

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=interpolation)


def generate_badge_template(filename: str = None) -> Image.Image:
    '''
    Creates a template badge from the `badge_settings.py` config file.

    #### Arguments

    `filename` (str, default = None): if set, save the template image as this filename

    #### Returns

    Image.Image: template image
    '''

    # components
    title_box = [(int(BORDER * BADGE_WIDTH),
                  int(BORDER * BADGE_HEIGHT)),
                (int((1 - BORDER) * BADGE_WIDTH),
                int((BORDER + TITLE_CARD_HEIGHT_RATIO) * BADGE_HEIGHT))]

    # empty image
    image = Image.new('RGBA', (BADGE_WIDTH, BADGE_HEIGHT), BG_COL)
    imagedraw = ImageDraw.Draw(image)

    # title
    imagedraw.rectangle(title_box, fill=TITLE_CARD_COL)
    title_w, title_h = imagedraw.textsize(TITLE_TEXT, font=TITLE_FONT_SIZE)
    imagedraw.text((int((BADGE_WIDTH - title_w) // 2),
                    int(((BORDER + TITLE_CARD_HEIGHT_RATIO) * BADGE_HEIGHT - title_h) // 2)),
                    TITLE_TEXT, fill=TITLE_TEXT_COL, font=TITLE_FONT_SIZE)

    # info keys
    for key, (x, y, is_primary) in INFO_KEY_POS.items():
        if is_primary:
            imagedraw.multiline_text((x, y), key, fill=MAIN_TEXT_COL,
                spacing=INFO_PRIMARY_SPACING * BADGE_HEIGHT, font=PRIMARY_KEY_FONT_SIZE)
        else:
            bbox = imagedraw.multiline_textbbox((x, y), key,
                spacing=INFO_SECONDARY_SPACING * BADGE_HEIGHT,
                font=SECONDARY_KEY_FONT_SIZE, align='center')
            dx = ((1 - BORDER - LOGO_WIDTH_RATIO) - (2 * BORDER + PHOTO_WIDTH_RATIO \
                + (bbox[2] - bbox[0]) / BADGE_WIDTH)) * BADGE_WIDTH // 2
            imagedraw.multiline_text((x + dx, y), key, fill=MAIN_TEXT_COL,
                spacing=INFO_SECONDARY_SPACING * BADGE_HEIGHT,
                font=SECONDARY_KEY_FONT_SIZE, align='center')

    # save if required
    if filename is not None:
        image.save(filename)

    # return a PIL.Image.Image, already opened and ready
    return image


def generate_badge(template: Union[str, Image.Image],
        firstname: str, lastname: str, role: str, company: str,
        pronouns: str, id_no: str, photo: Union[np.ndarray, Image.Image],
        logo_path: str, export_dir: str = None,
        filename_format: str = r"f'{lastname.upper()}_{firstname}.png'",
        primary_value_format: str = r"f'{name}\n{role}\n{company}'",
        secondary_value_format: str = r"f'{pronouns}\n{id_no.upper()}'") -> Image.Image:
    '''
    Creates and saves a filled badge from given information and the template.
   
    #### Arguments

    `template` (Union[str, Image.Image]): badge to fill in
    `firstname`, `lastname`, `role`, `company`, `pronouns`, `id_no` (str): text field values
    `photo` (Union[np.ndarray, Image.Image]): photo to use, in PIL Image or NumPy array form
    `logo_path` (str): string path to logo to use. If UNSPECIFIED_VALUE, does not use a logo.
    `export_dir` (str, default = BADGES_DIR): where to save badge image
    `filename_format` (str, default = r"f'{lastname.upper()}_{firstname}.png'"): format to save
        individual badges as
    `primary_value_format` (str, default = r"f'{name}\\n{role}\\n{company}'"): format to write
        primary values as
    `secondary_values_format` (str, default = r"f'{pronouns}\\n{id_no.upper()}'"): format to write
        secondary values as
        
    Format strings should be entered as raw f-strings, such as the default examples. The
    expression inside the raw string will be evaluated in the context of all passed variables, plus
    some additional useful forms: `name` (firstname lastname), `initials_lastname` (F. LASTNAME),
    `name_pronouns` (firstname lastname (pronouns)), `role_company` (role at company).
    Use a raw newline r"\\n" to separate the fields.

    #### Returns

    Image.Image: PIL Image of the generated badge
    '''    

    if export_dir is not None:
        if not os.path.isdir(export_dir):
            os.mkdir(export_dir)

    # placeholders for use in string formats
    _ = [firstname, lastname, role, company, pronouns, id_no, photo]
    name = f'{firstname} {lastname}'
    initials_lastname = f"{' '.join([part[0] + '.' for part in firstname.replace('-', ' ').split(' ')])} {lastname}"
    name_pronouns = f'{name} ({pronouns})'
    role_company = f'{role} at {company}'

    # start from template
    image = template.copy() if isinstance(template, Image.Image) else Image.open(template)
    imagedraw = ImageDraw.Draw(image)

    # photo
    photo_pos = (int(BORDER * BADGE_WIDTH),
                 int((2 * BORDER + TITLE_CARD_HEIGHT_RATIO) * BADGE_HEIGHT))
    photo_resize = (int(PHOTO_WIDTH_RATIO * BADGE_WIDTH),
                    int(PHOTO_HEIGHT_RATIO * BADGE_HEIGHT))
    if isinstance(photo, np.ndarray):
        photo = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    photo = photo.resize(photo_resize, resample=Image.ANTIALIAS)
    image.paste(photo, photo_pos)

    # logo
    if logo_path is not None:
        logo_img = Image.open(logo_path)
        logo_img = logo_img.resize((int(LOGO_WIDTH_RATIO * BADGE_WIDTH),
                                    int(LOGO_HEIGHT_RATIO * BADGE_HEIGHT)),
                                    resample=Image.ANTIALIAS)
        logo_pos = (int((1 - BORDER - LOGO_WIDTH_RATIO) * BADGE_WIDTH),
            int((2 * BORDER + TITLE_CARD_HEIGHT_RATIO) * BADGE_HEIGHT))

        # XXX: there may be some edge cases left over, but this catches the common file types
        if has_transparency(logo_img):
            image.paste(logo_img, logo_pos, mask=logo_img.convert("RGBA"))
        else:
            image.paste(logo_img, logo_pos)
        
    # info value primary fields
    info_text_multiline = eval(primary_value_format, locals())
    key_bbox = imagedraw.multiline_textbbox(
        (2 * BORDER * BADGE_WIDTH,
        (4 * BORDER + TITLE_CARD_HEIGHT_RATIO + PHOTO_HEIGHT_RATIO) * BADGE_HEIGHT),
        PRIMARY_KEYS, spacing=INFO_PRIMARY_SPACING * BADGE_HEIGHT,
        font=PRIMARY_VALUE_FONT_SIZE)
    text_x = 4 * BORDER * BADGE_WIDTH + (key_bbox[2] - key_bbox[0])
    text_y = (4 * BORDER + TITLE_CARD_HEIGHT_RATIO + PHOTO_HEIGHT_RATIO) * BADGE_HEIGHT
    imagedraw.multiline_text((text_x, text_y), info_text_multiline,
        fill=MAIN_TEXT_COL, spacing=INFO_PRIMARY_SPACING * BADGE_HEIGHT,
        font=PRIMARY_VALUE_FONT_SIZE)

    # info value secondary fields
    info_text_multiline = eval(secondary_value_format, locals())
    key_bbox = imagedraw.multiline_textbbox(
        ((2 * BORDER + PHOTO_WIDTH_RATIO) * BADGE_WIDTH,
         (3 * BORDER + TITLE_CARD_HEIGHT_RATIO) * BADGE_HEIGHT),
        info_text_multiline, spacing=INFO_SECONDARY_SPACING * BADGE_HEIGHT,
        font=SECONDARY_VALUE_FONT_SIZE, align='center')
    dx = ((1 - BORDER - LOGO_WIDTH_RATIO) - (2 * BORDER + PHOTO_WIDTH_RATIO \
        + (key_bbox[2] - key_bbox[0]) / BADGE_WIDTH)) * BADGE_WIDTH // 2
    dy = imagedraw.textsize(info_text_multiline,
        font=SECONDARY_VALUE_FONT_SIZE)[1] * INFO_SECONDARY_VERTICAL_OFFSET
    text_x = (2 * BORDER + PHOTO_WIDTH_RATIO) * BADGE_WIDTH
    text_y = (3 * BORDER + TITLE_CARD_HEIGHT_RATIO) * BADGE_HEIGHT
    imagedraw.multiline_text((text_x + dx, text_y + dy), info_text_multiline,
        fill=MAIN_TEXT_COL, spacing=INFO_SECONDARY_SPACING * BADGE_HEIGHT,
        font=SECONDARY_VALUE_FONT_SIZE, align='center')

    # save if needed
    if export_dir is not None:
        filename = eval(filename_format, locals())
        image.save(os.path.join(export_dir, filename), 'PNG')
    
    return image


def generate_badge_pdf(source: Union[str, list[tuple[str, Image.Image]]],
        pdf_filename: str = None) -> FPDF:
    '''
    Packs all badges in a directory into a PDF.
    
    #### Arguments
    
    `source` (Union[str, list[Image.Image]): either a list of names and PIL Images
        or a directory containing badge images. Badges appear in name order.
    `pdf_filename` (str, default = None): if set, save the pdf with this filename

    #### Returns

    FPDF: PyPDF2 object of pdf of filled badges
    '''

    # init pdf
    pdf = FPDF(format='A4', orientation='portrait')

    # get correct order if using list of names and images
    if isinstance(source, list):
        source.sort(key=lambda p: p[0])
        source = [s[1] for s in source]
    
    for i, image in enumerate(source):

        # add new page when full
        if i % (PDF_ROWS * PDF_COLUMNS) == 0:
            pdf.add_page()

        # add image in the correct position - requires fpdf2 to add PIL Images
        pdf.image(image if isinstance(image, Image.Image) else os.path.join(BADGES_DIR, image),
            A4_WIDTH / PDF_COLUMNS * (i % PDF_COLUMNS),
            A4_HEIGHT / PDF_ROWS * ((i // PDF_COLUMNS) % PDF_ROWS),
            A4_WIDTH / PDF_COLUMNS, A4_HEIGHT / PDF_ROWS)

    if pdf_filename is not None:
        pdf.output(pdf_filename, 'F')
        print(f'Exported PDF {pdf_filename}: {pdf.page} pages of {len(source)} badges')

    return pdf


######################################
############# -- Main -- #############
######################################

if __name__ == '__main__':

    # open excel workbook - install openpyxl dependency if not working
    try:
        df = pd.read_excel(*EXCEL_SPREADSHEET, )  # open excel workbook
    except ImportError:
        subprocess.check_call(['pip', 'install', 'openpyxl'])
        df = pd.read_excel(*EXCEL_SPREADSHEET)

    # clean data - remove empty rows and replace empty cells with a placeholder
    df.dropna(axis=0, how='all', inplace = True)
    df.fillna(UNSPECIFIED_VALUE, inplace=True)
    num_people = len(df.index)

    # make badge template and init list of badge images
    badge_template = generate_badge_template()
    all_badges: list[tuple[str, Image.Image]] = []
    service = None
    errored_rows = []

    # process spreadsheet row by row
    for i, (line, row) in enumerate(df.iterrows(), start=1):
        
        try:
            # get relevant fields, placeholder _ will contain timestamp and email of google entries
            *_, firstname, lastname, role, company, pronouns, id_no, img, logo = row.values

            # get path to submitted photo
            if 'https://drive.google.com/open' in img:
                # fetch from Google Drive - manual sign-in required if first time in a while
                photo_path = f'{lastname.upper()}_{firstname}_{id_no[:4]}.jpg'
                download_file(img, photo_path, service=service, output_dir=PHOTOS_DIR)
                image_path = os.path.join(PHOTOS_DIR, photo_path)
            else:
                # fetch from local folder (manually submitted images)
                image_path = os.path.join(PHOTOS_DIR, img) if img != UNSPECIFIED_VALUE else None

            # get path to submitted logo
            if 'https://drive.google.com/open' in logo:
                # fetch from Google Drive - manual sign-in required if first time in a while
                logo_path = f'{company}_{lastname.upper()}_{firstname}_{id_no[:4]}.png'
                download_file(logo, logo_path, service=service, output_dir=LOGOS_DIR)
                logo_path = os.path.join(LOGOS_DIR, logo_path)
            else:
                # fetch from local folder (manually submitted images)
                logo_path = os.path.join(LOGOS_DIR, logo) if logo != UNSPECIFIED_VALUE else None

            # get clean headshot profile picture
            remove_bg = remove_background_from_photo(image_path)
            profile_photo = get_profile_photo(remove_bg)
            if isinstance(profile_photo, list):
                profile_photo = profile_photo[0]

            # make the badge
            badge = generate_badge(badge_template,
                firstname, lastname, role, company, pronouns, id_no,
                profile_photo, logo_path, export_dir=BADGES_DIR)

            # add to list of badges
            all_badges.append((f'{lastname}, {firstname} - {id_no}', badge))
            print(f'Created badge for {firstname} {lastname} ({i} / {num_people}).')

        except Exception as e:

            print(f'Warning: Failed to create badge for {firstname} {lastname} \n'
            f'Sheet row {line + 2} - row contents: \n{row} \n'
            f'Error type: {type(e).__name__}\nError message: \n {e}')
            errored_rows.append(f'{firstname} {lastname}')
            continue

    # combine all badges into a pdf and save
    generate_badge_pdf(all_badges, pdf_filename=PDF_NAME)
    print(f'Finished generating (total: {len(all_badges)} out of {num_people} people)')
    if len(all_badges) != num_people:
        print(f'Warning: Failed to create badges for: {errored_rows}. See debug info above.')
