'''Utilities for working with operating system objects like files etc in opimage'''
import os
import skimage.data
import datetime
import dateutil
import matplotlib.pyplot as plt
import csv
import json

def get_image_stack(img_dir):
    '''returns an array of full paths for image files in the img_dir'''
    return [full_path(f, img_dir) for f in filter_image_files(os.listdir(img_dir))]

def filter_image_files(file_list):
    '''filters and keeps only image files'''
    return [f for f in file_list if is_image(f)]

def full_path(filename, img_dir):
    '''returns a full path by joining img_dir and f'''
    return os.path.join(img_dir, filename)

def is_image(filename):
    '''checks whether file is an image'''
    return filename[-3:] in ['bmp', 'BMP']

def image_file_to_array(image_file_name):
    '''from a file name, loads image and returns a numpy nd.array'''
    return skimage.data.load(image_file_name)

def fname_to_date(fname):
    date = None
    path, ext = os.path.splitext(fname)
    basename = path.split("/")[-1]
    try:
        date = dateutil.parser.parse(basename)
    except:
        date = webb_to_ISO(basename)
        date = dateutil.parser.parse(date)
    return date

def sort_by_datetime(file_list):
    result = [ [fname_to_date(fname), fname] for fname in file_list ]
    result = sorted(result, key=lambda date: date[0])
    return [r[1] for r in result]

def webb_to_ISO(string):
    y = '20' + string[0:2]
    m = string[2:4]
    d = string[4:6]
    h =  string[7:9]
    mn = string[9:11]
    s = string[11:13]
    return "-".join([y,m,d]) + 'T' + ":".join([h,mn,s])

def save_image(image, fname='save_image.png', title="NoTitle",cmap='bone'):
    fig, ax = plt.subplots(ncols=1, figsize=(6,6))
    ax.imshow(image,cmap)
    ax.set_title(title)
    plt.savefig(fname)
    plt.close()

def save_csv(data, colnames=[], fname='result.csv', sep=","):
    with open(fname, 'wb') as csv_file:
        writer = csv.writer(csv_file, delimiter=sep,quotechar='\'', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(colnames)
        for row in data:
            writer.writerow(row)

def box_to_string(tple):
    '''get corners of box as string'''
    return ":".join([str(tple[0].start), str(tple[0].stop) ,str(tple[1].start), str(tple[1].stop) ] )

def load_from_json(fname):
    '''slurp a json file into an object'''
    with open(fname, "r") as f:
        return json.loads(f.read() )

def write_to_json(obj, fname):
    '''dump an object as json to a file'''
    with open(fname, "w") as f:
        f.write(json.dumps(obj))
