#/usr/bin python
import os
import sys
from datetime import datetime
import argparse

#python PMIdataGenerator.py head

def get_date(line):
    date_str = line.split('.')[0]
    md = date_str[-4:] #month and day
    m = md[0:2]
    d = md[2:]
    if m == '02' and d == '29':
        d = '28'
    md = m + d
    y = date_str[-5] #year
    if y == '0':
        date = md + '2012'
    elif y == '1':
        date = md + '2013'
    date_object = datetime.strptime(date, '%m%d%Y') 
    return date_object

def pmi_from_dir(dir_):
    for root, dirname, files in os.walk(dir_):
        files = list(filter(lambda a: "icon" not in a and "html" not in a, files))
        if len(files) > 0:
            try:
                files = sorted(files, key = get_date)
                i = 0
                start_time = datetime.now()
                for img in files:
                    date = get_date(img)
                    donor = img[:3]
                    if i == 0:
                        start_time = date
                    diff = date - start_time
                    pmi = diff.days
                    i += 1
                    print('{}/{} : {}'.format(donor, img, pmi))
            except:
                pass

def pmi_from_paths(paths):
    lines = open(paths).readlines()
    lines.sort()
    start_time = datetime.now()
    donor = ''
    month = ''
    for path in lines:
        path = path.strip()
        img_name = path.split('/')[-1]
        date = get_date(img_name)
        d = path.split('/')[-2]
        if d != donor:
            start_time = date
            donor = d
            month = img_name.split('.')[0][4:6]

        diff = date - start_time
        pmi = diff.days
        print('{} : {}: {}'.format(path, pmi, month))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', default='', type=str)
    parser.add_argument('--pathsfile', default='', type=str)
    args = parser.parse_args()

    dir_ = args.dirname
    paths = args.pathsfile
    if len(dir_) > 0:
        pmi_from_dir(dir_)
    elif len(paths) > 0:
        pmi_from_paths(paths)
