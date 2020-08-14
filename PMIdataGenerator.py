#/usr/bin python
import os
import sys
from datetime import datetime

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

if __name__ == '__main__':
    #fp = open(sys.argv[1], 'r')
    #lines = fp.readlines()
    path = sys.argv[1]
    for root, dirname, files in os.walk(path):
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
