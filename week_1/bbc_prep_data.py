import os
import csv

#create a .csv file from the bbc/ folder that can be found here:
#http://mlg.ucd.ie/datasets/bbc.html
def prepare_data(dir):
    with open('classes.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('classes', 'text'))
    subdirs = [subdir for subdir in os.listdir(dir) if not subdir.endswith(".TXT")]
    for subdir in subdirs:
        print(subdir)
        dir_files = dir + subdir +'/'
        for filename in os.listdir(dir_files):

            with open(dir_files + filename, 'rb') as f:
                lines = f.read().splitlines()
                with open('classes.csv', 'a+') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow((str(subdir), lines))

if __name__ == '__main__':
        dir = 'datasets/bbc/'
        prepare_data(dir)
