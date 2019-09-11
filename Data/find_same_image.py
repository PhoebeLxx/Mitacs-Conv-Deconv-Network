import os

dir1 = './result660/'
dir2 = './result12840/'

for filename in os.listdir(dir1):
    if os.path.isfile(dir2 + filename):
        print(filename)