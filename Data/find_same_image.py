import os

dir1 = './image/'
dir2 = './image64/'

for filename in os.listdir(dir1):
    if os.path.isfile(dir2 + filename):
        print(filename)