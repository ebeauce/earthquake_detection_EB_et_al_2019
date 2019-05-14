import os
import zipfile

link = 'https://www.dropbox.com/sh/w288vojhm5f9hy8/AADP1aXKAea4VhfvDitSLCk3a?dl=0'

print('Download the data to {}'.format(os.getcwd()))

os.system('wget -c {}'.format(link))

#print('Check your current working directory, and extract the content of the downloaded folder here.')

filename = link[link.rfind('/')+1:]

zip_ = zipfile.ZipFile('./{}'.format(filename), mode='r')

print('Extract the data in the current working directory: {}'.format(os.getcwd()))

zip_.extractall('.')
zip_.close()

print('Remove the zip archive to free some space.')
os.system('rm {}'.format(filename))
