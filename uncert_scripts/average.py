#!/usr/bin/env python
#
# Sums up text files with different names from several folders to 
# compute averaged plots
#
import sys
import os
from visualize import filter_and_sort


def append_to_file(path, text):
  with open(path, 'a') as f:
    f.write(text)


def average(*ins):
  out = ins[-1]
  ins = ins[:-1]
  
  print out
  try:
    os.mkdir(out)
  except:
    pass

  out = os.path.join(out, 'numbers')
  try:
    os.mkdir(out)
  except:
    pass

  for folder in ins:
    folder = os.path.join(folder, 'numbers')
    files = os.listdir(folder)
    print files
    label_files = filter_and_sort('label', files)
    uncert_files = filter_and_sort('uncert', files)

    for (label_file, uncert_file) in zip(label_files, uncert_files):
      labels = open(os.path.join(folder, label_file)).read()
      uncerts = open(os.path.join(folder, uncert_file)).read()
    
      append_to_file(os.path.join(out, label_file), labels)
      append_to_file(os.path.join(out, uncert_file), uncerts)

  
  

if __name__ == '__main__':
  average(*sys.argv[1:])
