"""
Colab configurations to run flexible SVBRDF acquisition.
"""

import fileinput, glob, shutil, os

v1_import_statement = 'import tensorflow as tf'
v2_import_statement = 'import tensorflow.compat.v1 as tf \ntf.disable_v2_behavior()'

for filename in glob.glob("*.py"):
  print(filename)
  with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
      for line in file:
          print(line.replace(v1_import_statement, v2_import_statement), end='')

initial = "int(multiprocessing.cpu_count() / 4))"
final = "max(int(multiprocessing.cpu_count() / 4), 1))"
with fileinput.FileInput("dataReader.py", inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace(initial, final), end='')

test_path = "/content/multi-image-deepNet-SVBRDF-acquisition/testImagesExamples/test"
if not os.path.exists(test_path):
  os.mkdir("/content/multi-image-deepNet-SVBRDF-acquisition/testImagesExamples/test")
