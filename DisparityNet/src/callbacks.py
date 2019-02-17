from tensorflow.keras.callbacks import Callback
import csv
from shutil import copy2 as copy
from os import remove, mkdir
from os.path import exists, join, isdir


class EpochCSVLogger(Callback):
    def __init__(self, csv_dir, file_name='epoch.log.csv', delimiter=',', append=False):
        self.csv_dir = csv_dir
        self.file_name = file_name
        self.delimiter = delimiter
        self.append = append
        self.file = None
        self.csv = None
        self.is_initial = not exists(join(self.csv_dir, self.file_name))
        super(EpochCSVLogger, self).__init__()

    def create_dir(self):
        if not isdir(self.csv_dir):
            mkdir(self.csv_dir)

    def create_file(self):
        self.file = open(join(self.csv_dir, self.file_name), 'a' if self.append else 'w')
        self.csv = csv.writer(self.file)

    def create_backups(self):
        copy(join(self.csv_dir, self.file_name), join(self.csv_dir, self.file_name + '.bak'))

    def remove_backups(self):
        remove(join(self.csv_dir, self.file_name + '.bak'))

    def write_header(self, logs):
        if self.is_initial:
            keys = [x for x in logs.keys()]
            self.csv.writerow(['epoch']+[x for x in sorted(keys)])
            self.is_initial = False

    def on_train_begin(self, logs=None):
        self.create_dir()
        self.create_file()
        self.create_backups()

    def on_train_end(self, logs=None):
        self.file.close()
        self.remove_backups()

    def on_epoch_end(self, epoch, logs=None):
        self.write_header(logs)
        keys = [x for x in logs.keys()]
        self.csv.writerow([epoch]+[logs[x] for x in sorted(keys)])


class BatchCSVLogger(Callback):
    def __init__(self, csv_dir, file_name='batch.log.csv', delimiter=',', append=False):
        self.csv_dir = csv_dir
        self.file_name = file_name
        self.delimiter = delimiter
        self.append = append
        self.file = None
        self.csv = None
        self.is_initial = not exists(join(self.csv_dir, self.file_name))
        super(BatchCSVLogger, self).__init__()

    def create_dir(self):
        if not isdir(self.csv_dir):
            mkdir(self.csv_dir)

    def create_file(self):
        self.file = open(join(self.csv_dir, self.file_name), 'a' if self.append else 'w')
        self.csv = csv.writer(self.file)

    def create_backups(self):
        copy(join(self.csv_dir, self.file_name), join(self.csv_dir, self.file_name+'.bak'))

    def remove_backups(self):
        remove(join(self.csv_dir, self.file_name+'.bak'))

    def write_header(self, logs):
        if self.is_initial:
            keys = [x for x in logs.keys()]
            self.csv.writerow([x for x in sorted(keys)])
            self.is_initial = False

    def on_train_begin(self, logs=None):
        self.create_dir()
        self.create_file()
        self.create_backups()

    def on_train_end(self, logs=None):
        self.file.close()
        self.remove_backups()

    def on_batch_end(self, batch, logs=None):
        self.write_header(logs)
        keys = [x for x in logs.keys()]
        self.csv.writerow([logs[x] for x in sorted(keys)])
