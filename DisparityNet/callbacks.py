from tensorflow.keras.callbacks import TensorBoard, Callback
from data_generator import train_parameters
import csv
from shutil import copy2 as copy
from os import remove
from os.path import exists


class EpochCSVLogger(Callback):
    def __init__(self, filename, delimiter=',', append=False):
        self.filename = filename
        self.delimiter = delimiter
        self.append = append
        self.file = None
        self.csv = None
        self.is_initial = not exists(self.filename)
        super(EpochCSVLogger, self).__init__()

    def create_file(self):
        self.file = open(self.filename, 'a' if self.append else 'w')
        self.csv = csv.writer(self.file)

    def create_backups(self):
        copy(self.filename, self.filename+'.bak')

    def remove_backups(self):
        remove(self.filename+'.bak')

    def write_header(self, logs):
        if self.is_initial:
            keys = [x for x in logs.keys()]
            self.csv.writerow(['epoch']+[x for x in sorted(keys)])
            self.is_initial = False

    def on_train_begin(self, logs=None):
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
    def __init__(self, filename, delimiter=',', append=False):
        self.filename = filename
        self.delimiter = delimiter
        self.append = append
        self.file = None
        self.csv = None
        self.is_initial = not exists(self.filename)
        super(BatchCSVLogger, self).__init__()

    def create_file(self):
        self.file = open(self.filename, 'a' if self.append else 'w')
        self.csv = csv.writer(self.file)

    def create_backups(self):
        copy(self.filename, self.filename+'.bak')

    def remove_backups(self):
        remove(self.filename+'.bak')

    def write_header(self, logs):
        if self.is_initial:
            keys = [x for x in logs.keys()]
            self.csv.writerow([x for x in sorted(keys)])
            self.is_initial = False

    def on_train_begin(self, logs=None):
        self.create_file()
        self.create_backups()

    def on_train_end(self, logs=None):
        self.file.close()
        self.remove_backups()

    def on_batch_end(self, batch, logs=None):
        self.write_header(logs)
        keys = [x for x in logs.keys()]
        self.csv.writerow([logs[x] for x in sorted(keys)])


# Call Back For Tensorboard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
                          batch_size=train_parameters['batch_size'])

epoch_csv_logger = EpochCSVLogger(filename='csvs/epoch.log.csv', append=True)

batch_csv_logger = BatchCSVLogger(filename='csvs/batch.log.csv', append=True)
