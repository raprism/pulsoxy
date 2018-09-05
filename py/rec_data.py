#! /usr/bin/env python

# Extended version of http://shallowsky.com/software/scripts/ardmonitor
#
# Read the output of an Arduino which may be printing sensor output,
# and either
#   at the same time, monitor the user's input and send it to the Arduino.
# or
#   give defined start command and then log only data

import os, sys, time
import serial, select
from contextlib import contextmanager

class BoardSerial() :

    def __init__(self,
                 baud=115200,
                 baseports=['/dev/ttyUSB', '/dev/ttyACM'],
                 logfile_template='test{}.csv'
                 ):
        self.baud = baud
        self.baseports = baseports
        self.ser = None
        self.logfile_default = logfile_template

    def _connect(self,
                 baud=None,
                 timeout=1):
        if baud is not None:
            self.baud = baud
        if self.ser is not None:
            self.ser.close()
        for baseport in self.baseports:
            for i in range(0, 8):
                try:
                    port = baseport + str(i)
                    self.ser = serial.Serial(port, self.baud,
                                             timeout=timeout)
                    print("Opened", port)
                    self.ser.flushInput()
                    break
                except serial.SerialException:
                    self.ser = None
            if self.ser is not None:
                break

    @contextmanager
    def connect(self,
                baud=None):
        """
        with instance.connect():
            do_something()
        """
        self._connect(baud)
        yield self.ser
        if self.ser:
            self.ser.close()
            print("Closed", self.ser.port)

    def _interactive(self, input_cycle=0.2):
        while True:
            # Check whether the user has typed anything:
            inp, outp, err = select.select([sys.stdin, self.ser], [], [],
                                           input_cycle)
            # Check for user input:
            if sys.stdin in inp :
                line = sys.stdin.readline()
                self.ser.write(line.encode())
            # check for Arduino output:
            if self.ser in inp :
                line = self.ser.readline().strip()
                print("Board:", line)

    def _logging(self,
                 logfile=None,
                 start_command='',
                 omit_text='##',
                 omit_first=10):
        if start_command:
            self.ser.write(start_command.encode())
        if logfile is None:
            now = time.strftime('%y%m%d_%H%M%S', time.localtime())
            logfile = self.logfile_default.format('-' + now)
        elif os.path.isfile(logfile):
            print('Stopping, because log file {} exists'.format(logfile))
            return
        exc = None
        i = j = 0
        with open(logfile, 'wb') as fh:
            while True:
                try:
                    line = self.ser.readline()
                    i += 1
                    if i < omit_first:
                        continue
                    if line[:len(omit_text)] != omit_text.encode():
                        j += 1
                        fh.write(line)
                        if not j % 100:
                            print('{: 5}'.format(j))
                except:
                    exc = sys.exc_info()
                    break
        if exc:
            raise exc[1].with_traceback(exc[2])

    def run_logging(self,
                    logfile=None,
                    start_command='start',
                    overwrite=False):
        if logfile is not None \
           and overwrite and os.path.isfile(logfile):
            os.remove(logfile)
        self.run(mode='logging',
                 logfile=logfile,
                 start_command=start_command)

    def run_interactive(self):
        self.run(mode='interactive')

    def run(self,
            mode='logging',
            logfile=None,
            start_command='start'):
        with self.connect():
            try:
                if self.ser is None:
                    raise serial.SerialException
                if mode=='interactive':
                    self._interactive()
                elif mode=='logging':
                    self._logging(logfile, start_command=start_command)
                else:
                    raise NotImplementedError('mode {} not implemented'\
                                              .format(mode))
            except serial.SerialException:
                print("Disconnected (Serial exception)")
            except IOError:
                print("Disconnected (I/O Error)")
            except KeyboardInterrupt:
                print("Interrupt")

if __name__ == '__main__':

    serial_instance = BoardSerial()
    serial_instance.run()
