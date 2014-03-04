"""
Functions to assist in multithreaded processing with Python 2.7.
"""
import sys
import threading


class LockIterateApply(threading.Thread):
    def __init__(self, F, it, lock, sep='\n', out_stream=sys.stdout):
        self.lock = lock
        self.func = F
        self.out_stream = out_stream
        self.myiter = it
        self.sep = sep
        threading.Thread.__init__(self)

    def run(self):
        t = True
        while t:
            t = self.read_apply()

    def read_apply(self):
        try:
            self.lock.acquire()
            x = self.myiter.next()
            self.lock.release()
        except StopIteration:
            self.lock.release()
            return False
        y = self.transform(x)
        self.output(y)
        return True

    def transform(self, x):
        return self.func(x)

    def output(self, y):
        self.out_stream.write(str(y) + self.sep)
