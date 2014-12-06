"""
Functions to assist in multithreaded processing with Python 2.7.
"""
import sys
import threading


class LockIterateApply(threading.Thread):
    """
    Wraps an iterable into a "locked" iterable for threading, applies
    function and write to out_stream.
    """
    def __init__(self, func, iterable, lock, sep='\n', out_stream=sys.stdout):
        """
        Parameters
        ----------
        func : function of one variable
        iterable : iterable, yields func arg
        lock : threading.Lock()
        sep : str
            for writing to out_stream
        out_stream : open, buff, standard stream
            must have a .write() method
        """
        self.lock = lock
        self.func = func
        self.out_stream = out_stream
        self.myiter = iterable
        self.sep = sep
        threading.Thread.__init__(self)

    def run(self):
        t = True
        while t:
            t = self.read_apply()

    def read_apply(self):
        """
        locks iterable.next() and applies self.transform
        """
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
        """
        Writes to out_stream.
        """
        self.out_stream.write(str(y) + self.sep)


def threading_easy(func, iterable, n_threads, sep='\n', out_stream=sys.stdout):
    """
    Wraps the python threading library; takes an iterator, a function which
    acts on each element the iterator yields and starts up the prescribed
    number of threads. The output of each thread process is pass to an
    out_stream.

    Parameters
    ----------
    func : function of one variable
    iterable : iterable which yields function argument
    n_threads : int
    sep : string
        for concatenating results to write
    out_stream : open file, buffer, standard stream
        must have a .write() method

    Returns
    -------
    writes to out_stream

    Examples
    --------
    Function of one variable:
    >>> from time import sleep
    >>> import rosetta.parallel.threading_easy as te
    >>> my_iter = (x for x in range(10))
    >>> def double(n):
            sleep(1)
            return 2*x
    >>> te.threading_easy(my_iter, double, n_threads=10)

    Function of more than one variable:
    >>> from functools import partial
    >>> def double2(n, t):
            sleep(t)
            return 2*n
    >>> double = partial(double2, t=1)
    >>> te.threading_easy(my_iter, double, n_threads=10)

    Notes: in order to support the default sys.stdout out_stream, all results
    are converted to string before writing.

    """
    if n_threads is None or n_threads <= 1:
        for each in iterable:
            out_stream.write(('%s'+sep)%func(each))
    else:
        lock = threading.Lock()
        threads = []
        for i in range(n_threads):
            t = LockIterateApply(func, iterable, lock, sep, out_stream)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()
