"""
Functions to assist in parallel processing with Python 2.7.

* Memory-friendly iterator functionality (wrapping Pool.imap).
* Exit with Ctrl-C.
* Easy use of n_jobs. (e.g. when n_jobs == 1, processing is serial)
* Similar to joblib.Parallel but with the addition of imap functionality
  and a more effective way of handling Ctrl-C exit (we add a timeout).
"""
import itertools
from multiprocessing import cpu_count, Pool, Process, Manager, Lock
from multiprocessing.pool import IMapUnorderedIterator, IMapIterator
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import sys


###############################################################################
# Globals
###############################################################################
# Used as the timeout
GOOGLE = 1e100


###############################################################################
# Functions
###############################################################################

def _do_work_off_queue(lock, in_q, func, out_q, sep):
    while True:
        x = in_q.get()

        if x is None:
            out_q.put(x)
            return

        result = func(x)
        out_q.put(str(result) + sep)


def _write_to_output(out_q, stream, n_jobs):
    ends_seen = 0
    while True:
        x = out_q.get()
        if not x:
            ends_seen += 1
            if ends_seen == n_jobs:
                stream.flush()
                return
            else:
                continue
        stream.write(x)


def parallel_apply(func, iterable, n_jobs, sep='\n', out_stream=sys.stdout):
    """
    Writes the result of applying func to iterable using n_jobs to out_stream
    """
    # if there is only one job, simply read from iterable, apply function
    # and write to outpu
    if n_jobs == 1:
        for each in iterable:
            out_stream.write(str(func(each)) + sep)
        out_stream.flush()
        return

    # if there is more than one job, use a queue manager to communicate
    # between processes.
    manager = Manager()
    in_q = manager.Queue(maxsize=2 * n_jobs)
    out_q = manager.Queue(maxsize=2 * n_jobs)
    lock = Lock()

    # start pool workers
    pool = []
    for i in xrange(n_jobs):
        p = Process(target=_do_work_off_queue,
                    args=(lock, in_q, func, out_q, sep))
        p.start()
        pool.append(p)

    # start output worker
    out_p = Process(target=_write_to_output,
                    args=(out_q, out_stream, n_jobs))
    out_p.start()

    # put data on input queue
    iters = itertools.chain(iterable, (None,) * n_jobs)
    for each in iters:
        in_q.put(each)

    # finish job
    for p in pool:
        p.join()
    out_p.join()


def imap_easy(func, iterable, n_jobs, chunksize, ordered=True):
    """
    Returns a parallel iterator of func over iterable.

    Worker processes return one "chunk" of data at a time, and the iterator
    allows you to deal with each chunk as they come back, so memory can be
    handled efficiently.

    Parameters
    ----------
    func : Function of one variable
        You can use functools.partial to build this.
        A lambda function will not work
    iterable : List, iterator, etc...
        func is applied to this
    n_jobs : Integer
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
    chunksize : Integer
        Jobs/results will be sent between master/slave processes in chunks of
        size chunksize.  If chunksize is too small, communication overhead
        slows things down.  If chunksize is too large, one process ends up
        doing too much work (and large results will up in memory).
    ordered : Boolean
        If True, results are dished out in the order corresponding to iterable.
        If False, results are dished out in whatever order workers return them.

    Examples
    --------
    >>> from functools import partial
    >>> from rosetta.parallel.parallel_easy import imap_easy
    >>> def abfunc(x, a, b=1):
    ...     return x * a * b
    >>> some_numbers = range(3)
    >>> func = partial(abfunc, 2, b=3)
    >>> results_iterator = imap_easy(func, some_numbers, 2, 5)
    >>> for result in results_iterator:
    ...     print result
    0
    6
    12
    """
    n_jobs = _n_jobs_wrap(n_jobs)

    if n_jobs == 1:
        results_iter = itertools.imap(func, iterable)
    else:
        _trypickle(func)
        pool = Pool(n_jobs)
        if ordered:
            results_iter = pool.imap(func, iterable, chunksize=chunksize)
        else:
            results_iter = pool.imap_unordered(
                func, iterable, chunksize=chunksize)

    return results_iter


def map_easy(func, iterable, n_jobs):
    """
    Returns a parallel map of func over iterable.
    Returns all results at once, so if results are big memory issues may arise

    Parameters
    ----------
    func : Function of one variable
        You can use functools.partial to build this.
        A lambda function will not work
    iterable : List, iterator, etc...
        func is applied to this
    n_jobs : Integer
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    Examples
    --------
    >>> from functools import partial
    >>> from rosetta.parallel.parallel_easy import map_easy
    >>> def abfunc(x, a, b=1):
    ...     return x * a * b
    >>> some_numbers = range(5)
    >>> func = partial(abfunc, 2, b=3)
    >>> map_easy(func, some_numbers)
    [0, 6, 12, 18, 24]
    """
    n_jobs = _n_jobs_wrap(n_jobs)

    if n_jobs == 1:
        return map(func, iterable)
    else:
        _trypickle(func)
        pool = Pool(n_jobs)
        return pool.map_async(func, iterable).get(GOOGLE)


def map_easy_padded_blocks(func, iterable, n_jobs, pad, blocksize=None):
    """
    Returns a parallel map of func over iterable, computed by splitting
    iterable into padded blocks, then piecing the result together.

    Parameters
    ----------
    func : Function of one variable
        You can use functools.partial to build this.
        A lambda function will not work
    iterable : List, iterator, etc...
        func is applied to this
    n_jobs : Integer
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
    pad : Nonnegative Integer
        Each block is processed with pad extra on each side.
    blocksize : Nonnegative Integer
        If None, use 100 * pad

    Returns
    -------
    result : List
        Equivalent to list(func(iterable))

    Examples
    --------
    >>> numbers = [0, 0, 2, -1, 4, 2, 6, 7, 6, 9]
    >>> pad = 1
    >>> n_jobs = -1
    >>> def rightmax(mylist):
    ...     return [max(mylist[i: i+2]) for i in range(len(mylist))]
    >>> result = map_easy_padded_blocks(rightmax, numbers, n_jobs, pad)
    >>> benchmark = rightmax(numbers)
    >>> result == benchmark
    True
    """
    mylist = list(iterable)

    # We will pad each side of the blocks with this to avoid edge effects.
    max_blocksize = len(mylist) - pad - 1

    if blocksize is None:
        blocksize = min(max_blocksize, 100 * pad)

    assert pad + blocksize < len(mylist)

    # Get an iterator over padded blocks
    block_idx, pads_used = _get_split_idx(len(mylist), blocksize, pad=pad)
    block_iter = (mylist[start: end] for start, end in block_idx)

    # Process each block
    processed_blocks = map_easy(func, block_iter, n_jobs)

    result = []
    for block, (leftpad, rightpad) in zip(processed_blocks, pads_used):
        result += block[leftpad: len(block) - rightpad]

    return result


def _get_split_idx(N, blocksize, pad=0):
    """
    Returns a list of indexes dividing an array into blocks of size blocksize
    with optional padding.  Padding takes into account that the resultant block
    must fit within the original array.

    Parameters
    ----------
    N : Nonnegative integer
        Total array length
    blocksize : Nonnegative integer
        Size of each block
    pad : Nonnegative integer
        Pad to add on either side of each index

    Returns
    -------
    split_idx : List of 2-tuples
        Indices to create splits
    pads_used : List of 2-tuples
        Pads that were actually used on either side

    Examples
    --------
    >>> split_idx, pads_used = _get_split_idx(5, 2)
    >>> print split_idx
    [(0, 2), (2, 4), (4, 5)]
    >>> print pads_used
    [(0, 0), (0, 0), (0, 0)]

    >>> _get_split_idx(5, 2, pad=1)
    >>> print split_idx
    [(0, 3), (1, 5), (3, 5)]
    >>> print pads_used
    [(0, 1), (1, 1), (1, 0)]
    """
    num_fullsplits = N // blocksize
    remainder = N % blocksize

    split_idx = []
    pads_used = []
    for i in range(num_fullsplits):
        start = max(0, i * blocksize - pad)
        end = min(N, (i + 1) * blocksize + pad)
        split_idx.append((start, end))

        leftpad = i * blocksize - start
        rightpad = end - (i + 1) * blocksize
        pads_used.append((leftpad, rightpad))

    # Append the last split if there is a remainder
    if remainder:
        start = max(0, num_fullsplits * blocksize - pad)
        split_idx.append((start, N))

        leftpad = num_fullsplits * blocksize - start
        pads_used.append((leftpad, 0))

    return split_idx, pads_used


def _n_jobs_wrap(n_jobs):
    """
    For dealing with positive or negative n_jobs.

    Parameters
    ----------
    n_jobs : Integer

    Returns
    -------
    n_jobs_modified : Integer
        If -1, equal to multiprocessing.cpu_count() (all CPU's used).
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
    """
    if not isinstance(n_jobs, int):
        raise ValueError(
            "type(n_jobs) = %s, but n_jobs should be an int" % type(n_jobs))

    if (n_jobs == 0) or (n_jobs < -1 * cpu_count()):
        msg = "Must have -1 + cpu_count() <= n_jobs < 0  OR  1 <= n_jobs"
        raise ValueError("n_jobs = %d, but %s" % (n_jobs, msg))

    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)

    return n_jobs


def _imap_wrap(func):
    """
    Adds timeout to IMapIterator and IMapUnorderedIterator.
    This allows exit upon Ctrl-C.  This is a fix
    of the known python bug  bugs.python.org/issue8296 given by
    https://gist.github.com/aljungberg/626518

    Parameters
    ----------
    func : Either IMapIterator or IMapUnorderedIterator

    Returns
    -------
    wrap : Function
        Wrapped version of func, with timeout specified
    """
    # func will be a next() method of IMapIterator.
    # Note that the first argument to methods are always 'self'.
    def wrap(self, timeout=None):
        return func(self, timeout=timeout if timeout is not None else GOOGLE)
    return wrap


def _trypickle(func):
    """
    Attempts to pickle func since multiprocessing needs to do this.
    """
    genericmsg = "Pickling of func (necessary for multiprocessing) failed."

    boundmethodmsg = genericmsg + '\n\n' + """
    func contained a bound method, and these cannot be pickled.  This causes
    multiprocessing to fail.  Possible causes/solutions:

    Cause 1) You used a lambda function or an object's method, e.g.
        my_object.myfunc
    Solution 1) Wrap the method or lambda function, e.g.
        def func(x):
            return my_object.myfunc(x)

    Cause 2) You are pickling an object that had an attribute equal to a
        method or lambda func, e.g. self.myfunc = self.mymethod.
    Solution 2)  Don't do this.
    """

    try:
        cPickle.dumps(func)
    except TypeError as e:
        if 'instancemethod' in e.message:
            sys.stderr.write(boundmethodmsg + "\n")
        else:
            sys.stderr.write(genericmsg + '\n')
        raise
    except:
        sys.stderr.write(genericmsg + '\n')
        raise


# Redefine IMapUnorderedIterator so we can exit with Ctrl-C
IMapUnorderedIterator.next = _imap_wrap(IMapUnorderedIterator.next)
IMapIterator.next = _imap_wrap(IMapIterator.next)


if __name__ == '__main__':
    # Can't get doctest to work with multiprocessing...
    pass
