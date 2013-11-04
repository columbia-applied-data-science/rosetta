from optparse import OptionParser
import os
import pandas as pd
import numpy as np
import statsmodels.iolib.foreign as smio


def main():
    r"""
    Converts STATA files to a Pytables store

    Examples
    ---------
    Convert every *.dta file to an object in the store mystore.h5
    $ python STATA_converter.py -s mystore.h5  data/*.dta
    """
    usage = "usage: %prog [options] dataset"
    usage += '\n'+main.__doc__
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-t", "--usetable",
        help="Use table format for storing (slower/larger/queryable)."
        " [default: %default] ",
        action="store_true", dest='usetable', default=False)
    parser.add_option(
        "-f", "--float_na_rep",
        help="Convert all values to float and convert this value to NaN"
        " [default: %default] ",
        action="store", dest="float_na_rep", type='float', default=None)
    parser.add_option(
        "-s", "--storename",
        help="Use this name as the output HDF5 store.  [default: %default]",
        action="store", dest='storename', default='mystore.h5')
    parser.add_option(
        "-o", "--overwrite",
        help="If True and a key already exists, overwrite it.  If False and "
        " a key exists, do nothing",
        action="store_true", dest="overwrite", default=False)

    (options, infiles) = parser.parse_args()

    # Raise an exception if the length of args is less than 1
    assert len(infiles) >= 1

    ## Call the function that does the real work
    convert_files(infiles, **options.__dict__)


def convert_files(
    infiles, usetable=False, float_na_rep=None, storename='mystore.h5',
    overwrite=False):
    """
    """
    store = pd.HDFStore(storename)

    for filepath in infiles:
        # Get the key name
        pathname, filename = os.path.split(filepath)
        # Store keys cannot start with numbers
        basekey = filename.replace('.', '_')
        if basekey[0].isdigit():
            key = '/key_' + basekey
        else:
            key = '/' + basekey

        # If the key already exists...
        if key in store.keys():
            if overwrite:
                store.remove(key)
                frame = _get_frame(filepath, float_na_rep)
                store.put(key, frame, table=usetable)
            else:
                pass
        else:
            frame = _get_frame(filepath, float_na_rep)
            store.put(key, frame, table=usetable)

    store.close()


def _get_frame(filepath, float_na_rep):
    # Convert to record array
    arr = smio.genfromdta(filepath)

    # Convert to DataFrame
    frame = pd.DataFrame.from_records(arr)

    # If float_na_rep, cast as float, replace float_na_rep with NaN
    if float_na_rep:
        frame = frame.astype('float').replace(float_na_rep, np.nan)

    return frame
        


if __name__ == '__main__':
    main()
