import os
import re
import sys
import subprocess
import shutil
import sqlite3

try:
    from docx import opendocx, getdocumenttext
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
from pyth.plugins.rtf15.reader import Rtf15Reader
from pyth.plugins.plaintext.writer import PlaintextWriter
from unidecode import unidecode

###############################################################################
# Functions for converting various format files to .txt
###############################################################################
def file_to_txt(file_path, dst_dir, new_file_name=None, ret_fname=False, 
        clean_path=False):
    """
    Takes a file path and writes the file in txt format to dst_dir.
    If file is already .txt, then simply copies the file.

    Parameters
    ----------
    file_path : string
        file for processing
    dst_dir : string
        destination directory
    new_file_name : str or None
        if str will write to dst_dir/new_file_name.txt
    ret_fname : bool
        if True will return file_name for successfully processed files.
    clean_fname : bool
        will return a 'cleaned,' i.e. escape char free version of the path

    Notes
    -----
    Currently only support pdf, txt, rtf, doc and docx.

    """
    if clean_path:
        try:
            file_path = _filepath_clean(file_path)
        except IOError:
            sys.stdout.write('unable to clean file_name %s \n'%file_path)
    file_name = os.path.split(file_path)[1]
    name, ext = os.path.splitext(file_name)
    ext = re.sub(r'\.', '', ext)
    if new_file_name:
        file_name = new_file_name
    converter_func_name = '_%s_to_txt'%ext
    if converter_func_name in globals().keys():
        #calls one of the _to_txt()
        out = eval(converter_func_name)(file_path, dst_dir, file_name) 
        if out: sys.stdout.write('unable to process file %s'%file_path)
        if ret_fname: return file_name
    else:
        sys.stdout.write('file type %s not supported, skipping %s \n'%(ext,
            file_name))


def folder_to_sqlite(folder, outfile_path, batch_size=10000):
    """
    Convert all textfiles inside folder to sqlite file.

    Parameters
    ----------
    folder : String
        Name of folder containing text files.
    outfile_path : String
        Name of file to save.
    batch_size : int
        Number of files to be committed to sqlite file in each batch. A larger
        number allows faster disk usage, but requires more memory.
    """
    # Remove old file if necessary.
    if os.path.isfile(outfile_path):
        os.remove(outfile_path)

    filenames = os.listdir(folder)
    num_files = len(filenames)

    with sqlite3.connect(outfile_path) as conn:
        conn.text_factory = str
        c = conn.cursor()
        query = ("CREATE TABLE rosetta_file "
                 "(filename text UNIQUE, contents text)")
        c.execute(query)

        offset = 0
        batch = batch_size
        while offset < num_files:
            pairs = []
            for filename in filenames[offset: offset + batch]:
                path = os.path.join(folder, filename)
                with open(path) as f:
                    pairs.append((filename, f.read()))

            c.executemany('INSERT INTO rosetta_file VALUES (?, ?)', pairs)
            conn.commit()
            offset += batch


def records_to_sqlite(records, outfile_path):
    '''
    Convert records into a sqlite file.

    Parameters
    ----------
    records : 2-tuple (filename : string, contents : string)
        The filenames must be unique.
    oufile : string
        Path to save rosetta file to.
    '''
    # Remove old file if necessary.
    if os.path.isfile(outfile_path):
        os.remove(outfile_path)

    with sqlite3.connect(outfile_path) as conn:
        conn = sqlite3.connect(outfile_path)
        conn.text_factory = str
        c = conn.cursor()
        query = ("CREATE TABLE rosetta_file "
                 "(filename text UNIQUE, contents text)")
        c.execute(query)

        c.executemany('INSERT INTO rosetta_file VALUES (?, ?)', records)
        conn.commit()


def _filepath_clean(file_path):
    """
    replaces chars which need to be escaped with a '_' in the filename;

    Returns
    -------
    file_name : str
        clean file name

    """
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    if re.search(r'[,\s|:\'\.]', name):
        clean_name = re.sub(r'[,\s|:\'\.]', '_', name)
        clean_file_name = clean_name + ext
        clean_file_path = os.path.join(dir_name, clean_file_name)
    else:
        clean_file_path = file_path
    return clean_file_path


def _filepath_clean_copy(file_path):
    """
    creates a copy of the file with chars which need to be escaped
    replaced with a '_';

    Returns
    -------
    file_name : str
        clean file name

    """
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    if re.search(r'[,\s|:\'\.]', name):
        clean_name = re.sub(r'[,\s|:\'\.]', '_', name)
        clean_file_name = clean_name + ext
        clean_file_path = os.path.join(dir_name, clean_file_name)
        shutil.copyfile(file_path, clean_file_path)
    else:
        clean_file_path = file_path
    return clean_file_path


def _txt_to_txt(file_path, dst_dir, file_name):
    """
    Simply copies the file to the target dir.
    """
    if file_name is None:
        file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, file_name)
    return subprocess.call(['cp', file_path, file_dst])


def _pdf_to_txt(file_path, dst_dir, file_name):
    """
    Uses the pdftotxt unix util, with --layout option, to convert file_name
    to .txt and save in dst_dir

    Notes
    -----
    Download and install Xpdf from http://www.foolabs.com/xpdf/download.html
    Follow the instruciton in INSTALL - should work on most *nix systems.
    """
    if file_name is None:
        file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, re.sub(r'\.pdf$', '.txt', file_name))
    return subprocess.call(["pdftotext",  "-layout", file_path, file_dst])


def _doc_to_txt(file_path, dst_dir, file_name):
    """
    Uses catdoc unix util to convert file_name
    to .txt and save in dst_dir.

    Notes
    -----
    To install catdoc:
        apt-get catdoc on unix/linux
        brew install on mac
    """
    if file_name is None:
        file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, re.sub(r'\.doc$', '.txt', file_name))
    with open(file_dst, 'w') as f:
        return subprocess.call(["catdoc",  "-w", file_path, file_dst], stdout=f)


def _docx_to_txt(file_path, dst_dir, file_name):
    """
    Uses the docx python module to extract text from a docx file and save
    to .txt in dst_dir.
    """
    if not HAS_DOCX:
        raise ImportError(
            "docx was not importable, therefore _docx_to_txt cannot be used.")

    if file_name is None:
        file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, re.sub(r'\.docx$', '.txt', file_name))
    doc = opendocx(file_path)
    txt = '\n'.join(getdocumenttext(doc))
    txt = unidecode(txt)
    with open(file_dst, 'w') as f:
        f.write(txt)
    return 0


def _rtf_to_txt(file_path, dst_dir, file_name):
    """
    Uses the pyth python module to extract text from a rtf file and save
    to .txt in dst_dir.
    """
    if file_name is None:
        file_name = os.path.split(file_path)[1]
    file_dst = os.path.join(dst_dir, re.sub(r'\.rtf$', '.txt', file_name))
    doc = Rtf15Reader.read(open(file_path))
    txt = PlaintextWriter.write(doc).getvalue()
    txt = unidecode(txt)
    with open(file_dst, 'w') as f:
        f.write(txt)
    return 0
