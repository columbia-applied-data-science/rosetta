README_data
===========

A scheme for managing data that works.

General guidelines
==================

* Don't version data.
* To avoid excess sharing of processed data (which changes often), it is preferable to share raw data and the scripts and notebooks that transform *raw* into *processed*.  
* Contents of any *raw* folder should never be modified or deleted.  This way, your script will create the same output as everyone else's script.

/data/ Scheme
=============

/data/ is mounted on the high speed SSD and is the preferred directory for project data.

Personal directories
--------------------
/data/username/clientname-XX
* Example: /data/ian/clientname-XX
* Example: /data/ian/anything-you-want
* Your personal data.  Use space wisely (e.g. don't copy raw or decoded data to here)

Production directories
----------------------
/data/prod/clientname-XX/
* Example: /data/prod/akin-01/
* Shared among all users
* These should be maintained with production ready data, raw and processed
* Use this for raw/decoded data

Shell scripts
-------------
Will set variables to direct certain input/output from/to personal and production directories.

Example 1:  Inside a shell script

    PRODDATA=$DATA/prod/akin-01 # Production directory
    MYDATA=$DATA/$ME/akin-01  # Personal directory

Shell variables
---------------
### The `DATA` variable points to the top-level data location

Example 1: On server, in your `.bashrc`

    export DATA=/data

Example 2: On laptop, in your `.bashrc`

    export DATA=$HOME/servername/data

### The `ME` variable should be set to select personal/production directory.

Example 1:  On laptop, in your `.bashrc`

    export ME=prod

Example 2:  On server, before developing and testing, in your `.bashrc`

    export ME=ian

Example 3:  On server, before production runs, type in your terminal (*never* in your `.bashrc`):

    export ME=prod


Source code scheme
==================

Locations
---------
Each user maintains his own sourcecode somewhere in $HOME.

Example 1 (Clone repo on the server)

    git clone https://github.com/jrl-labs/jrl_utils.git  $HOME/lib/jrl_utils

Example 2 (Develop locally and copy)

    scp -r  ~/lib/my-local-repo  servername:~/lib/my-remote-copy

Example 3 (Develop locally, copy diff, print progress, don't copy the repo)
(Make sure to use the trailing slash)

    rsync -az --progress --exclude=".git"  ~/lib/my-local-repo/  servername:~/lib/my-remote-copy/

Example 4 (set an alias in .bashrc to send every repo to server)

    alias syncrepos='rsync -az --progress --exclude=".git" ~/lib/ servername:~/lib/'

Shell variables
---------------
Add your sourcecode directory to your PYTHONPATH

    export PYTHONPATH=$HOME/lib

For each repo, add a shell variable pointing to the repo directory.  It should be an ALL CAPS version of the repo name.

    export JRL_UTILS=$HOME/lib/jrl_utils

Note:  You can always use aliases or other shell variables to make short names or other links.


Configuration files
-------------------
Keep configuration files in the repo under REPONAME/conf/.
* Version files before using them in a production run
* Keep a personal copy that you do not version (e.g. features_ian.conf)


Transferring files
==================

Using the internets
-------------------
On your laptop, tar the directory, split it into chunks of size 50MB, then rsync the (many many) files to the server's `raw-archives` directory:

    tar -czvf - mydir | split --bytes=50M - mydir.tar.gz.split_
    rsync -a --progress mydir.tar.gz.split_*  servername:/data/raw-archives/

On your server, reconstruct the directory

    cat mydir.tar.gz.split_* | tar -xzvf -




