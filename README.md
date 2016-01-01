# astromalign
Astrometric alignment of image stacks


This code depends on the Astrometry.net python libraries, which
can be installed from the latest release:
http://astrometry.net/downloads/astrometry.net-latest.tar.gz
or from github:
https://github.com/dstndstn/astrometry.net

You can either install it:
- make
- make py
- make install INSTALL_DIR=/place/to/install
- # add /place/to/install/lib/python to your PYTHONPATH

or build a local copy by downloading or cloning into a directory
(renamed to `astrometry`) within the `astromalign` directory.  You
need to `make && make py` in the `astrometry` directory to build the
python libraries.

