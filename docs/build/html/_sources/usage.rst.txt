Usage
=====

Installation
------------

To use the Multiproc package, first clone it from GitHub:

.. code-block:: console

   $ git clone -o github https://github.com/JPBureik/Multiproc.git

Then create a virtual environment and install it using pip:

.. code-block:: console

   $ cd Multiproc
   $ python3 -m virtualenv multiprocenv
   $ source multiprocenv/bin/activate
   (.multiprocenv) $ pip install -e .

For any usage it is highly recommended that you use the
latest version of the code on the `stable` branch.
To make sure you're on the correct branch and up-to-date, run:

.. code-block:: console

   $ git checkout stable
   $ git pull github stable
