Usage
=====

Installation
------------

To use the Multiproc package, first clone it from GitHub:

.. code-block:: console

   $ git clone -o github https://github.com/JPBureik/Multiproc.git

Then create a virtual environment and install it using :code:`pip`:

.. code-block:: console

   $ cd Multiproc
   $ python3 -m virtualenv multiprocenv
   $ source multiprocenv/bin/activate
   (.multiprocenv) $ pip install -e .

For use in an existing project, :code:`pip` install the package directly from GitHub:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/JPBureik/Multiproc.git@master