================================================================
Psyop
================================================================

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge| |torchapp badge|

.. |testing badge| image:: https://github.com/rbturnbull/psyop/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/psyop/actions

.. |docs badge| image:: https://github.com/rbturnbull/psyop/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/psyop
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/d3a9e5f1b7d7b8593c9df1cd46fe7557/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/psyop/coverage/

.. |torchapp badge| image:: https://img.shields.io/badge/torch-app-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/
    
.. end-badges

.. start-quickstart

Parameter Space Yield Optimizer

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/psyop.git


Command Line Usage
==================================

See the options for training a model with the command:

.. code-block:: bash

    psyop-tools train --help

See the options for making predictions with the command:

.. code-block:: bash

    psyop --help

See other available commands with:

.. code-block:: bash

    psyop-tools --help

Python API Usage
==================================

You can train the model using the Python API:
.. code-block:: python

    from psyop import Psyop

    psyop = Psyop()
    
    psyop.train(...)

You can also make predictions using the Python API:


.. code-block:: python

    psyop(...)

.. end-quickstart


Credits
==================================

.. start-credits

Robert Turnbull
For more information contact: <robert.turnbull@unimelb.edu.au>

Created using torchapp (https://github.com/rbturnbull/torchapp).

.. end-credits

