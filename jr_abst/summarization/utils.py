from __future__ import with_statement
from contextlib import contextmanager
import collections
import logging
import warnings

try:
    from html.entities import name2codepoint as n2cp
except ImportError:
    from htmlentitydefs import name2codepoint as n2cp
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq

import numpy as np
import numbers
import scipy.sparse

from six import iterkeys, iteritems, itervalues, u, string_types, unichr
from six.moves import range

#from smart_open import open

from multiprocessing import cpu_count

if sys.version_info[0] >= 3:
    unicode = str

logger = logging.getLogger(__name__)


PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)

NO_CYTHON = RuntimeError(
    "Cython extensions are unavailable. "
    "Without them, this gensim functionality is disabled. "
    "If you've installed from a package, ask the package maintainer to include Cython extensions. "
    "If you're building gensim from source yourself, run `python setup.py build_ext --inplace` "
    "and retry. "
)
"""An exception that gensim code raises when Cython extensions are unavailable."""

def tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors="strict", to_lower=False, lower=False):
    """Iteratively yield tokens as unicode strings, optionally removing accent marks and lowercasing it.
    Parameters
    ----------
    text : str or bytes
        Input string.
    deacc : bool, optional
        Remove accentuation using :func:`~gensim.utils.deaccent`?
    encoding : str, optional
        Encoding of input string, used as parameter for :func:`~gensim.utils.to_unicode`.
    errors : str, optional
        Error handling behaviour, used as parameter for :func:`~gensim.utils.to_unicode`.
    lowercase : bool, optional
        Lowercase the input string?
    to_lower : bool, optional
        Same as `lowercase`. Convenience alias.
    lower : bool, optional
        Same as `lowercase`. Convenience alias.
    Yields
    ------
    str
        Contiguous sequences of alphabetic characters (no digits!), using :func:`~gensim.utils.simple_tokenize`
    Examples
    --------
    .. sourcecode:: pycon
        >>> from gensim.utils import tokenize
        >>> list(tokenize('Nic nemuže letet rychlostí vyšší, než 300 tisíc kilometru za sekundu!', deacc=True))
        [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']
    """
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)



def deprecated(reason):
    """Decorator to mark functions as deprecated.
    Calling a decorated function will result in a warning being emitted, using warnings.warn.
    Adapted from https://stackoverflow.com/a/40301488/8001386.
    Parameters
    ----------
    reason : str
        Reason of deprecation.
    Returns
    -------
    function
        Decorated function
    """
    if isinstance(reason, string_types):
        def decorator(func):
            fmt = "Call to deprecated `{name}` ({reason})."

            @wraps(func)
            def new_func1(*args, **kwargs):
                warnings.warn(
                    fmt.format(name=func.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                return func(*args, **kwargs)

            return new_func1
        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func = reason
        fmt = "Call to deprecated `{name}`."

        @wraps(func)
        def new_func2(*args, **kwargs):
            warnings.warn(
                fmt.format(name=func.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return new_func2

    else:
        raise TypeError(repr(type(reason)))


def has_pattern():
    """Check whether the `pattern <https://github.com/clips/pattern>`_ package is installed.
    Returns
    -------
    bool
        Is `pattern` installed?
    """
    try:
        from pattern.en import parse  # noqa:F401
        return True
    except ImportError:
        return False


def effective_n_jobs(n_jobs):
    """Determines the number of jobs can run in parallel.
    Just like in sklearn, passing n_jobs=-1 means using all available
    CPU cores.
    Parameters
    ----------
    n_jobs : int
        Number of workers requested by caller.
    Returns
    -------
    int
        Number of effective jobs.
    """
    if n_jobs == 0:
        raise ValueError('n_jobs == 0 in Parallel has no meaning')
    elif n_jobs is None:
        return 1
    elif n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    return n_jobs