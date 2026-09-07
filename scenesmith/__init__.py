"""SceneSmith package.

Importing this package applies compatibility patches required for the
installed dependency versions. See ``_compat.py`` for details.
"""

from ._compat import apply_compat_patches

apply_compat_patches()
