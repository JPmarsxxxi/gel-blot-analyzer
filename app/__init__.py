import os

# Windows + conda commonly link multiple copies of the OpenMP runtime (numpy's
# MKL build vs torch's bundled one); this avoids a hard crash on import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
