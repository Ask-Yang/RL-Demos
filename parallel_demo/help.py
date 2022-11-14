import threading
import gzip
import io
import glob
from concurrent import futures

help(type(futures.ProcessPoolExecutor()))