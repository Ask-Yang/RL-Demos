import gzip
import io
import glob
from concurrent import futures


def find_writing(filename):
    writings = set()
    with gzip.open(filename) as f:
        for line in io.TextIOWrapper(f,encoding='ascii'):
            fields = line.split()
            if fields[6] == '/writing.html':
                writings.add(fields[0])
    return writings


def find_all_writings(logdir):
    files = glob.glob(logdir+'/*.log.gz')
    all_writings = set()
    with futures.ProcessPoolExecutor() as pool:
        for writings in pool.map(find_writing, files):
            all_writings.update(writings)
    return all_writings


if __name__ == '__main__':
    writings = find_all_writings('logs')
    for ipaddr in writings:
        print(ipaddr, " send a writing request")

