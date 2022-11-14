import gzip
import io
import glob


def find_errors(filename):
    errors = set()
    f = open(filename, encoding='utf-8')
    line = f.readline()
    while line:
        print(line)
        line = f.readline()
    f.close()
    return errors


def find_all_robots(logdir):
    '''
    Find all hosts across and entire sequence of files
    '''
    files = glob.glob(logdir+'/*.log.gz')
    all_robots = set()
    for robots in map(find_robots, files):
        all_robots.update(robots)
    return all_robots

if __name__ == '__main__':
    robots = find_all_robots('logs')
    for ipaddr in robots:
        print(ipaddr)