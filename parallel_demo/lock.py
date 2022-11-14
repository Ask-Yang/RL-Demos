import threading
from contextlib import contextmanager


_local = threading.local()


@contextmanager
def acquire(*locks):
    locks = sorted(locks, key=lambda x: id(x))
    lock_acquired = getattr(_local, 'lock_acquired', [])
    if lock_acquired and max(id(lock) for lock in lock_acquired) >= id(locks[0]):
        raise RuntimeError('Lock Order Violation')
    lock_acquired.extend(locks)
    _local.acquired = lock_acquired

    try:
        for lock in locks:
            lock.acquire()
        yield
    finally:
        for lock in reversed(locks):
            lock.release()
        del lock_acquired[-len(locks):]


def philosopher(left, right):
    i=0
    while i<100:
        with acquire(left, right):
             print(threading.currentThread().getName(), ' philosopher is eating')
             i+=1


num_sticks = 10
chopsticks = [threading.Lock() for n in range(num_sticks)]

for n in range(num_sticks):
    t = threading.Thread(target=philosopher,
                         args=(chopsticks[n], chopsticks[(n+1) % num_sticks]))
    t.start()