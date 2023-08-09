import os

NUM_WORKERS = os.cpu_count() - 1
print(NUM_WORKERS)
NUM_WORKERS = len(os.sched_getaffinity(0)) - 1
print(NUM_WORKERS)