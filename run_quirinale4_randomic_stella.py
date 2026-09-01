"""Free-core-aware launcher for the single quirinale_dico4 random-IC
diagnostic (decm_quirinale4_randomic_diag.py). Same politeness model as
the other schedulers, just a single target.

Usage (on stella, inside tmux so it survives disconnects):
    cd /home/sarawalk/bowtie2_py39/bowtie2
    /home/sarawalk/bowtie2_py39/bin/python3.9 run_quirinale4_randomic_stella.py
"""
import os
import sys
import time
import subprocess
import datetime as dt

HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
PYTHON = '/home/sarawalk/bowtie2_py39/bin/python3.9'
WORKER = HOME + 'decm_quirinale4_randomic_diag.py'
LOG_DIR = HOME + 'logs/'

SAFETY_MARGIN_CORES = 2
POLL_INTERVAL_S = 60
IDLE_SAMPLE_WINDOW_S = 1.0
IDLE_THRESHOLD = 0.85

MASTER_LOG = LOG_DIR + 'decm_q4randomic_scheduler_log.txt'


def mlog(msg):
    line = f'[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print(line, flush=True)
    with open(MASTER_LOG, 'a') as f:
        f.write(line + '\n')


def read_cpu_times():
    times = {}
    with open('/proc/stat') as f:
        for line in f:
            if not line.startswith('cpu'):
                break
            parts = line.split()
            label = parts[0]
            if label == 'cpu':
                continue
            vals = list(map(int, parts[1:]))
            user, nice, system, idle, iowait, irq, softirq, steal = vals[:8]
            busy = user + nice + system + irq + softirq + steal
            times[label] = (busy, idle + iowait)
    return times


def free_core_ids():
    t0 = read_cpu_times()
    time.sleep(IDLE_SAMPLE_WINDOW_S)
    t1 = read_cpu_times()

    free = []
    for label in t0:
        busy0, idle0 = t0[label]
        busy1, idle1 = t1[label]
        d_busy = busy1 - busy0
        d_idle = idle1 - idle0
        total = d_busy + d_idle
        idle_frac = d_idle / total if total > 0 else 1.0
        if idle_frac >= IDLE_THRESHOLD:
            core_id = int(label.replace('cpu', ''))
            free.append(core_id)
    return sorted(free)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    mlog('=== q4randomic scheduler starting, 1 job queued ===')

    while True:
        free_cores = free_core_ids()
        capacity = max(0, len(free_cores) - SAFETY_MARGIN_CORES)
        if capacity > 0:
            core = free_cores[0]
            logpath = LOG_DIR + 'decm_q4randomic_log.txt'
            logfh = open(logpath, 'a')

            env = dict(os.environ)
            env.update({
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1',
                'NUMBA_NUM_THREADS': '1',
            })

            cmd = ['taskset', '-c', str(core), 'nice', '-n', '19', PYTHON, WORKER]

            proc = subprocess.Popen(cmd, cwd=HOME, env=env,
                                     stdout=logfh, stderr=subprocess.STDOUT,
                                     start_new_session=True)

            mlog(f'launched quirinale_dico4_randomic_diag on core {core}')
            rc = proc.wait()
            mlog(f'finished quirinale_dico4_randomic_diag (exit={rc})')
            logfh.close()
            break

        time.sleep(POLL_INTERVAL_S)

    mlog('=== q4randomic scheduler done ===')


if __name__ == '__main__':
    main()
