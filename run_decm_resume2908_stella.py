"""Free-core-aware scheduler for the 2026-08-29 DECM resume batch.

Same politeness model as run_decm_batch2608_stella.py (re-measures idle
cores via /proc/stat before every launch, safety margin, taskset+nice+
single-thread env), but drives decm_dico_calculator_resume2908.py (warm
start from the existing checkpoint) over just the 3 networks that only
looked like they needed more time: quirinale_dico2, crisi_dico1,
ita_elections_dico0.

Usage (on stella, inside tmux so it survives disconnects):
    cd /home/sarawalk/bowtie2_py39/bowtie2
    /home/sarawalk/bowtie2_py39/bin/python3.9 run_decm_resume2908_stella.py
"""
import os
import sys
import time
import subprocess
import datetime as dt

HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
PYTHON = '/home/sarawalk/bowtie2_py39/bin/python3.9'
WORKER = HOME + 'decm_dico_calculator_resume2908.py'
LOG_DIR = HOME + 'logs/'

SAFETY_MARGIN_CORES = 2
POLL_INTERVAL_S = 60
IDLE_SAMPLE_WINDOW_S = 1.0
IDLE_THRESHOLD = 0.85

TARGETS = [
    ('ita_elections', 0, 50288),
    ('crisi',         1, 31874),
    ('quirinale',     2, 18704),
]

MASTER_LOG = LOG_DIR + 'decm_resume2908_scheduler_log.txt'


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
    pending = list(TARGETS)
    running = {}

    mlog(f'=== resume2908 scheduler starting, {len(pending)} networks queued ===')

    while pending or running:
        for key in list(running.keys()):
            info = running[key]
            rc = info['proc'].poll()
            if rc is not None:
                elapsed = time.time() - info['start']
                mlog(f'finished {key[0]}_dico{key[1]} on core {info["core"]} '
                     f'(exit={rc}, wall={elapsed/3600:.2f}h)')
                info['logfh'].close()
                del running[key]

        free_cores = free_core_ids()
        usable = [c for c in free_cores if c not in {info['core'] for info in running.values()}]
        capacity = max(0, len(usable) - SAFETY_MARGIN_CORES)

        launched_this_round = 0
        while capacity > 0 and pending:
            dataset, dico, n = pending.pop(0)
            core = usable[launched_this_round]
            tag = f'{dataset}_dico{dico}'
            logpath = LOG_DIR + f'decm_resume2908_{tag}_log.txt'
            logfh = open(logpath, 'a')

            env = dict(os.environ)
            env.update({
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1',
                'NUMBA_NUM_THREADS': '1',
            })

            cmd = ['taskset', '-c', str(core), 'nice', '-n', '19',
                   PYTHON, WORKER, '--dataset', dataset, '--dico', str(dico)]

            proc = subprocess.Popen(cmd, cwd=HOME, env=env,
                                     stdout=logfh, stderr=subprocess.STDOUT,
                                     start_new_session=True)

            running[(dataset, dico)] = {
                'proc': proc, 'core': core, 'start': time.time(), 'logfh': logfh,
            }
            mlog(f'launched {tag} (N={n:,}) on core {core} '
                 f'({len(pending)} pending, {len(running)} running)')

            capacity -= 1
            launched_this_round += 1

        time.sleep(POLL_INTERVAL_S)

    mlog('=== resume2908 scheduler done, all 3 networks finished or timed out ===')


if __name__ == '__main__':
    main()
