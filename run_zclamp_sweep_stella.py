"""Free-core-aware scheduler for the ita_elections_dico1 z_clamp sweep.

Same politeness model as the other batch/resume schedulers. Drives
decm_zclamp_sweep_ita_dico1.py over 5 candidate z_clamp values, each
capped at SWEEP_HOURS (a diagnostic, not a convergence attempt).

Usage (on stella, inside tmux so it survives disconnects):
    cd /home/sarawalk/bowtie2_py39/bowtie2
    /home/sarawalk/bowtie2_py39/bin/python3.9 run_zclamp_sweep_stella.py
"""
import os
import sys
import time
import subprocess
import datetime as dt

HOME = '/home/sarawalk/bowtie2_py39/bowtie2/'
PYTHON = '/home/sarawalk/bowtie2_py39/bin/python3.9'
WORKER = HOME + 'decm_zclamp_sweep_ita_dico1.py'
LOG_DIR = HOME + 'logs/'

SAFETY_MARGIN_CORES = 2
POLL_INTERVAL_S = 60
IDLE_SAMPLE_WINDOW_S = 1.0
IDLE_THRESHOLD = 0.85

SWEEP_HOURS = 6

# both directions around the 1e-6 used in the batch, plus the pre-fix
# default (1e-8) as a reference point
Z_CLAMPS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

MASTER_LOG = LOG_DIR + 'decm_zsweep_scheduler_log.txt'


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
    pending = list(Z_CLAMPS)
    running = {}

    mlog(f'=== zsweep scheduler starting, {len(pending)} z_clamp values queued ===')

    while pending or running:
        for key in list(running.keys()):
            info = running[key]
            rc = info['proc'].poll()
            if rc is not None:
                elapsed = time.time() - info['start']
                mlog(f'finished z_clamp={key:.0e} on core {info["core"]} '
                     f'(exit={rc}, wall={elapsed/3600:.2f}h)')
                info['logfh'].close()
                del running[key]

        free_cores = free_core_ids()
        usable = [c for c in free_cores if c not in {info['core'] for info in running.values()}]
        capacity = max(0, len(usable) - SAFETY_MARGIN_CORES)

        launched_this_round = 0
        while capacity > 0 and pending:
            z_clamp = pending.pop(0)
            core = usable[launched_this_round]
            tag = f'zclamp_{z_clamp:.0e}'
            logpath = LOG_DIR + f'decm_zsweep_{tag}_log.txt'
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
                   PYTHON, WORKER, '--zclamp', str(z_clamp), '--hours', str(SWEEP_HOURS)]

            proc = subprocess.Popen(cmd, cwd=HOME, env=env,
                                     stdout=logfh, stderr=subprocess.STDOUT,
                                     start_new_session=True)

            running[z_clamp] = {
                'proc': proc, 'core': core, 'start': time.time(), 'logfh': logfh,
            }
            mlog(f'launched z_clamp={z_clamp:.0e} on core {core} '
                 f'({len(pending)} pending, {len(running)} running)')

            capacity -= 1
            launched_this_round += 1

        time.sleep(POLL_INTERVAL_S)

    mlog('=== zsweep scheduler done, all z_clamp candidates finished or timed out ===')


if __name__ == '__main__':
    main()
