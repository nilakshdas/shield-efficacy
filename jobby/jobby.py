import os
import random
import socket
import string
import sys
import time

from celery import Celery
from celery.signals import celeryd_after_setup
import libtmux
from libtmux.exc import TmuxSessionExists

from .utils import get_env


APP_NAME = 'jobby'
TASK_NAME = 'run_script'
STABLE_STATE = 'RUNNING'
BROKER_URL = get_env('JOBBY_BROKER_URL')
BACKEND_URL = get_env('JOBBY_BACKEND_URL')
DATABASE_URL = get_env('JOBBY_DATABASE_URL')
RUNTIME_ENV = get_env('JOBBY_PYTHON_RUNTIME_ENV')
JOBBY_JOBS_DIR = get_env('JOBBY_JOBS_DIR')
JOBBY_SCRATCH_DIR = get_env('JOBBY_SCRATCH_DIR')
JOBBY_LOGS_DIR = os.path.join(JOBBY_SCRATCH_DIR, 'logs')
JOBBY_LOCKS_DIR = os.path.join(JOBBY_SCRATCH_DIR, 'locks')
CUDA_VISIBLE_DEVICES = get_env('CUDA_VISIBLE_DEVICES', required=False)


assert os.path.isdir(JOBBY_JOBS_DIR)
if not os.path.exists(JOBBY_LOGS_DIR):
    os.makedirs(JOBBY_LOGS_DIR)
if not os.path.exists(JOBBY_LOCKS_DIR):
    os.makedirs(JOBBY_LOCKS_DIR)

app = Celery(APP_NAME, broker=BROKER_URL, backend=BACKEND_URL)
app.conf.update(worker_prefetch_multiplier=1, 
                worker_send_task_events=True)

new_session_id = lambda: 'jobby-%s-%s' % (
    time.strftime('%Y%m%dT%H%M%S'),
    ''.join(random.sample(string.ascii_lowercase, 4)))


@celeryd_after_setup.connect
def wait_for_networking_services(sender, instance, **kwargs):
    if 'master' in sender:
        wait = 45
        print('Waiting %ds for networking services to spin up...' % wait)
        time.sleep(wait)


@app.task(bind=True)
def run_script(self, script_path, script_args):
    print('Running "%s" with "%s"' % (script_path, script_args))

    hostname = socket.gethostname()
    self.update_state(
        state='STARTED',
        meta={'hostname': hostname})

    script_full_path = os.path.join(JOBBY_JOBS_DIR, script_path)
    assert os.path.isfile(script_full_path)

    server = libtmux.Server()
    log_file, lock_file = None, None
    session, session_id = None, None
    
    while True:
        session_id = new_session_id()
        try:
            session = server.new_session(session_id)

            log_file = os.path.join(JOBBY_LOGS_DIR, '%s.log' % session_id)
            assert not os.path.exists(log_file)

            lock_file = os.path.join(JOBBY_LOCKS_DIR, '%s.lock' % session_id)
            with open(lock_file, 'w') as f:
                f.write(self.request.id)

        except (TmuxSessionExists, AssertionError):
            time.sleep(1)

        else:
            break

    assert session is not None
    assert os.path.isfile(lock_file)

    activate_env_cmd = 'source activate %s' % RUNTIME_ENV
    run_script_cmd = 'python %s %s' % (script_full_path, script_args)

    session.attached_pane.send_keys(activate_env_cmd)
    time.sleep(5)  # Waiting for env to get activated
    session.attached_pane.send_keys(' '.join([
        'export CUDA_VISIBLE_DEVICES=%s;' % (CUDA_VISIBLE_DEVICES,),
        '%s |& tee %s;' % (run_script_cmd, log_file),
        'rm -f %s;' % (lock_file,),
        'exit']))

    while os.path.exists(lock_file):
        self.update_state(
            state='RUNNING',
            meta={'hostname': hostname,
                  'session_id': session_id})

        time.sleep(5)

    self.update_state(
        state='FINISHED',
        meta={'hostname': hostname,
              'session_id': session_id})

    return hostname, session_id


if __name__ == '__main__':
    script_path = sys.argv[1]
    arg_str = '' if len(sys.argv) == 2 \
        else ' '.join(sys.argv[2:])

    result = app.send_task(
        '%s.%s' % (APP_NAME, TASK_NAME),
        args=(script_path, arg_str))

    start, timeout = time.time(), 30
    while True:
        now = time.time()
        if (now - start > timeout
                or result.state == STABLE_STATE):
            print(result.info)
            break
        time.sleep(1)
