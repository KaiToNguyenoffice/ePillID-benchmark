"""Wrapper that runs training and writes output to a log file."""
import subprocess
import sys
import os

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_output.log')

def main():
    args = sys.argv[1:]
    python = sys.executable
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_nocv.py')
    cmd = [python, '-u', script] + args

    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        log.write(f"CMD: {' '.join(cmd)}\n{'=' * 60}\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

        proc.wait()
        log.write(f"\n{'=' * 60}\nExit code: {proc.returncode}\n")

    sys.exit(proc.returncode)

if __name__ == '__main__':
    main()
