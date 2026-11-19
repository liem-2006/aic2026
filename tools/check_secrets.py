"""Offline secret guard. Reports locations and rule names, never matched values."""
import argparse
import re
import subprocess
import sys

RULES = {
    'google-aq': rb'AQ\.[A-Za-z0-9_-]{20,}',
    'google-api': rb'AIza[0-9A-Za-z_-]{35}',
    'github-token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
    'openai-token': rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}',
    'huggingface-token': rb'\bhf_[A-Za-z0-9]{30,}',
    'aws-access': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'private-key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'credential-url': rb'https?://[^\s/:]+:[^\s/@]{8,}@',
}
ASSIGNMENT = re.compile(
    rb'''(?i)(?:api[_-]?key|secret|password|access[_-]?token|auth[_-]?token)\s*["']?\s*[:=]\s*["']([^"'\r\n]{12,})["']''')

def findings(data):
    for name, pattern in RULES.items():
        for match in re.finditer(pattern, data):
            yield name, data[:match.start()].count(b'\n') + 1
    for match in ASSIGNMENT.finditer(data):
        value = match.group(1)
        if (re.fullmatch(rb'[A-Za-z0-9_.+/=-]{20,}', value)
                and not any(c in value.lower() for c in (b'example', b'your_', b'placeholder', b'removed_', b'process.env'))
                and re.search(rb'[A-Za-z]', value) and re.search(rb'[0-9]', value)):
            yield 'hardcoded-credential', data[:match.start()].count(b'\n') + 1

def git(*args):
    return subprocess.check_output(['git', *args])

def scan(history=False, staged=False):
    failures = 0
    seen = set()
    if history:
        for line in git('rev-list', '--objects', '--all').splitlines():
            oid, _, path = line.partition(b' ')
            if git('cat-file', '-t', oid.decode()).strip() != b'blob':
                continue
            seen.add((oid.decode(), path.decode('utf-8', 'replace')))
        for commit in git('rev-list', '--all').splitlines():
            data = git('show', '-s', '--format=%B', commit.decode())
            for rule, line in findings(data):
                print(f'{commit.decode()[:12]} commit-message:{line}: {rule}'); failures += 1
    elif staged:
        for entry in git('ls-files', '--stage', '-z').split(b'\0'):
            if entry:
                meta, path = entry.split(b'\t', 1)
                seen.add((meta.split()[1].decode(), path.decode('utf-8', 'replace')))
    else:
        for entry in git('ls-tree', '-r', '-z', 'HEAD').split(b'\0'):
            if entry:
                meta, path = entry.split(b'\t', 1)
                if meta.split()[1] == b'blob':
                    seen.add((meta.split()[2].decode(), path.decode('utf-8', 'replace')))
    for oid, path in sorted(seen):
        data = git('cat-file', 'blob', oid)
        for rule, line in findings(data):
            print(f'{oid[:12]} {path}:{line}: {rule}'); failures += 1
    print(f'Checked {len(seen)} blobs; {failures} findings (values redacted).')
    return bool(failures)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--history', action='store_true')
    mode.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    sys.exit(scan(args.history, args.staged))
