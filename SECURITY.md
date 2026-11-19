# Credentials and local assets

Use `.env` or environment variables for Gemini and DRES credentials. Never commit
real credentials, session files, notebooks containing outputs, datasets or model
weights. `.env.example` contains placeholders only. The optional Gemini UI stores
a supplied key in browser localStorage; use a private local browser profile and
clear that value on shared machines. Run the app locally; do not expose the
competition backend as an authenticated public service.

Install the local pre-commit guard after creating the retrieval virtual environment:

```sh
git config core.hooksPath .githooks
```

Run `python tools/check_secrets.py --staged` to inspect staged contents, or use
`--history` to inspect all locally reachable refs. The guard reports locations,
not secret values. GitHub Actions runs the history guard and its offline tests.
Pattern checks reduce risk but cannot guarantee detection of all secrets.

Previously published keys were revoked by their owner and the public history was
filtered. Old clones, forks, pull-request refs or caches may retain old objects.
Use a fresh clone after the rewrite. Do not merge or push branches from old clones.
If another credential is exposed, revoke it first and contact the maintainer
privately without placing the credential in a public issue.
