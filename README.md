# Payment's Pipeline

A Flask payment processing backend: user auth, payments through a real gateway, webhooks, fraud checks and an audit trail.

> **Status: under rebuild.** The app boots and its models create cleanly on SQLite and Postgres, but the payment routes are not production-safe yet (#2 #3 #4). Work is tracked on the [project board](https://github.com/users/CodersAcademy006/projects/11).

## Goal (MVP)

| Area     | Target                                                                                      | Issues            |
| -------- | ------------------------------------------------------------------------------------------- | ----------------- |
| Boot     | App factory, `db`, config from env vars, `/health`                                          | #1 #5 #16 #18 #19 |
| Auth     | Hashed passwords, identity from the session/token, never from the request body              | #2 #3             |
| Money    | Integer minor units plus currency, no floats                                                | #6                |
| Payments | State machine (`created -> pending -> succeeded / failed / refunded`) with idempotency keys | #4 #7             |
| Gateway  | Razorpay test mode, hosted checkout, signed webhooks as the only source of success          | #8 #29            |
| Ledger   | Double-entry rows for every money movement, plus reconciliation                             | #24 #25           |
| Refunds  | Full and partial refunds, dispute tracking                                                  | #26               |
| Risk     | Rule-based velocity and amount checks                                                       | #27               |
| Audit    | Append-only audit log on every state change                                                 | #28               |
| Ops      | Tests on the money paths, pinned deps, migrations, logs and metrics                         | #12 #15 #30 #32   |

Earlier scaffold modules (blockchain, IoT, quantum crypto and others) were removed because they are outside the MVP. They remain in git history before the restructure commit.

## Layout

```
backend/
  app/          Flask app factory
  api/          blueprints: auth, payment, notifications
  models/       SQLAlchemy models: user, transaction, audit_log, idempotency_key
  schemas/      request/response validation (pydantic)
  services/     idempotency and rate-limit logic
  tests/
  config.py
  wsgi.py
frontend/
  public/       index.html
  src/          React components, store, constants
.github/workflows/ci.yml   lint, format, whitespace, tests on SQLite and Postgres
```

## Run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export SECRET_KEY=change-me                      # required, the app refuses to start without it
export DATABASE_URL=postgresql+psycopg://...     # optional, defaults to sqlite:///payments.db
flask --app backend.wsgi run
```

## Checks

CI must pass before anything merges into `main`. A single stray space fails it. Run the same gate locally:

```bash
ruff check .                                     # lint: bugs, security, imports, style
ruff format --check .                            # Python formatting
npx prettier@3.6.2 --check .                     # JS, HTML, Markdown, YAML formatting
git diff --check "$(git hash-object -t tree /dev/null)" HEAD   # whitespace in every file
python -m pytest                                 # tests, warnings count as failures
```

To fix formatting instead of checking it: `ruff format . && npx prettier@3.6.2 --write .`

## License

MIT, see [LICENSE](LICENSE).
