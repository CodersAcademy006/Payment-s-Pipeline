# Payment's Pipeline

A Flask payment processing backend: user auth, payments through a real gateway, webhooks, fraud checks and an audit trail.

> **Status: under rebuild. The app does not run yet.**
> The current code is an early scaffold. CI is red on purpose and lists the real bugs. Work is tracked on the [project board](https://github.com/users/CodersAcademy006/projects/11).

## Goal (MVP)

| Area | Target | Issues |
|---|---|---|
| Boot | App factory, `db`, config from env vars, `/health` | #1 #5 #16 #18 #19 |
| Auth | bcrypt passwords, identity from the session/token, never from the request body | #2 #3 |
| Money | Integer minor units plus currency, no floats | #6 |
| Payments | State machine (`created -> pending -> succeeded / failed / refunded`) with idempotency keys | #4 #7 |
| Gateway | Razorpay test mode, hosted checkout, signed webhooks as the only source of success | #8 #29 |
| Ledger | Double-entry rows for every money movement, plus reconciliation | #24 #25 |
| Refunds | Full and partial refunds, dispute tracking | #26 |
| Risk | Rule-based velocity and amount checks | #27 |
| Audit | Append-only audit log on every state change | #28 |
| Ops | Tests on the money paths, pinned deps, migrations, logs and metrics | #12 #15 #30 #32 |

Earlier scaffold modules (blockchain, IoT, quantum crypto and others) were removed because they are outside the MVP. They remain in git history before the restructure commit.

## Layout

```
backend/
  app/          Flask app factory
  api/          blueprints: auth, payment, notifications
  models/       SQLAlchemy models: user, transaction, audit_log
  schemas/      request/response validation (pydantic)
  services/     payment gateway, idempotency and rate-limit logic
  tests/
  config.py
  wsgi.py
frontend/
  public/       index.html
  src/          React components, store, constants
.github/workflows/ci.yml   lint for bug classes + pytest
```

## Checks

CI runs on every push to `main` and on every pull request. To run the same checks locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install ruff flask flask-sqlalchemy pytest
ruff check --select E9,F63,F7,F82 .
python -m pytest -q
```

## License

MIT, see [LICENSE](LICENSE).
