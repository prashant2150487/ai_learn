# Cookie-based authentication

Sessions are stored in the database; the browser receives an **HttpOnly** cookie (`session_token`). Predictions require a valid session.

## Layout

```text
app/auth/
├── constants.py      # Cookie name, TTL
├── security.py       # bcrypt password hashing
├── schemas.py        # Login / register / user DTOs
├── service.py        # Users, sessions, bootstrap demo user
└── dependencies.py   # get_current_user (reads cookie)

app/api/routes/auth.py   # HTTP endpoints
```

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/register` | No | Create account |
| POST | `/auth/login` | No | Sets session cookie |
| POST | `/auth/logout` | Cookie | Clears session |
| GET | `/auth/me` | Cookie | Current user |

Protected: `POST /predict`, `POST /predict/batch` (public: `/health`, `/model-info`).

## Default demo user

Created on first startup if no users exist (override via `.env`):

- Email: `demo@example.com`
- Password: `demo-pass-123`

## Example (PowerShell)

```powershell
# Login (cookie stored in session)
Invoke-WebRequest -Uri http://127.0.0.1:8000/auth/login -Method Post `
  -ContentType "application/json" `
  -Body '{"email":"demo@example.com","password":"demo-pass-123"}' `
  -SessionVariable session

# Authenticated predict
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" `
  -WebSession $session `
  -Body '{"MedInc":8.32,"HouseAge":41,"AveRooms":6.98,"AveBedrms":1.02,"Population":322,"AveOccup":2.55,"Latitude":37.88,"Longitude":-122.23}'
```

In Swagger (`/docs`), use **Authorize** is not used for cookies — log in via `POST /auth/login` in the same browser tab so the cookie is sent on later requests.

## Production

Set in `.env`:

```env
COOKIE_SECURE=true
DATABASE_URL=postgresql+psycopg2://...
```

Use HTTPS so secure cookies work.
