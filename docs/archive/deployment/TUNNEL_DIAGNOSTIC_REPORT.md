# Tunnel Diagnostic Report

Date/time: 2026-05-21 15:42 AEST

## Summary

VisionGuard local services are healthy. The failure is occurring before Cloudflare returns a new Quick Tunnel URL, while `cloudflared` requests `https://api.trycloudflare.com/tunnel`.

Final classification: Cloudflare Quick Tunnel API/network issue from the current network path.

Do not update Vercel `NEXT_PUBLIC_API_BASE_URL` until a new `https://<name>.trycloudflare.com` URL is created and both backend health endpoints pass through that URL.

## Local Backend Status

- `http://127.0.0.1:8000/api/realtime/health`: reachable, `ok: true`
- `http://127.0.0.1:8000/api/archive/health`: reachable, `ok: true`
- Archive database: `data/visionguard_archive.sqlite`
- Archive record count: `50`
- Backend listener: Python process on `127.0.0.1:8000`

## Local Frontend Status

- `http://localhost:3000`: reachable, HTTP `200`
- Frontend listener: Node process on port `3000`

## Existing/Old Tunnel Status

Tested old URLs:

- `https://lane-substance-train-millennium.trycloudflare.com`
- `https://november-insights-blue-ports.trycloudflare.com`
- `https://vermont-editions-birth-instantly.trycloudflare.com`

Result: all failed DNS resolution with `Could not resolve host`. No old tunnel is currently usable.

## Vercel Status

- Production frontend `https://visionguard-systemui.vercel.app/`: reachable, HTTP `200`
- Vercel environment was not modified.
- Vercel redeploy was not run.

## Cloudflare API Reachability

`curl -v https://api.trycloudflare.com/tunnel`:

- DNS resolution succeeded.
- IPv4 targets included `104.16.231.132` and `104.16.230.132`.
- TCP connection to `api.trycloudflare.com:443` succeeded.
- TLS handshake timed out.

`curl -I https://api.trycloudflare.com/tunnel`:

- Timed out connecting to port `443`.

`curl -X POST https://api.trycloudflare.com/tunnel`:

- Failed with SSL connection error or timeout.

General Cloudflare checks:

- `https://www.cloudflare.com`: reachable, HTTP `200`
- `https://developers.cloudflare.com`: timed out during this diagnostic run

Interpretation: the current network can reach some Cloudflare endpoints, but the path to the Quick Tunnel API and some Cloudflare developer endpoints is unstable or blocked during TLS/API access.

## DNS Results

Local resolver:

- Resolver: UTS DNS servers such as `138.25.16.8`
- `api.trycloudflare.com` resolved to:
  - `104.16.231.132`
  - `104.16.230.132`

Public resolvers:

- `dig @1.1.1.1 api.trycloudflare.com`: same A records
- `dig @8.8.8.8 api.trycloudflare.com`: same A records

Interpretation: DNS is not the primary issue. Local and public resolvers agree.

## TLS Results

`openssl s_client -connect api.trycloudflare.com:443 -servername api.trycloudflare.com -brief` produced no handshake result before being terminated after a long wait.

System clock:

- `Thu May 21 15:40:55 AEST 2026`

Interpretation: system clock is not obviously wrong. TLS handshake to `api.trycloudflare.com` is hanging from this network path.

## Proxy/VPN Observations

Environment:

- No `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` variables were set.

macOS Wi-Fi proxy settings:

- Web proxy: disabled
- Secure web proxy: disabled
- SOCKS proxy: disabled
- Stored local proxy addresses such as `127.0.0.1:10919` / `127.0.0.1:10036` were present but disabled.

Network interfaces:

- Multiple `utun` interfaces were present.

Interpretation: no active shell proxy was found. Multiple `utun` interfaces suggest VPN/security/network extension components may be present, but nothing was changed automatically.

## cloudflared Version/Source

- Installed `cloudflared` binary: not found in `PATH`
- Homebrew `cloudflared`: not installed
- `npx -y cloudflared --version`: `cloudflared version 2026.5.0`

Only the `npx` cloudflared package was tested.

## Quick Tunnel Attempt Logs Summary

Command:

```bash
npx -y cloudflared tunnel --url http://localhost:8000 --loglevel debug
```

Result:

```text
Requesting new quick Tunnel on trycloudflare.com...
failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": context deadline exceeded (Client.Timeout exceeded while awaiting headers)
```

The failure happens before a `trycloudflare.com` URL is created. No tunnel URL was generated, so no remote tunnel health check could be performed.

## Alternative Tunnel Check

- `ngrok`: not installed

No alternative tunnel was started.

## Recommended Next Action

Most likely cause: current network path to Cloudflare Quick Tunnel creation API is blocked, degraded, or intermittently failing at TLS/API request time.

Next command to run after switching network or waiting a few minutes:

```bash
npx -y cloudflared tunnel --url http://localhost:8000
```

If it returns a URL, immediately verify:

```bash
curl https://<new-tunnel-url>/api/realtime/health
curl https://<new-tunnel-url>/api/archive/health
```

Only after both checks pass should Vercel `NEXT_PUBLIC_API_BASE_URL` be updated and production redeployed.

Recommended network actions:

- Retry after a short wait.
- Switch to a phone hotspot or home network.
- Toggle VPN/security proxy tools manually if currently enabled.
- Consider installing the Homebrew `cloudflared` binary later:

```bash
brew install cloudflared
```

Do not make Vercel changes until a verified tunnel URL exists.
