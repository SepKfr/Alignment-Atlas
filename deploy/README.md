# Deployment (systemd + Grobid)

This folder contains systemd units so you can run **Grobid** and the **Alignment Atlas** Streamlit app as services. The app expects Grobid at `http://localhost:8070` for PDF extraction (see `GROBID_URL` and `src/ingest/stages.py`).

## 1. Run Grobid all the time

Grobid is required for section-aware PDF parsing (unless you set `GROBID_REQUIRED=0`). To keep it running on the server:

**Prerequisites:** Docker installed.

```bash
# Copy unit and enable/start
sudo cp deploy/grobid.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable grobid
sudo systemctl start grobid
```

Optional: pull the image once so the first start is faster:

```bash
docker pull grobid/grobid:0.8.2-crf
```

Check that Grobid is up:

```bash
curl -s http://localhost:8070/api/isalive
```

Useful commands:

- `sudo systemctl status grobid`
- `sudo systemctl restart grobid`
- `sudo journalctl -u grobid -f`

## 2. Run the Streamlit app with systemd

So that the app starts **after** Grobid and keeps running:

1. Copy the app unit and adjust paths/user:

   ```bash
   sudo cp deploy/alignment-atlas.service /etc/systemd/system/
   ```

2. Edit the service to match your server (user, install path, venv):

   ```bash
   sudo systemctl edit alignment-atlas
   ```

   Or edit the file directly:

   ```bash
   sudo nano /etc/systemd/system/alignment-atlas.service
   ```

   Set at least:

   - **User** – user that owns the app and has the venv
   - **WorkingDirectory** – repo root (where `app.py` lives)
   - **ExecStart** – path to `streamlit` in your venv and any flags, e.g.  
     ` /path/to/Alignment-Atlas/.venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0`
   - **EnvironmentFile** – optional path to `.env` (e.g. `OPENAI_API_KEY`, `GROBID_URL`)

3. Enable and start:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable alignment-atlas
   sudo systemctl start alignment-atlas
   ```

The unit is configured with `After=grobid.service` and `Wants=grobid.service`, so:

- Grobid starts before the app when the server boots.
- If you start only `alignment-atlas`, systemd will start `grobid` too.

## 3. Summary

| Service            | Role                         | Port  |
|--------------------|------------------------------|-------|
| `grobid.service`   | PDF extraction (Docker)      | 8070  |
| `alignment-atlas.service` | Streamlit app            | 8501 (default) |

With both enabled, after a reboot Grobid and the app will be running and PDF extraction will work as long as `GROBID_URL` points at `http://localhost:8070` (or your Grobid host).

## 4. Without Docker (Grobid from JAR)

If you run Grobid from the official JAR instead of Docker, add a separate systemd unit that runs the JAR (e.g. `java -jar grobid-core.jar` with the right config) and keep `alignment-atlas.service` with `After=<your-grobid-unit>.service` and `Wants=<your-grobid-unit>.service` so the app still starts after Grobid.
