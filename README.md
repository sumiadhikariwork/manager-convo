# Conversation records

Upload the recording of a coaching or feedback conversation. The app transcribes
it, aligns the audio to the agenda the manager was working from, and fills in the
feedback form from what was actually said — every answer linked back to the
moment in the recording it came from.

The point is transparency. Today the manager's notes are written after the fact,
from memory, and the person coached rarely sees them. Here the record is drafted
from the recording itself, the manager reviews and edits it, and what changed
between the draft and the final version is on the record.

![The review screen: the filled-in form on the left, the aligned transcript and player on the right](docs/review.png)

---

## What it does

```
 recording ──▶ transcribe ──▶ align to the agenda ──▶ draft every field ──▶ manager reviews
                   │                  │                        │                    │
              timestamped        Open · Goals ·          notes, blockers,      edits tracked
               utterances       Reality · Way Forward    actions, feedback      against the draft
```

**Alignment.** The agenda runs in order and does not repeat, so placing the
conversation on it is a monotonic segmentation problem, not a classification
one. Both analysts solve it that way: the offline one with an exact dynamic
program over spoken cues and the planned running order, the Claude one by
picking boundary indices which are then repaired to be monotonic and in range no
matter what comes back.

**Drafting.** Each field in the template carries a brief — what a manager is
meant to record there — and that brief is what the model is asked to answer,
from that part of the conversation only.

**Citations are never generated.** The model returns *segment indices*, and the
application resolves them against the stored transcript. A quote on the record is
therefore always something that was genuinely said, at a timestamp you can click
and hear. This is enforced in `app/pipeline.py::_build_evidence` and covered by
`test_every_citation_matches_the_stored_transcript`.

**The draft and the edit are both kept.** A field stores what the pipeline wrote
(`draft_value`) and what the record currently says (`value`). Editing one never
overwrites the other, so the export can say plainly which entries the manager
changed, and re-running the pipeline leaves edited fields alone.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit it - see Configuration below

uvicorn app.main:app --reload # http://localhost:8000
```

### See it working without setting anything up

```bash
SPEECH_PROVIDER=fixture python scripts/seed_demo.py
uvicorn app.main:app --reload
```

This loads a worked example: a fifteen-minute GROW check-in between a manager
and a report, with the audio, the aligned transcript, the filled-in form and the
audit trail all populated. The audio is silent — it exists so the player,
seeking and the timeline are real; the words come from
`tests/fixtures/monthly_checkin.txt`.

### Transcribing real audio

```bash
pip install -r requirements-asr.txt   # faster-whisper
export SPEECH_PROVIDER=faster_whisper
```

Transcription runs locally, on the machine running the app. For recordings of
one-to-one conversations that is usually the deciding factor: the audio does not
leave your infrastructure. `WHISPER_MODEL=small` is a reasonable default;
`medium` is noticeably better on accented speech and cross-talk, and slower.

### Drafting with Claude

```bash
export ANALYSIS_PROVIDER=claude
export ANTHROPIC_API_KEY=sk-ant-...   # or run `ant auth login`
```

Without it, set `ANALYSIS_PROVIDER=heuristic` and the app runs entirely offline —
see [Degrading honestly](#degrading-honestly).

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `SPEECH_PROVIDER` | `faster_whisper` | or `fixture`, which reads a sidecar transcript next to the audio |
| `WHISPER_MODEL` | `small` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `WHISPER_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` on CPU, `float16` on GPU |
| `ANALYSIS_PROVIDER` | `claude` | or `heuristic` for the fully offline path |
| `ANTHROPIC_API_KEY` | — | unset is fine if you have used `ant auth login` |
| `CLAUDE_MODEL` | `claude-opus-5` | |
| `DATA_DIR` | `./data` | recordings and the SQLite database |
| `DATABASE_URL` | `sqlite:///./data/manager_convo.sqlite3` | any SQLAlchemy URL |
| `MAX_UPLOAD_MB` | `200` | |
| `STORAGE_BACKEND` | `local` | or `s3` for any S3-compatible bucket |
| `STORAGE_BUCKET` | — | required when `STORAGE_BACKEND=s3` |
| `STORAGE_ENDPOINT_URL` | — | set for R2/MinIO/B2; omit for AWS S3 |
| `STORAGE_REGION` | `auto` | |
| `STORAGE_ACCESS_KEY_ID` / `STORAGE_SECRET_ACCESS_KEY` | — | |
| `STORAGE_PUBLIC_BASE_URL` | — | set when the bucket is behind a CDN; playback then skips presigning |
| `SPEECH_PROVIDER` | `faster_whisper` | `assemblyai` for the hosted, webhook-driven path |
| `ASSEMBLYAI_API_KEY` | — | required when `SPEECH_PROVIDER=assemblyai` |
| `JOB_RUNNER` | `thread` | `deferred` on a host with no background execution |
| `PUBLIC_BASE_URL` | — | required for webhooks; falls back to `VERCEL_URL` |
| `WEBHOOK_SECRET` | — | shared with the transcription service; unset disables the check |
| `AUTO_CREATE_TABLES` | `true` | set false on serverless and run `scripts/migrate.py` |

---

## Templates

A template is the machine-readable version of the paper form. `app/templates.py`
ships two:

- **`grow-monthly-15`** — the monthly one-to-one: Open (2 min), Goals (5),
  Reality (4), Way Forward (4), plus a whole-conversation record block.
- **`sbi-feedback-20`** — a direct feedback conversation using
  Situation–Behaviour–Impact.

Adding one is a matter of describing it. Nothing else needs to change:

```python
SectionSpec(
    id="reality",
    title="Reality",
    minutes=4,
    prompt="What is in the way? What have you already tried?",
    cues=("in the way", "blocked", "i tried", "stuck"),   # for the offline aligner
    fields=(
        FormFieldSpec(id="notes", label="Notes", kind="text",
                      guidance="How the person described the situation."),
        FormFieldSpec(id="blockers", label="What is in the way", kind="list",
                      guidance="Each obstacle named, one per item."),
    ),
)
```

Field kinds are `text`, `list`, `actions` (action / owner / due / support) and
`choice`. `guidance` does double duty: it is the hint shown under the field in
the UI, and the brief the model is asked to answer.

![Uploading a recording, with the template's fields shown alongside](docs/upload.png)

---

## Degrading honestly

If the Claude call is unavailable or declines, the run does not fail. It
continues on the offline analyst, and the record says so — a banner in the UI, a
`degraded_reason` in the metrics, an entry in the audit trail, and a line in the
export.

The offline analyst is deliberately extractive: it quotes, it does not compose.
Every value it writes is lifted verbatim from the transcript, which is what makes
it safe to fall back to unattended. It is a floor, not a substitute — the
confidence it reports stays low on purpose, and re-running once Claude is
available replaces every field the manager has not edited.

---

## API

| Method | Path | |
|---|---|---|
| `GET` | `/api/templates` | available conversation templates |
| `GET` | `/api/config` | what this deployment is wired up to |
| `POST` | `/api/uploads` | ask for somewhere to put a recording (presigned, when storage supports it) |
| `POST` | `/api/conversations` | multipart upload through the app; starts processing |
| `POST` | `/api/conversations/complete` | register a recording already uploaded to storage |
| `GET` | `/api/conversations` | list |
| `GET` | `/api/conversations/{id}` | full record: form, transcript, metrics |
| `GET` | `/api/conversations/{id}/status` | cheap poll target while processing |
| `PATCH` | `/api/conversations/{id}/fields/{field_id}` | edit a field |
| `POST` | `/api/conversations/{id}/fields/{field_id}/revert` | restore the draft |
| `POST` | `/api/conversations/{id}/reprocess` | re-run; edited fields are kept |
| `POST` | `/api/conversations/{id}/resume` | push a stalled run forward; safe at any time |
| `POST` | `/api/webhooks/transcription` | called by the transcription service when a job finishes |
| `GET` | `/api/conversations/{id}/audio` | recording, with `Range` support |
| `GET` | `/api/conversations/{id}/audit` | append-only trail |
| `GET` | `/api/conversations/{id}/export.md` | shareable record (`?transcript=true`) |
| `GET` | `/api/conversations/{id}/export.json` | the whole record as data |
| `DELETE` | `/api/conversations/{id}` | record and recording |

Interactive docs at `/docs`.

---

## Layout

```
app/
  templates.py         the agenda and the form, as data
  models.py            Conversation, Segment, FormField, AuditEvent
  pipeline.py          the stage machine: resumable, idempotent, runner-agnostic
  storage.py           local disk and S3-compatible buckets behind one interface
  transcription/       fixture, faster-whisper and AssemblyAI behind one protocol
  analysis/
    heuristic.py       offline DP aligner, speaker roles, extractive drafter
    claude.py          structured alignment and drafting, plus output repair
    metrics.py         talk ratio, questions, planned vs actual timing
  export.py            Markdown and JSON renderings
  main.py              HTTP API and the web app
  static/              single-page review UI, no build step
scripts/seed_demo.py   loads the worked example
scripts/migrate.py     creates the schema, for deployments that need it explicit
Dockerfile             container image, for any host with a persistent volume
api/index.py           Vercel entrypoint; vercel.json routes everything to it
```

---

## Deploying

The app runs in two shapes. Same code, same tests; what changes is where state
lives and what drives the work between requests.

| | **Server** | **Serverless** |
|---|---|---|
| Host | Docker, Render, Railway, Fly, a VM | Vercel, or any FaaS |
| Database | SQLite on a volume | Postgres |
| Recordings | disk | S3-compatible bucket, uploaded direct from the browser |
| Transcription | faster-whisper, locally | AssemblyAI, by webhook |
| Background work | a thread | webhook + resume, one stage per request |
| Audio leaves your infrastructure | no | yes — to the transcription service |

### Server

```bash
docker compose up --build          # http://localhost:8000
```

The compose file mounts a named volume at `/data`; recordings and the database
survive restarts because of it. Render, Railway, Fly.io, a VM or your own
Kubernetes all take the same image — point `DATA_DIR` at the mounted volume and
set `ANTHROPIC_API_KEY`. Most managed hosts inject `$PORT`, which the
container's start command honours.

The first transcription downloads the Whisper model (a few hundred MB for
`small`) into `/data/models`, so give the volume room and expect the first run
after a fresh deploy to be slow.

### Serverless (Vercel)

A serverless function has a read-only filesystem, no state between requests, a
few-megabyte cap on request bodies, and it freezes the moment it responds. The
serverless configuration replaces each of those assumptions rather than working
around them:

```
browser ──presigned PUT──▶ bucket                    (audio never enters a function)
   │
   └─POST /api/conversations/complete──▶ hand off to AssemblyAI, return   (~1s)
                                              │
                    AssemblyAI ──webhook──▶ /api/webhooks/transcription
                                              │
                                     align + draft + store   (~60s)
```

The webhook is the one invocation guaranteed to happen after the upload
responds, so it carries the conversation the rest of the way. Nothing is held
in memory between stages, and every stage is idempotent — a duplicate webhook
is a no-op, and a conversation whose invocation died is picked back up by
`POST /api/conversations/{id}/resume`, which the browser's own status polling
calls for free.

**1. Provision the three services.**

- Postgres — [Neon](https://neon.tech), Vercel Postgres, or Supabase.
- An S3-compatible bucket — [Cloudflare R2](https://developers.cloudflare.com/r2/)
  is a good default (no egress fees). AWS S3 and Backblaze B2 work identically.
- An [AssemblyAI](https://www.assemblyai.com) API key.

**2. Let the browser upload to the bucket.** The presigned `PUT` comes from a
different origin than the bucket, so CORS has to allow it, or uploads fail with
an opaque browser error:

```json
[{
  "AllowedOrigins": ["https://your-app.vercel.app"],
  "AllowedMethods": ["PUT", "GET"],
  "AllowedHeaders": ["content-type"],
  "ExposeHeaders": ["etag"],
  "MaxAgeSeconds": 3000
}]
```

**3. Create the schema**, once, from your machine:

```bash
DATABASE_URL='postgresql://...' python scripts/migrate.py
```

**4. Set the environment variables** in the Vercel project:

```bash
JOB_RUNNER=deferred            # one stage per request; no background threads
AUTO_CREATE_TABLES=false       # migrate.py did it; do not repeat per cold start
DATABASE_URL=postgresql://...
PUBLIC_BASE_URL=https://your-app.vercel.app   # so the webhook can find you
WEBHOOK_SECRET=<a long random string>

STORAGE_BACKEND=s3
STORAGE_BUCKET=conversation-recordings
STORAGE_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com  # omit for AWS S3
STORAGE_REGION=auto
STORAGE_ACCESS_KEY_ID=...
STORAGE_SECRET_ACCESS_KEY=...

SPEECH_PROVIDER=assemblyai
ASSEMBLYAI_API_KEY=...

ANALYSIS_PROVIDER=claude
ANTHROPIC_API_KEY=...
```

`PUBLIC_BASE_URL` matters: without an address the transcription service can
call back to, the handoff fails immediately and says so, rather than leaving a
conversation stuck forever. On Vercel it falls back to `VERCEL_URL`, but that
changes per deployment — set it explicitly to your production domain.

**5. Deploy.** `vercel.json` routes everything to `api/index.py` and asks for a
300-second maximum duration, which is what the analysis stage needs headroom
for. That duration requires a Pro plan; on Hobby the cap is 60 seconds and a
long conversation may time out mid-analysis. It is not lost when that happens —
the record stays at its last completed stage and the next status poll resumes
it — but it will take a few rounds to get through.

#### What this costs you

Transcription moves off your infrastructure. The recording is fetched by
AssemblyAI from a presigned URL, which is a real change to the privacy posture
of the app: on the server deployment the audio never leaves the machine you
control. If that property is what you wanted this system for, deploy it as a
server instead — the choice is between the two tables above, not a detail.

In exchange, the hosted path gets you genuine speaker diarisation. On this path
the manager/report labels are measured rather than inferred from who asks the
questions, which makes the talk-time figures and per-speaker attribution
meaningfully more trustworthy.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

151 tests, no network and no model download. Every external service is stubbed
at its boundary: the fixture speech provider reads a sidecar transcript, the
Claude analyst runs against a fake client so the repair logic (out-of-order
boundaries, out-of-range citations, refusals) is tested directly, and the
serverless path is exercised with a stub bucket and a stub transcription
service — including the cases that only happen in production, like a duplicate
webhook and an invocation that dies partway through.

---

## Consent and data

Uploading requires confirming that both people knew the conversation was being
recorded. That is a checkbox, not a legal control — recording a workplace
conversation is regulated differently in different places, and in several
jurisdictions all-party consent is required. Decide the policy before deploying
this, not after.

Where the data goes depends on how you deploy it, and the difference is worth
being explicit about with the people being recorded:

- **Server deployment** — recordings under `DATA_DIR`, the record in SQLite, both
  on the machine you run. With `SPEECH_PROVIDER=faster_whisper` the audio never
  leaves it. With `ANALYSIS_PROVIDER=claude` the *transcript text* goes to the
  Claude API for alignment and drafting; the audio does not.
  `ANALYSIS_PROVIDER=heuristic` sends nothing anywhere.
- **Serverless deployment** — recordings sit in your bucket, and the
  transcription service fetches each one to transcribe it. The audio does leave
  your infrastructure. Check that against whatever you told the people in the
  room.

There is no authentication or per-user access control in this codebase. It
assumes a trusted network. Before real use it needs an identity layer, and a
decision about who can read whose record — the person coached should be able to
read theirs, which is the entire point.
