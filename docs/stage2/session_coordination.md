# Stage 2 — multi-session eval coordination (2026-04-24)

Design note for the shared-file coordination protocol that lets two (or more)
Claude Code sessions run Stage-2 evaluations against the same model without
duplicating work or stepping on each other's GPUs. Operational file is
[`/mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17/eval_coordination.md`](/mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17/eval_coordination.md);
this document records the design rationale and onboarding instructions.

---

## 1. Problem

Two Claude Code sessions were independently running evals against the same
Stage-2 training checkpoints. Symptoms we hit within ~30 minutes:

- **Overlapping launches.** Session B spawned `eval_source_emotion` on
  ckpts 3k/4k/6k/…/11k, not knowing session A had started the *same* sweep
  5 minutes earlier. GPU memory ran out mid-load.
- **Invisible incidents.** Session A's Clotho ckpt-1000 OOM'd twice on
  GPU 1 and session B had no way to learn about it.
- **Lost user context.** User gave instructions in one session's terminal
  ("source dataset only", "dense trajectory", "report comparison"); the
  other session was unaware.

Root cause: no shared state between sessions. Each session sees only its
own conversation. The file system is the only common surface.

## 2. Mental model

User framed it as **a group chat for a team project**:

- **User = 팀장 (team lead)** — assigns roles, posts decisions, reviews.
- **Each Claude session = 팀원 (team member)** — claims work, reports
  progress, logs blockers.
- **Coordination file = 단톡방 (group chat)** — the single place everyone
  reads/writes to see what's going on.

Assignments can be either **pushed** (user writes into their area and a
session picks it up) or **pulled** (session reads the queue and claims).

## 3. Design

One markdown file, six sections, top-to-bottom reading order:

```
📖 Overview / 목적
🛠 How to use
👤 User area           ← only the user writes here
📋 TODO                ← sessions translate User area → executable items
🏃 Running now         ← active claims (session, PID, GPU, ETA)
✅ Completed           ← results summary (most recent on top)
⚠ Notes & incidents    ← blockers, retries, lessons
```

### Why these six

- **Overview** — a session re-loaded fresh should understand the file's
  role in 30 seconds without prior context.
- **How to use** — the exact append/remove/move protocol. Human-readable
  so sessions don't need a library, just `cat` + `Edit`.
- **User area vs TODO** — separating them keeps the user's voice ("I want
  X") distinct from the session's interpretation ("run `eval_foo
  ckpts=…`"). Makes it obvious when a user request is still unprocessed.
- **Running vs Completed vs Notes** — 3-state lifecycle. "Running" is the
  anti-duplicate gate; "Completed" is the digest; "Notes" is the incident
  log.

### Why markdown, not JSON

- Two sessions + one human all read and edit this. JSON would need a
  formal schema + machine validator; markdown survives free-form edits.
- No parsing machinery needed. `cat` shows the current state, `Edit`
  appends a line. Sessions read the whole file each time.

### Concurrency

- Writes are "append-only in spirit" — sessions rarely delete each
  other's lines.
- Actual concurrent writes are rare (minutes between eval launches). When
  they happen, **last-write-wins**; sessions re-append missing claims on
  the next check.
- Same `(eval_script, ckpt)` claimed by two sessions simultaneously →
  later PID yields and removes its claim. Detectable by comparing start
  times in the Running section.

### Session identity

- Sessions self-assign `A`, `B`, `C` by appearance order in the file.
- User can override by telling a session its ID via prompt (see §5 below).

## 4. Protocol (normative)

**Before launching any eval:**
1. `cat` the coordination file. Read 👤 User area first; then 📋 TODO.
2. If User area has a new instruction not yet in TODO: translate it to
   executable units and add them to TODO.
3. Check 🏃 Running now for overlap on the `(eval_script, ckpt)` pair
   you're about to run. If overlap, pick a different item.
4. Add a claim line to 🏃 Running now:
   ```
   [SessionID | PID | GPU | started HH:MM | ETA +Xm] eval_name ckpts=... — status
   ```
5. Launch the python command.

**While running:** update the `status` field on the claim line if state
changes materially (e.g., `5/12 done`, `OOM retry`).

**On success:**
- Remove the claim line from 🏃 Running now.
- Prepend a line to ✅ Completed with representative metrics:
  ```
  [SessionID | finished HH:MM] eval_name ckpt=N → metric=VALUE (N_rows rows)
  ```

**On failure / abort:**
- Remove the claim line from 🏃 Running now.
- Add a bullet to ⚠ Notes & incidents with root cause and mitigation.

**User area etiquette:** only the user writes to it. Sessions read and
translate. Newest at top.

## 5. Onboarding a session

When the user wants to bring a new session into the protocol, the
following prompt is sufficient:

> ```
> /mnt/tmp/results/Qwen3.5AE-Stage2-lora-asr14-emo34-env35-txt17/eval_coordination.md
> 파일이 세션 간 단톡방이다. 앞으로 eval 시작 전 이 파일을 읽고,
> ## Running now 섹션에 내 claim을 append 한 뒤에만 런칭. 끝나면
> ## Completed로 옮기고 결과 요약 남긴다. 너는 session {A|B|C}.
> ```

Effect:
- Session reads the overview, understands the state machine.
- Session knows its ID (to prefix claims).
- From that point on, all eval launches flow through the coordination
  file — no more overlaps.

## 6. Governance

- The coordination file is **not** a rolling log — it is the current
  state. Old claims that completed go to ✅ Completed (compacted to key
  metrics only). Old incidents go to ⚠ Notes & incidents but stay; they
  are cheap history.
- If the file grows > a few hundred lines, the team lead can "rotate"
  it: move ✅ Completed rows (not the Notes) into a dated archive and
  reset the live section.
- The **authoritative** numeric record stays in
  [`eval_harness.md`](eval_harness.md) and per-eval
  `summary.json`. The coordination file is a progress board, not a
  report.
- When a session wraps up for the day, it leaves any still-running
  claims in 🏃 Running now with a note; the next session that wakes up
  takes over monitoring those PIDs.

## 7. Limitations / future work

- **File-lock on writes** is not implemented; sessions use `Read → Edit`
  which is not atomic. Acceptable given how rare concurrent writes are
  in practice; if a collision is observed, the losing write is re-applied
  on next check.
- **No per-GPU ownership record.** Two sessions can independently decide
  to target GPU 5 if it looks free *at read time*. The incident log
  recommends `nvidia-smi --query-compute-apps` as a pre-launch check —
  this should be folded into a shared helper script eventually.
- **Session IDs are purely conventional.** If a session restarts (fresh
  context), it re-derives its ID by reading existing claims. This is
  fine for 2 sessions, gets ambiguous at ≥ 4.
- **User area is not version-controlled within the file.** If the user
  revises an earlier instruction, the correct move is "newest instruction
  wins"; sessions might still be acting on the old one until they re-read.
  In practice, user interactions are rare enough that the next
  eval-launch cycle catches up.

## 8. Related docs

- [`eval_harness.md`](eval_harness.md) — what we measure,
  scripts, numeric results.
- [`eval_plan.md`](eval_plan.md) — benchmark matrix and
  success criteria.
- [`design.md`](design.md) — training setup.
