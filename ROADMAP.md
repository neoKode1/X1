# ARIA Project Roadmap
> Living document. Goals evolve — keep updating this.
> Last updated: 2026-03-22

---

## What We're Building

Two intertwined projects:

| Project | Location | What it is |
|---|---|---|
| **robot-brain** | `clawd/robot-brain/` | Physical robot AI — Raspberry Pi 5, motors, camera, real GPIO |
| **prototype-x1** | `~/Desktop/prototypes/prototype-x1/` | Local AI brain — runs on your Mac, sees your screen, builds itself |

---

## Phase 1 — Brain Skeleton ✅ DONE
- [x] Modular Python architecture (config, LLM adapter, action schema, hardware abstraction)
- [x] Dual LLM support: Anthropic Claude + OpenAI GPT-4o
- [x] Structured action output: MOVE / SPEAK / LOOK / WAIT / THINK / QUERY
- [x] Mock hardware layer (runs on any machine)
- [x] 18 tests passing

## Phase 2 — Vision ✅ DONE
- [x] OpenCV webcam integration (auto-detect, JPEG capture, base64 encode)
- [x] Vision adapters: GPT-4o vision + Claude vision
- [x] LOOK action triggers real camera capture
- [x] Vision context injected into brain's reasoning loop
- [x] `look` / `snap` REPL commands
- [x] 28 tests passing

---

## Phase 3 — prototype-x1: Local Brain (72-hour sprint)

> **Goal:** A local AI brain that knows how to DO things, not just talk.
> Think: Augment's capabilities + more, running entirely on your machine.

### 3.1 Core Brain (no cloud dependency)
- [ ] Local model support via Ollama (llama3, mistral, or similar)
- [ ] Skill registry — brain declares what it can do, executes tools directly
- [ ] Fallback to Claude/GPT-4o if local model unavailable

### 3.2 Vision & Perception
- [ ] Webcam live feed (reuse Phase 2 camera.py)
- [ ] **Screen grab** — capture full desktop or active window as context
- [ ] Describe what it sees + what's on screen together

### 3.3 Self-Understanding — Boot Sequence
- [ ] On first run: read USER.md, SOUL.md, environment, directory structure
- [ ] Scan who the user is (name, projects, recent files, git repos)
- [ ] Build an internal model of its own capabilities and limitations
- [ ] Store initial observations in memory

### 3.4 Memory & Context (smart indexing)
- [ ] Episodic memory: timestamped logs of what happened
- [ ] Semantic memory: vector-embedded facts, retrievable by similarity
- [ ] Working memory: rolling conversation window
- [ ] Periodic consolidation: daily logs → long-term facts (like MEMORY.md but automated)

### 3.5 Skills Library (built-in capabilities)
- [ ] File system: read, write, search, summarize files
- [ ] Code: write, run, test Python scripts
- [ ] Screen: grab, annotate, describe
- [ ] Web search: via Breach SDK *(package TBD — user to provide)*
- [ ] Shell: run safe commands, capture output
- [ ] Git: status, diff, commit, push
- [ ] Calendar / email awareness (read-only first)

### 3.6 SDK Integrations
- [ ] **Breach** — web search / intelligence layer *(awaiting SDK link from user)*
- [ ] **Twelve Monkeys** — *(awaiting SDK link from user)*

### 3.7 Self-Building
- [ ] Brain can write new skill modules and load them at runtime
- [ ] Skill hot-reload without restart
- [ ] Brain proposes its own improvements, user approves

---

## Phase 4 — Physical Robot Integration (robot-brain)
- [ ] Wire motors to Pi 5 via L298N H-bridge
- [ ] Real hardware mode: `HARDWARE_PLATFORM=raspberry_pi`
- [ ] Servo for camera pan/tilt (LOOK direction control)
- [ ] Salvaged speaker integration (espeak TTS over the power board amp)
- [ ] Real-time obstacle avoidance using vision feedback loop

## Phase 5 — Self-Assembly
- [ ] Brain generates its own GPIO wiring code
- [ ] Robot navigates by sight, maps its space
- [ ] Brain writes new skills for new hardware it encounters
- [ ] Git-push its own code changes after human review

---

## Ultimate Outcomes

> *Feed these in as they become clear.*

1. **A robot that builds and programs itself** — given salvaged parts, it figures out how to use them
2. **A local AI brain richer than any cloud assistant** — all skills local, no internet required for core function
3. **Persistent identity** — ARIA knows who she is, who you are, and grows over time
4. **Zero-shot hardware adaptation** — plug in a new sensor; brain writes its own driver

---

## SDK / Dependency Notes

| Tool | Status | Notes |
|---|---|---|
| `anthropic` | ✅ installed | Default LLM provider |
| `openai` | ✅ installed | Vision + fallback |
| `opencv-python` | ✅ installed | Camera capture |
| `Pillow` | ✅ installed | Image processing |
| `Breach SDK` | ⏳ PENDING | User to provide pip package or link |
| `Twelve Monkeys SDK` | ⏳ PENDING | User to provide pip package or link |
| `ollama` (Python client) | 📋 planned | Local model support in prototype-x1 |
| `chromadb` or `faiss` | 📋 planned | Vector memory for semantic search |

