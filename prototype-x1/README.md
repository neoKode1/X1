# prototype-x1 - Local AI Brain

A local AI brain that knows how to DO things.
Sees your screen. Sees your webcam. Builds itself.

## What this is

Not a chatbot. Not a wrapper. A brain that:
- Has skills: file system, code execution, screen grab, web intelligence
- Sees: webcam + desktop screen capture as live context
- Remembers: episodic + semantic memory, auto-indexed
- Knows you: boots by reading your environment, projects, identity
- Builds itself: writes new skill modules at runtime, hot-reloads them
- Runs locally via Ollama; falls back to Claude/GPT-4o if needed

## Quick Start

  cd ~/Desktop/prototypes/prototype-x1
  pip install -r requirements.txt
  cp .env.example .env
  python main.py

## Structure

  brain/core.py       reasoning loop + LLM dispatch
  brain/skills.py     skill registry + hot-reload
  brain/memory.py     episodic + semantic + working memory
  skills/             built-in skill modules
  vision/             webcam + screen capture
  memory/             auto-created runtime logs + vector store
  sdks/               Breach + Twelve Monkeys (PENDING)

## SDK Status

  Breach (web intelligence)  PENDING - user to provide pip package
  Twelve Monkeys             PENDING - user to provide pip package
