<script lang="ts">
  import { connected, brainState, model, chat, actionLog, visionFrame, sendCommand } from '$lib/ws.svelte.js';

  let input = $state('');

  function submit() {
    const text = input.trim();
    if (!text || !connected.value) return;
    sendCommand(text);
    input = '';
  }

  function handleKey(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  }

  const stateColors: Record<string, string> = {
    idle: '#4ade80', thinking: '#facc15', acting: '#60a5fa', error: '#f87171'
  };

  function actionColor(type: string) {
    const map: Record<string, string> = {
      MOVE: '#60a5fa', SPEAK: '#4ade80', LOOK: '#a78bfa',
      THINK: '#facc15', WAIT: '#94a3b8', QUERY: '#fb923c', SKILL: '#e879f9'
    };
    return map[type] ?? '#94a3b8';
  }

  function tsLabel(ms: number) {
    return new Date(ms).toLocaleTimeString();
  }
</script>

<!-- Status bar -->
<header class="status-bar">
  <span class="logo">X1 <span class="dim">// ARIA</span></span>
  <span class="dot" style="background:{stateColors[brainState.value]}"></span>
  <span class="state-label">{brainState.value.toUpperCase()}</span>
  <span class="dim sep">|</span>
  <span class="dim">model: {model.value}</span>
  <span class="dim sep">|</span>
  <span class="conn" class:online={connected.value}>
    {connected.value ? '● connected' : '○ connecting…'}
  </span>
</header>

<!-- Main layout -->
<main class="grid">

  <!-- ── Chat panel ── -->
  <section class="panel chat-panel">
    <h2 class="panel-title">Chat</h2>
    <div class="chat-log" id="chat-log">
      {#each chat as entry (entry.id)}
        <div class="chat-entry {entry.role}">
          <span class="role-tag">{entry.role === 'user' ? 'YOU' : 'ARIA'}</span>
          <span class="ts">{tsLabel(entry.ts)}</span>
          <p class="msg">{entry.text}{entry.streaming ? '▌' : ''}</p>
        </div>
      {/each}
      {#if chat.length === 0}
        <p class="placeholder">Send a command to wake ARIA…</p>
      {/if}
    </div>
    <div class="chat-input-row">
      <textarea
        class="cmd-input"
        rows="2"
        placeholder={connected.value ? 'Type a command…' : 'Waiting for server…'}
        disabled={!connected.value}
        bind:value={input}
        onkeydown={handleKey}
      ></textarea>
      <button class="send-btn" onclick={submit} disabled={!connected.value || !input.trim()}>
        SEND
      </button>
    </div>
  </section>

  <!-- ── Right column ── -->
  <div class="right-col">

    <!-- Vision -->
    <section class="panel vision-panel">
      <h2 class="panel-title">
        Vision
        <span class="dim small">{visionFrame.source || '—'}</span>
      </h2>
      {#if visionFrame.src}
        <img class="vision-img" src={visionFrame.src} alt="vision frame" />
        <p class="vision-desc">{visionFrame.description}</p>
      {:else}
        <div class="vision-placeholder">No frame yet — trigger a LOOK action</div>
      {/if}
    </section>

    <!-- Action log -->
    <section class="panel action-panel">
      <h2 class="panel-title">Action Log</h2>
      <div class="action-log">
        {#each actionLog as entry (entry.id)}
          <div class="action-row">
            <span class="action-badge" style="background:{actionColor(entry.action.type)}">
              {entry.action.type}
            </span>
            <span class="action-result">{entry.action.result ?? JSON.stringify(entry.action.params)}</span>
            <span class="ts">{tsLabel(entry.ts)}</span>
          </div>
        {/each}
        {#if actionLog.length === 0}
          <p class="placeholder">No actions yet</p>
        {/if}
      </div>
    </section>

  </div>
</main>

<style>
  :global(*, *::before, *::after) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) { background: #0d0f14; color: #e2e8f0; font-family: 'Courier New', monospace; font-size: 14px; height: 100dvh; overflow: hidden; }

  .status-bar { display: flex; align-items: center; gap: 10px; padding: 8px 16px; background: #111318; border-bottom: 1px solid #1e2330; font-size: 13px; }
  .logo { font-weight: 700; font-size: 15px; color: #60a5fa; }
  .dim { color: #475569; }
  .sep { padding: 0 4px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .state-label { font-weight: 600; letter-spacing: 1px; font-size: 11px; }
  .conn { color: #f87171; }
  .conn.online { color: #4ade80; }

  main.grid { display: grid; grid-template-columns: 1fr 380px; gap: 12px; padding: 12px; height: calc(100dvh - 41px); overflow: hidden; }

  .panel { background: #111318; border: 1px solid #1e2330; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
  .panel-title { padding: 10px 14px; border-bottom: 1px solid #1e2330; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #60a5fa; display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
  .small { font-size: 10px; color: #475569; }

  .chat-panel { height: 100%; }
  .chat-log { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
  .chat-entry { display: flex; flex-direction: column; gap: 3px; }
  .chat-entry.user .msg { color: #cbd5e1; }
  .chat-entry.aria .msg { color: #4ade80; }
  .role-tag { font-size: 10px; letter-spacing: 1px; font-weight: 700; color: #475569; }
  .chat-entry.user .role-tag { color: #60a5fa; }
  .ts { font-size: 10px; color: #334155; }
  .msg { white-space: pre-wrap; word-break: break-word; line-height: 1.5; }
  .placeholder { color: #334155; font-style: italic; padding: 8px; }

  .chat-input-row { display: flex; gap: 8px; padding: 10px; border-top: 1px solid #1e2330; flex-shrink: 0; }
  .cmd-input { flex: 1; background: #0d0f14; border: 1px solid #1e2330; border-radius: 6px; color: #e2e8f0; padding: 8px 10px; font-family: inherit; font-size: 13px; resize: none; outline: none; }
  .cmd-input:focus { border-color: #3b82f6; }
  .cmd-input:disabled { opacity: 0.4; cursor: not-allowed; }
  .send-btn { background: #1d4ed8; color: #fff; border: none; border-radius: 6px; padding: 0 18px; font-family: inherit; font-size: 12px; font-weight: 700; letter-spacing: 1px; cursor: pointer; }
  .send-btn:hover:not(:disabled) { background: #2563eb; }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .right-col { display: flex; flex-direction: column; gap: 12px; overflow: hidden; }
  .vision-panel { flex: 0 0 auto; }
  .vision-img { width: 100%; max-height: 220px; object-fit: contain; background: #0d0f14; }
  .vision-desc { padding: 8px 12px; font-size: 12px; color: #94a3b8; border-top: 1px solid #1e2330; }
  .vision-placeholder { padding: 40px 12px; text-align: center; color: #334155; font-style: italic; }

  .action-panel { flex: 1; min-height: 0; }
  .action-log { flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 6px; }
  .action-row { display: flex; align-items: flex-start; gap: 8px; flex-wrap: wrap; }
  .action-badge { font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; color: #0d0f14; flex-shrink: 0; }
  .action-result { font-size: 12px; color: #94a3b8; flex: 1; min-width: 0; word-break: break-all; }
</style>
