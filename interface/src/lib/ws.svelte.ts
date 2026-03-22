import type {
  WsMessage, ChatEntry, ActionLogEntry,
  TokenPayload, ResponsePayload, ActionPayload,
  VisionPayload, StatusPayload
} from './types.js';

const WS_URL = 'ws://localhost:8000/ws';

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

// ── Reactive state (Svelte 5 runes) ──────────────────────────────────────────
export const connected = $state({ value: false });
export const brainState = $state({ value: 'idle' as 'idle' | 'thinking' | 'acting' | 'error' });
export const model = $state({ value: '—' });
export const chat = $state<ChatEntry[]>([]);
export const actionLog = $state<ActionLogEntry[]>([]);
export const visionFrame = $state({ src: '', description: '', source: '' });

// ── WebSocket singleton ───────────────────────────────────────────────────────
let socket: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function dispatch(msg: WsMessage) {
  switch (msg.kind) {
    case 'status': {
      const p = msg.payload as StatusPayload;
      brainState.value = p.state;
      model.value = p.model;
      break;
    }
    case 'thinking': {
      brainState.value = 'thinking';
      // Ensure there's an in-progress ARIA entry
      const last = chat.at(-1);
      if (!last || last.role !== 'aria' || !last.streaming) {
        chat.push({ id: uid(), role: 'aria', text: '', streaming: true, ts: msg.ts });
      }
      break;
    }
    case 'token': {
      const p = msg.payload as TokenPayload;
      const last = chat.at(-1);
      if (last && last.role === 'aria' && last.streaming) {
        last.text += p.text;
      }
      break;
    }
    case 'response': {
      const p = msg.payload as ResponsePayload;
      const last = chat.at(-1);
      if (last && last.role === 'aria' && last.streaming) {
        last.text = p.text;
        last.streaming = false;
      } else {
        chat.push({ id: uid(), role: 'aria', text: p.text, streaming: false, ts: msg.ts });
      }
      brainState.value = 'idle';
      break;
    }
    case 'action': {
      const p = msg.payload as ActionPayload;
      brainState.value = 'acting';
      actionLog.unshift({ id: uid(), action: p.action, ts: msg.ts });
      if (actionLog.length > 100) actionLog.pop();
      break;
    }
    case 'vision': {
      const p = msg.payload as VisionPayload;
      visionFrame.src = `data:image/jpeg;base64,${p.frame_b64}`;
      visionFrame.description = p.description;
      visionFrame.source = p.source;
      break;
    }
    case 'error': {
      brainState.value = 'error';
      break;
    }
  }
}

function connect() {
  if (socket && socket.readyState < 2) return;
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    connected.value = true;
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  socket.onmessage = (evt) => {
    try { dispatch(JSON.parse(evt.data) as WsMessage); }
    catch { console.error('bad ws message', evt.data); }
  };

  socket.onclose = () => {
    connected.value = false;
    brainState.value = 'idle';
    reconnectTimer = setTimeout(connect, 3000);
  };

  socket.onerror = () => { socket?.close(); };
}

export function sendCommand(text: string) {
  if (!socket || socket.readyState !== 1) return;
  chat.push({ id: uid(), role: 'user', text, streaming: false, ts: Date.now() });
  socket.send(JSON.stringify({ kind: 'command', payload: { text }, ts: Date.now() }));
}

export function initWs() { connect(); }

