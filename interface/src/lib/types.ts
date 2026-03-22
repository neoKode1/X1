// Message types flowing over the WebSocket

export type ActionType = 'MOVE' | 'SPEAK' | 'LOOK' | 'THINK' | 'WAIT' | 'QUERY' | 'SKILL';

export interface BrainAction {
  type: ActionType;
  params: Record<string, unknown>;
  result?: string;
}

export interface WsMessage {
  kind: 'token' | 'response' | 'action' | 'vision' | 'error' | 'status' | 'thinking';
  payload: unknown;
  ts: number; // unix ms
}

export interface TokenPayload {
  text: string;
}

export interface ResponsePayload {
  text: string;
  actions: BrainAction[];
}

export interface ActionPayload {
  action: BrainAction;
}

export interface VisionPayload {
  frame_b64: string;   // JPEG base64
  description: string;
  source: 'webcam' | 'screen';
}

export interface StatusPayload {
  state: 'idle' | 'thinking' | 'acting' | 'error';
  model: string;
  uptime_s: number;
}

export interface ErrorPayload {
  message: string;
}

// Chat entry in the UI log
export interface ChatEntry {
  id: string;
  role: 'user' | 'aria';
  text: string;
  streaming: boolean;
  ts: number;
}

// Action log entry
export interface ActionLogEntry {
  id: string;
  action: BrainAction;
  ts: number;
}

