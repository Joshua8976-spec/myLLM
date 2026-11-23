import { type Message, type ModelMode, MODES, type AppSettings } from '../types';

export interface StreamCallbacks {
  onChunk: (chunkText: string) => void;
  onDone: (fullText: string) => void;
  onError: (error: Error) => void;
}

export async function checkBackendHealth(apiUrl: string): Promise<{ ok: boolean; model?: string; latencyMs: number; error?: string }> {
  const start = performance.now();
  try {
    // Strip trailing /v1 if present to ping /health
    const base = apiUrl.replace(/\/v1\/?$/, '');
    const res = await fetch(`${base}/health`, { method: 'GET', signal: AbortSignal.timeout(8000) });
    const latency = Math.round(performance.now() - start);
    if (!res.ok) {
      return { ok: false, latencyMs: latency, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { ok: true, model: data.model || 'studyverse-qwen-2.5-3b', latencyMs: latency };
  } catch (err: unknown) {
    const latency = Math.round(performance.now() - start);
    return { ok: false, latencyMs: latency, error: err instanceof Error ? err.message : String(err) };
  }
}

export async function streamChatCompletion(
  messages: Message[],
  mode: ModelMode,
  settings: AppSettings,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const systemPrompt = MODES[mode].systemPrompt;

  // Format messages for OpenAI compatibility
  const apiMessages = [
    { role: 'system', content: systemPrompt },
    ...messages.map((m) => ({
      role: m.role,
      content: m.content,
    })),
  ];

  const endpoint = `${settings.apiUrl.replace(/\/+$/, '')}/chat/completions`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: settings.model,
        messages: apiMessages,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
        stream: true,
      }),
      signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Inference error (HTTP ${response.status}): ${errText || response.statusText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let accumulatedText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;

        const dataStr = trimmed.replace(/^data:\s*/, '');
        if (dataStr === '[DONE]') {
          callbacks.onDone(accumulatedText);
          return;
        }

        try {
          const parsed = JSON.parse(dataStr);
          const deltaContent = parsed.choices?.[0]?.delta?.content;
          if (deltaContent) {
            accumulatedText += deltaContent;
            callbacks.onChunk(deltaContent);
          }
        } catch {
          // Incomplete chunk, keep going
        }
      }
    }

    callbacks.onDone(accumulatedText);
  } catch (err: unknown) {
    if (signal?.aborted) {
      callbacks.onDone('');
      return;
    }
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
  }
}
