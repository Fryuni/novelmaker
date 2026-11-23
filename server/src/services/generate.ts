// Generierung: Parameter bestimmen, streamen, protokollieren.
import type { ServerResponse } from 'node:http';
import {
  WORDS_OPTION_NAMES,
  LENGTH_OPTION_NAMES,
  estimateMessagesTokens,
  estimateTokens,
  t,
  wordsToTokens,
  type CostEstimate,
  type GenerateRequest,
  type ModelRef,
  type Provider,
  type RenderedMessage,
} from '@novelmaker/shared';
import type { DB } from '../db';
import { collectionParams, createLog, finishLog, getProvider, findModel } from '../repo/llm';
import { getPrompt } from '../repo/prompts';
import { getAppSettings } from '../repo/settings';
import { streamChat, type StreamEvent } from '../llm/client';
import { inputNumber, type Prepared } from './promptdata';

export interface GenerationParams {
  /** geht so an den Anbieter */
  params: Record<string, unknown>;
  /** Obergrenze in Tokens nur für den sichtbaren Text; runGeneration setzt sie beim Streamen durch. null = keine */
  outputLimit: number | null;
}

/**
 * Request-Parameter bestimmen. Gibt weder die Modellsammlung noch der Aufruf max_tokens vor, ergibt sich aus
 * „Words“ (bzw. `lengthNames`, z. B. für Zusammenfassungen) eine Grenze für den sichtbaren Text. Die setzt
 * runGeneration selbst durch; an den Anbieter geht sie plus Reserve, weil viele Anbieter das Nachdenken von
 * Reasoning-Modellen zu max_tokens zählen – sonst bricht die Antwort ab, bevor sichtbarer Text kommt.
 */
export function buildParams(
  db: DB,
  model: ModelRef,
  prep: Prepared | null,
  override?: Record<string, unknown>,
  opts?: { lengthNames?: string[] },
): GenerationParams {
  const params: Record<string, unknown> = { ...collectionParams(db, model.providerId, model.modelId), ...(override ?? {}) };
  const settings = getAppSettings(db);
  let outputLimit: number | null = null;
  if (prep && params.max_tokens == null) {
    if (settings.deriveMaxTokens) {
      const words = inputNumber(prep.prompt, prep.components, prep.inputs, WORDS_OPTION_NAMES);
      if (words && words > 0) outputLimit = Math.round(words * settings.maxTokensFactor + settings.maxTokensBuffer);
    }
    if (outputLimit == null && opts?.lengthNames) {
      const len = inputNumber(prep.prompt, prep.components, prep.inputs, opts.lengthNames);
      if (len && len > 0) outputLimit = Math.round(len * 2 + 300);
    }
    if (outputLimit != null) params.max_tokens = providerMaxTokens(db, model, prep, outputLimit, settings.reasoningReserve);
  }
  for (const [k, v] of Object.entries(params)) if (v === '' || v === null) delete params[k];
  return { params, outputLimit };
}

/** Sichtbare Grenze + Reserve fürs Nachdenken, soweit Modell (max. Antwortlänge) und Kontextfenster es zulassen. */
function providerMaxTokens(db: DB, model: ModelRef, prep: Prepared, outputLimit: number, reserve: number): number {
  let max = outputLimit + Math.max(0, Number(reserve) || 0);
  const m = findModel(db, model.providerId, model.modelId);
  if (m?.maxOutput) max = Math.min(max, m.maxOutput);
  // Manche Anbieter lehnen ab, wenn Prompt + max_tokens nicht ins Kontextfenster passen (Prompt nur geschätzt, daher Abstand)
  if (m?.contextLength) max = Math.min(max, Math.floor(m.contextLength * 0.95) - estimateMessagesTokens(prep.result.messages));
  return Math.max(outputLimit, max);
}

export function estimateCost(db: DB, model: ModelRef, promptTokens: number, completionTokens: number): number | null {
  const m = findModel(db, model.providerId, model.modelId);
  if (!m?.pricing) return null;
  return promptTokens * m.pricing.prompt + completionTokens * m.pricing.completion;
}

const LOCAL_KINDS = new Set(['ollama', 'lmstudio', 'koboldcpp']);

/** Lokaler Server ohne Kosten? (Ollama & Co. oder ein eigener Server auf diesem Rechner) */
export function isLocalProvider(p: Provider): boolean {
  if (LOCAL_KINDS.has(p.kind)) return true;
  if (p.kind !== 'custom') return false;
  try {
    const host = new URL(p.baseUrl).hostname;
    return host === 'localhost' || host === '127.0.0.1' || host === '[::1]' || host === '::1';
  } catch {
    return false;
  }
}

/** Antwortlänge relativ zur Markierung bei den mitgelieferten Bearbeiten-Prompts */
const REWRITE_FACTOR: Record<string, number> = { shorten: 0.6, expand: 1.6 };

/** Tokens und Kosten schätzen, bevor generiert wird. */
export function estimateGeneration(db: DB, req: GenerateRequest, prep: Prepared, params: Record<string, unknown>): CostEstimate {
  const promptTokens = estimateMessagesTokens(prep.result.messages);
  const rawMax = Number(params.max_tokens);
  const maxCompletionTokens = Number.isFinite(rawMax) && rawMax > 0 ? Math.round(rawMax) : null;

  let completionTokens: number | null = null;
  let completionBasis: CostEstimate['completionBasis'] = null;
  let words: number | null = null;
  const w = inputNumber(prep.prompt, prep.components, prep.inputs, WORDS_OPTION_NAMES);
  if (w && w > 0) {
    words = w;
    completionTokens = wordsToTokens(w);
    completionBasis = 'words';
  } else if (req.kind === 'summary') {
    words = inputNumber(prep.prompt, prep.components, prep.inputs, LENGTH_OPTION_NAMES) ?? 200;
    completionTokens = wordsToTokens(words);
    completionBasis = 'length';
  } else if (req.kind === 'rewrite' && req.selection?.trim()) {
    let factor = 1;
    if (prep.prompt.id) {
      try {
        factor = REWRITE_FACTOR[getPrompt(db, prep.prompt.id).builtinKey ?? ''] ?? 1;
      } catch {
        /* Prompt nicht gespeichert */
      }
    }
    completionTokens = Math.round(estimateTokens(req.selection) * factor);
    completionBasis = 'selection';
  } else if (maxCompletionTokens) {
    completionTokens = maxCompletionTokens;
    completionBasis = 'max_tokens';
  }
  if (completionTokens != null && maxCompletionTokens) completionTokens = Math.min(completionTokens, maxCompletionTokens);

  let pricing: CostEstimate['pricing'] = null;
  let local = false;
  let contextLength: number | null = null;
  if (req.model?.providerId) {
    try {
      const provider = getProvider(db, req.model.providerId);
      const m = provider.models.find((x) => x.id === req.model!.modelId);
      contextLength = m?.contextLength ?? null;
      // OpenRouter meldet für Router-Modelle (z. B. openrouter/auto) negative Preise = unbekannt
      if (m?.pricing && m.pricing.prompt >= 0 && m.pricing.completion >= 0) pricing = m.pricing;
      else if (isLocalProvider(provider)) {
        local = true;
        pricing = { prompt: 0, completion: 0 };
      }
    } catch {
      /* Provider gelöscht */
    }
  }
  const promptCost = pricing ? promptTokens * pricing.prompt : null;
  return {
    promptTokens,
    completionTokens,
    completionBasis,
    words,
    maxCompletionTokens,
    pricing,
    local,
    promptCost,
    cost: pricing && completionTokens != null ? promptCost! + completionTokens * pricing.completion : null,
    maxCost: pricing && maxCompletionTokens ? promptCost! + maxCompletionTokens * pricing.completion : null,
    contextLength,
  };
}

export interface RunOptions {
  db: DB;
  provider: Provider;
  modelId: string;
  messages: RenderedMessage[];
  params: Record<string, unknown>;
  /** Grenze für den sichtbaren Text (aus buildParams); das Nachdenken zählt nicht mit */
  outputLimit?: number | null;
  log: { novelId: string | null; promptId: string | null; promptName: string; kind: string };
  signal: AbortSignal;
  onEvent?: (ev: StreamEvent) => void;
}

export interface RunResult {
  text: string;
  /** Nachdenken des Modells (nur zur Anzeige) */
  reasoning: string;
  status: 'ok' | 'error' | 'aborted';
  error: string | null;
  logId: string;
  promptTokens: number | null;
  completionTokens: number | null;
  cost: number | null;
  finish: string | null;
  durationMs: number;
}

export async function runGeneration(o: RunOptions): Promise<RunResult> {
  const started = Date.now();
  const logId = createLog(o.db, { ...o.log, providerId: o.provider.id, modelId: o.modelId, messages: o.messages, params: o.params });
  let text = '';
  let reasoning = '';
  let promptTokens: number | null = null;
  let completionTokens: number | null = null;
  let cost: number | null = null;
  let finish: string | null = null;
  let status: RunResult['status'] = 'ok';
  let error: string | null = null;
  try {
    for await (const ev of streamChat({ provider: o.provider, model: o.modelId, messages: o.messages, params: o.params, signal: o.signal })) {
      if (ev.type === 'delta') text += ev.text;
      else if (ev.type === 'reasoning') reasoning += ev.text;
      else if (ev.type === 'usage') {
        promptTokens = ev.promptTokens ?? promptTokens;
        completionTokens = ev.completionTokens ?? completionTokens;
        cost = ev.cost ?? cost;
      } else if (ev.type === 'finish') finish = ev.reason;
      o.onEvent?.(ev);
      if (ev.type === 'delta' && o.outputLimit && estimateTokens(text) >= o.outputLimit) {
        // Grenze für den sichtbaren Text erreicht: abbrechen (schließt die Verbindung zum Anbieter)
        finish = 'length';
        break;
      }
    }
  } catch (e) {
    if (o.signal.aborted || (e as Error).name === 'AbortError') status = 'aborted';
    else {
      status = 'error';
      error = (e as Error).message;
    }
  }
  // Der Anbieter hat bei max_tokens abgebrochen, bevor der sichtbare Text seine Grenze erreicht hat:
  // Das Nachdenken hat die Reserve aufgebraucht.
  if (finish === 'length' && o.outputLimit && estimateTokens(text) < o.outputLimit) finish = 'length-reasoning';
  const pt = promptTokens ?? estimateMessagesTokens(o.messages);
  const ct = completionTokens ?? estimateTokens(text) + estimateTokens(reasoning);
  if (cost == null) cost = estimateCost(o.db, { providerId: o.provider.id, modelId: o.modelId }, pt, ct);
  const durationMs = Date.now() - started;
  finishLog(o.db, logId, { response: text, reasoning, promptTokens: pt, completionTokens: ct, cost, durationMs, status, error });
  return { text, reasoning, status, error, logId, promptTokens: pt, completionTokens: ct, cost, finish, durationMs };
}

/** Server-Sent Events an den Browser. */
export class SseWriter {
  private closed = false;
  readonly abort = new AbortController();

  constructor(private res: ServerResponse) {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    res.on('close', () => {
      if (!this.closed) this.abort.abort();
      this.closed = true;
    });
  }

  send(event: string, data: unknown) {
    if (this.closed) return;
    this.res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }

  end() {
    if (this.closed) return;
    this.closed = true;
    this.res.end();
  }
}

export function providerFor(db: DB, model: ModelRef | undefined | null): Provider {
  if (!model?.providerId || !model.modelId) throw Object.assign(new Error(t('No model selected.')), { statusCode: 400 });
  return getProvider(db, model.providerId);
}
