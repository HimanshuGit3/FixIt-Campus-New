/**
 * Download the CLIP model into server/.cache/models at build time, so a fresh container
 * (Render, Docker) starts with the model on disk instead of fetching ~90 MB on every cold start.
 *   npm run prefetch-model -w server
 * Needs no database or secrets; uses the same model id and cache folder as services/clip.js.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  env as hf,
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
} from '@huggingface/transformers';

const serverRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modelId = process.env.AI_MODEL_ID || 'Xenova/clip-vit-base-patch32';
hf.cacheDir = path.join(serverRoot, '.cache', 'models');

const started = Date.now();
const opts = { dtype: 'q8' };
const loaded = await Promise.all([
  AutoTokenizer.from_pretrained(modelId),
  CLIPTextModelWithProjection.from_pretrained(modelId, opts),
  CLIPVisionModelWithProjection.from_pretrained(modelId, opts),
  AutoProcessor.from_pretrained(modelId),
]);
await Promise.all(loaded.map((m) => m.dispose?.()));
console.log(`Model ${modelId} cached in ${hf.cacheDir} (${Date.now() - started} ms)`);
