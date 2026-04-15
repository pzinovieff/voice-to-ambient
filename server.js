import express from 'express';
import cors from 'cors';
import multer from 'multer';
import Replicate from 'replicate';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ─── Configuration ───

const PORT = process.env.PORT || 3000;

const MUSICGEN_PROMPT =
  'cinematic ambient soundscape, atmospheric piano with long reverb, ethereal vocal pads, slow evolving textures, delicate harp arpeggios, emotional and contemplative, film score underscore, wide stereo, Olafur Arnalds and Nils Frahm style, 60 BPM, minor key';

const CONTINUATION_PROMPTS = [
  MUSICGEN_PROMPT,
  'cinematic ambient building intensity, layered strings and piano, swelling pads, emotional crescendo, orchestral undertones, wide reverb, contemplative',
  'cinematic ambient peak emotion, full orchestral pads, resonant piano chords, shimmering harp, expansive and yearning, cathedral reverb',
  'cinematic ambient gentle resolution, fading piano, dissolving textures, distant reverb tails, peaceful and reflective, sparse and delicate',
];

// ─── Express setup ───

const app = express();
app.use(cors());
app.use(express.static(join(__dirname, 'public')));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

let replicate;
if (process.env.REPLICATE_API_TOKEN) {
  replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
} else {
  console.warn('WARNING: REPLICATE_API_TOKEN not set. Music generation will fail.');
  console.warn('Create a .env file with: REPLICATE_API_TOKEN=your_key_here');
}

// ─── Helper: call MusicGen ───

async function callMusicGen(audioDataUri, prompt, duration = 30) {
  if (!replicate) throw new Error('REPLICATE_API_TOKEN not configured');

  const output = await replicate.run(
    'meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eeddar',
    {
      input: {
        model_version: 'melody',
        prompt,
        input_audio: audioDataUri,
        duration,
        output_format: 'wav',
        normalization_strategy: 'peak',
      },
    }
  );

  // output is a ReadableStream or URL string depending on version
  if (typeof output === 'string') return output;
  if (output?.url) return output.url;
  if (typeof output === 'object' && output[Symbol.asyncIterator]) {
    // It's a stream — collect to get the URL
    let result;
    for await (const chunk of output) { result = chunk; }
    return typeof result === 'string' ? result : result?.url || result;
  }
  return output;
}

// ─── Helper: convert buffer to data URI ───

function bufferToDataUri(buffer, mimeType = 'audio/wav') {
  return `data:${mimeType};base64,${buffer.toString('base64')}`;
}

// ─── SSE: Server-Sent Events for progress ───

function sendSSE(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

// ─── Generate endpoint (SSE for progress) ───

app.post('/api/generate', upload.single('audio'), async (req, res) => {
  if (!replicate) {
    return res.status(500).json({ error: 'REPLICATE_API_TOKEN not configured. Set it in your environment.' });
  }

  if (!req.file) {
    return res.status(400).json({ error: 'No audio file uploaded' });
  }

  // Set up SSE
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  try {
    const voiceDataUri = bufferToDataUri(req.file.buffer, req.file.mimetype || 'audio/wav');
    const segmentUrls = [];

    // Generate 4 segments (continuation chain)
    for (let i = 0; i < 4; i++) {
      sendSSE(res, 'progress', { segment: i + 1, total: 4, status: `Generating section ${i + 1} of 4...` });

      let inputAudio;
      if (i === 0) {
        // First segment uses the voice recording
        inputAudio = voiceDataUri;
      } else {
        // Subsequent segments use the previous segment's URL directly
        // MusicGen melody model will extract the melodic contour
        inputAudio = segmentUrls[i - 1];
      }

      const prompt = CONTINUATION_PROMPTS[i] || CONTINUATION_PROMPTS[0];

      console.log(`[MusicGen] Generating segment ${i + 1}/4...`);
      const startTime = Date.now();

      const resultUrl = await callMusicGen(inputAudio, prompt, 30);

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`[MusicGen] Segment ${i + 1} complete in ${elapsed}s: ${resultUrl}`);

      segmentUrls.push(resultUrl);

      sendSSE(res, 'segment', { segment: i + 1, url: resultUrl });
    }

    sendSSE(res, 'complete', { segments: segmentUrls });
    res.end();
  } catch (err) {
    console.error('[MusicGen] Error:', err.message);
    sendSSE(res, 'error', { message: err.message });
    res.end();
  }
});

// ─── Single segment endpoint (faster, for Option B fallback) ───

app.post('/api/generate-single', upload.single('audio'), async (req, res) => {
  if (!replicate) {
    return res.status(500).json({ error: 'REPLICATE_API_TOKEN not configured' });
  }
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file uploaded' });
  }

  try {
    const voiceDataUri = bufferToDataUri(req.file.buffer, req.file.mimetype || 'audio/wav');
    console.log('[MusicGen] Generating single segment...');
    const startTime = Date.now();

    const resultUrl = await callMusicGen(voiceDataUri, MUSICGEN_PROMPT, 30);

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[MusicGen] Single segment complete in ${elapsed}s: ${resultUrl}`);

    res.json({ url: resultUrl });
  } catch (err) {
    console.error('[MusicGen] Error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─── Health check ───

app.get('/api/health', (req, res) => {
  res.json({ ok: true, hasApiKey: !!process.env.REPLICATE_API_TOKEN });
});

// ─── Start ───

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  if (!process.env.REPLICATE_API_TOKEN) {
    console.warn('WARNING: REPLICATE_API_TOKEN not set. Music generation will fail.');
  }
});
