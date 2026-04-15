import express from 'express';
import cors from 'cors';
import multer from 'multer';
import Replicate from 'replicate';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const TMP_DIR = join(__dirname, 'tmp');
if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });

// Shorter prompt = less processing overhead
const MUSICGEN_PROMPT =
  'cinematic ambient, atmospheric piano, ethereal pads, harp, emotional, film score, 60 BPM, minor key';

// Duration: 15s is enough for POC, halves cost vs 30s
const MUSICGEN_DURATION = 15;

const app = express();
app.use(cors());
app.use(express.static(join(__dirname, 'public')));
app.use('/audio', express.static(TMP_DIR));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

let replicate;
if (process.env.REPLICATE_API_TOKEN) {
  replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
}

// ─── Generate endpoint ───

app.post('/api/generate-single', upload.single('audio'), async (req, res) => {
  if (!replicate) {
    return res.status(500).json({ error: 'REPLICATE_API_TOKEN not configured' });
  }
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file uploaded' });
  }

  try {
    // Convert uploaded audio to data URI
    // Trim to max 10s worth of data — MusicGen only needs a few seconds of melody input
    // (Longer recordings waste compute without improving output quality)
    const maxInputBytes = 500 * 1024; // ~10s of webm audio
    const trimmedBuffer = req.file.buffer.length > maxInputBytes
      ? req.file.buffer.subarray(0, maxInputBytes)
      : req.file.buffer;
    const audioDataUri = `data:${req.file.mimetype || 'audio/webm'};base64,${trimmedBuffer.toString('base64')}`;

    console.log('[MusicGen] Starting generation...');
    const startTime = Date.now();

    const output = await replicate.run(
      'meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb',
      {
        input: {
          model_version: 'melody-large',
          prompt: MUSICGEN_PROMPT,
          input_audio: audioDataUri,
          duration: MUSICGEN_DURATION,
          output_format: 'mp3',          // mp3 = smaller download, faster transfer
          normalization_strategy: 'peak',
        },
      }
    );

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[MusicGen] Generation complete in ${elapsed}s`);
    console.log(`[MusicGen] Output type: ${typeof output}, constructor: ${output?.constructor?.name}`);

    // Extract the audio URL — Replicate returns a FileOutput object
    // where String(output) gives the URL
    const audioUrl = String(output);

    console.log(`[MusicGen] Audio URL: ${audioUrl}`);

    // Download the audio and serve locally (avoids CORS issues)
    if (audioUrl.startsWith('http')) {
      console.log('[MusicGen] Downloading to serve locally...');
      const audioResp = await fetch(audioUrl);
      if (!audioResp.ok) throw new Error(`Failed to download audio: ${audioResp.status}`);
      const audioBuffer = Buffer.from(await audioResp.arrayBuffer());
      const filename = `musicgen_${Date.now()}.mp3`;
      writeFileSync(join(TMP_DIR, filename), audioBuffer);
      console.log(`[MusicGen] Saved ${audioBuffer.length} bytes as ${filename}`);
      return res.json({ url: `/audio/${filename}` });
    }

    // Already a local path
    res.json({ url: audioUrl });

  } catch (err) {
    console.error('[MusicGen] Error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ ok: true, mode: 'replicate', hasApiKey: !!process.env.REPLICATE_API_TOKEN });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  if (!process.env.REPLICATE_API_TOKEN) {
    console.warn('WARNING: REPLICATE_API_TOKEN not set');
  } else {
    console.log('Using Replicate API for MusicGen');
  }
});
