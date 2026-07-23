import fs from 'fs/promises';
import path from 'path';
import { pipeline } from 'stream/promises';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/api/v1';
const DEFAULT_MODEL = 'vidu/viduq3-turbo_text2video';
const DEFAULT_RESOLUTION = '720P';
const DEFAULT_SIZE = '1280*720';
const DEFAULT_DURATION = 5;
const DEFAULT_AUDIO = false;
const DEFAULT_WATERMARK = false;
const DEFAULT_POLL_INTERVAL_MS = 20000;
const DEFAULT_TIMEOUT_MS = 60 * 60 * 1000;
const DEFAULT_SUBMIT_RETRY_DELAY_MS = 30000;
const DEFAULT_SUBMIT_MAX_RETRIES = 8;
const KLING_OUTPUT_ROOT = path.join(__dirname, '..', 'kling_t2v', 'outputs', 'kling_v3_720p_5s');
const OUTPUT_ROOT = path.join(__dirname, 'outputs', 'viduq3_turbo_720p_5s');

function parseArgs(argv) {
  const args = {
    wait: true,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
    model: process.env.VIDU_MODEL?.trim() || DEFAULT_MODEL,
    baseUrl: (process.env.DASHSCOPE_BASE_URL?.trim() || DEFAULT_BASE_URL).replace(/\/+$/, ''),
    resolution: process.env.VIDU_RESOLUTION?.trim() || DEFAULT_RESOLUTION,
    size: process.env.VIDU_SIZE?.trim() || DEFAULT_SIZE,
    duration: Number(process.env.VIDU_DURATION ?? DEFAULT_DURATION),
    audio:
      process.env.VIDU_AUDIO == null
        ? DEFAULT_AUDIO
        : process.env.VIDU_AUDIO.trim().toLowerCase() === 'true',
    watermark:
      process.env.VIDU_WATERMARK == null
        ? DEFAULT_WATERMARK
        : process.env.VIDU_WATERMARK.trim().toLowerCase() === 'true',
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];

    if (token === '--no-wait') {
      args.wait = false;
      continue;
    }
    if (token === '--wait') {
      args.wait = true;
      continue;
    }
    if (token === '--selected-prompts') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--selected-prompts` requires a file path.');
      }
      args.selectedPromptsPath = value;
      i += 1;
      continue;
    }
    if (token === '--resume-run-dir') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--resume-run-dir` requires a directory path.');
      }
      args.resumeRunDir = value;
      i += 1;
      continue;
    }
    if (token === '--timeout-minutes') {
      const value = Number(argv[i + 1]);
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error('`--timeout-minutes` must be a positive number.');
      }
      args.timeoutMs = Math.round(value * 60 * 1000);
      i += 1;
      continue;
    }
    if (token === '--poll-seconds') {
      const value = Number(argv[i + 1]);
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error('`--poll-seconds` must be a positive number.');
      }
      args.pollIntervalMs = Math.round(value * 1000);
      i += 1;
      continue;
    }
    if (token === '--limit') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--limit` must be a positive integer.');
      }
      args.limit = value;
      i += 1;
      continue;
    }
    if (token === '--model') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--model` requires a value.');
      }
      args.model = value;
      i += 1;
      continue;
    }
    if (token === '--base-url') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--base-url` requires a value.');
      }
      args.baseUrl = value.replace(/\/+$/, '');
      i += 1;
      continue;
    }
    if (token === '--resolution') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--resolution` requires a value.');
      }
      args.resolution = value;
      i += 1;
      continue;
    }
    if (token === '--size') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--size` requires a value.');
      }
      args.size = value;
      i += 1;
      continue;
    }
    if (token === '--duration') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--duration` must be a positive integer.');
      }
      args.duration = value;
      i += 1;
      continue;
    }
    if (token === '--audio') {
      args.audio = true;
      continue;
    }
    if (token === '--no-audio') {
      args.audio = false;
      continue;
    }
    if (token === '--watermark') {
      args.watermark = true;
      continue;
    }
    if (token === '--no-watermark') {
      args.watermark = false;
      continue;
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  if (!Number.isInteger(args.duration) || args.duration < 1 || args.duration > 16) {
    throw new Error('ViduQ3-Turbo duration must be an integer between 1 and 16 seconds.');
  }

  return args;
}

function getRequiredEnv(name) {
  const value = process.env[name];
  if (!value || !value.trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value.trim();
}

function nowStamp() {
  const now = new Date();
  const parts = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0'),
  ];
  return `${parts[0]}${parts[1]}${parts[2]}_${parts[3]}${parts[4]}${parts[5]}`;
}

function slugify(text, maxLength = 80) {
  return String(text ?? '')
    .normalize('NFKD')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, maxLength)
    .toLowerCase();
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function readJson(filePath) {
  const content = await fs.readFile(filePath, 'utf8');
  return JSON.parse(content);
}

async function writeJson(filePath, data) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

function previewPrompt(prompt, maxLength = 120) {
  if (!prompt) {
    return '';
  }
  return prompt.length <= maxLength ? prompt : `${prompt.slice(0, maxLength - 3)}...`;
}

async function findLatestSelectedPromptsPath() {
  const directoryEntries = await fs.readdir(KLING_OUTPUT_ROOT, { withFileTypes: true });
  const runDirs = directoryEntries
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(KLING_OUTPUT_ROOT, entry.name))
    .sort()
    .reverse();

  for (const runDir of runDirs) {
    const candidate = path.join(runDir, 'selected_prompts.json');
    try {
      await fs.access(candidate);
      return candidate;
    } catch {
      // Keep scanning older runs.
    }
  }

  throw new Error(`Unable to find any selected_prompts.json under ${KLING_OUTPUT_ROOT}`);
}

function makeHeaders(apiKey, includeAsync = false) {
  const headers = {
    Authorization: `Bearer ${apiKey}`,
  };

  if (includeAsync) {
    headers['X-DashScope-Async'] = 'enable';
    headers['Content-Type'] = 'application/json';
  }

  return headers;
}

async function readResponseBody(response) {
  const contentType = response.headers.get('content-type') ?? '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const body = await readResponseBody(response);

  if (!response.ok) {
    const message =
      typeof body === 'string'
        ? body
        : body?.message ?? body?.code ?? JSON.stringify(body);
    const error = new Error(`HTTP ${response.status} for ${url}: ${message}`);
    error.status = response.status;
    error.body = body;
    throw error;
  }

  return body;
}

async function submitTask({ baseUrl, apiKey, payload }) {
  return fetchJson(`${baseUrl}/services/aigc/video-generation/video-synthesis`, {
    method: 'POST',
    headers: makeHeaders(apiKey, true),
    body: JSON.stringify(payload),
  });
}

function isRetryableSubmissionError(error) {
  return [429, 500, 503, 504].includes(error?.status ?? -1);
}

async function submitTaskWithRetries({ baseUrl, apiKey, payload, sampleLabel }) {
  let lastError = null;

  for (let attempt = 1; attempt <= DEFAULT_SUBMIT_MAX_RETRIES; attempt += 1) {
    try {
      return await submitTask({ baseUrl, apiKey, payload });
    } catch (error) {
      lastError = error;
      if (!isRetryableSubmissionError(error) || attempt === DEFAULT_SUBMIT_MAX_RETRIES) {
        throw error;
      }

      const delayMs = DEFAULT_SUBMIT_RETRY_DELAY_MS * attempt;
      console.log(
        `  retrying submission for ${sampleLabel} after ${Math.round(delayMs / 1000)}s because of HTTP ${error.status}`,
      );
      await sleep(delayMs);
    }
  }

  throw lastError ?? new Error(`Submission failed for ${sampleLabel}`);
}

async function queryTask({ baseUrl, apiKey, taskId }) {
  return fetchJson(`${baseUrl}/tasks/${taskId}`, {
    method: 'GET',
    headers: makeHeaders(apiKey, false),
  });
}

function extractTaskId(response) {
  return response?.output?.task_id ?? response?.task_id ?? null;
}

function extractTaskStatus(response) {
  return response?.output?.task_status ?? response?.task_status ?? null;
}

function normalizeStatus(status) {
  return String(status ?? '').trim().toUpperCase();
}

function isSucceededStatus(status) {
  return normalizeStatus(status) === 'SUCCEEDED';
}

function isFailedStatus(status) {
  return ['FAILED', 'CANCELED', 'CANCELLED', 'EXPIRED', 'UNKNOWN'].includes(normalizeStatus(status));
}

function extractFailureMessage(response) {
  return response?.output?.message ?? response?.message ?? response?.code ?? 'Unknown error';
}

function extractVideoOutputs(response) {
  const outputs = [];
  const seenUrls = new Set();

  function addOutput(candidate) {
    if (!candidate?.url || seenUrls.has(candidate.url)) {
      return;
    }
    seenUrls.add(candidate.url);
    outputs.push(candidate);
  }

  const videoUrl = response?.output?.video_url;
  if (videoUrl) {
    addOutput({
      url: videoUrl,
      orig_prompt: response?.output?.orig_prompt ?? null,
      actual_prompt: response?.output?.actual_prompt ?? null,
    });
  }

  const results = response?.output?.results;
  if (Array.isArray(results)) {
    for (const item of results) {
      if (item?.video_url) {
        addOutput({
          url: item.video_url,
          orig_prompt: item.orig_prompt ?? null,
          actual_prompt: item.actual_prompt ?? null,
        });
      }
    }
  }

  return outputs;
}

async function downloadFile(url, destination) {
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${url}: HTTP ${response.status}`);
  }

  await ensureDir(path.dirname(destination));
  const { createWriteStream } = await import('fs');
  await pipeline(response.body, createWriteStream(destination));
}

function summarizeSelection(selectedEntries) {
  return selectedEntries.map((entry) => ({
    sample_index: entry.sample_index,
    macro_domain: entry.macro_domain,
    micro_domain: entry.micro_domain,
    prompt_preview: previewPrompt(entry.prompt, 180),
  }));
}

function buildOutputPaths(runDir) {
  return {
    videosDir: path.join(runDir, 'videos'),
    metadataDir: path.join(runDir, 'metadata'),
  };
}

async function saveGeneratedAssets(task, queryResponse, runDir) {
  const outputs = extractVideoOutputs(queryResponse);
  if (outputs.length === 0) {
    throw new Error(`Task ${task.task_id} succeeded but returned no downloadable video URL.`);
  }

  const paths = buildOutputPaths(runDir);
  await ensureDir(paths.videosDir);
  await ensureDir(paths.metadataDir);

  const downloads = [];
  for (let i = 0; i < outputs.length; i += 1) {
    const output = outputs[i];
    const fileBase = `${String(task.sample_index).padStart(2, '0')}_${slugify(task.macro_domain, 48)}_${task.task_id}_${i + 1}`;
    const videoPath = path.join(paths.videosDir, `${fileBase}.mp4`);
    const metadataPath = path.join(paths.metadataDir, `${fileBase}.json`);

    await downloadFile(output.url, videoPath);
    await writeJson(metadataPath, {
      task,
      queryResponse,
      downloaded_at: new Date().toISOString(),
      video_index: i + 1,
    });

    downloads.push({
      ...output,
      video_path: videoPath,
      metadata_path: metadataPath,
    });
  }

  return downloads;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildSubmissionPayload(args, prompt) {
  return {
    model: args.model,
    input: {
      prompt,
    },
    parameters: {
      resolution: args.resolution,
      size: args.size,
      duration: args.duration,
      audio: args.audio,
      watermark: args.watermark,
    },
  };
}

async function persistSubmittedTasks(runDir, context, tasks) {
  await writeJson(path.join(runDir, 'submitted_tasks.json'), {
    created_at: new Date().toISOString(),
    ...context,
    tasks,
  });
}

async function loadResumeState(runDir) {
  const config = await readJson(path.join(runDir, 'run_config.json'));
  const selectedPrompts = await readJson(path.join(runDir, 'selected_prompts.json'));
  const submittedTasks = await readJson(path.join(runDir, 'submitted_tasks.json'));

  return {
    config,
    selectedEntries: selectedPrompts.selected_prompts ?? [],
    tasks: submittedTasks.tasks ?? [],
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = getRequiredEnv('DASHSCOPE_API_KEY');

  let runDir;
  let selectedEntries;
  let tasks;
  let selectedPromptsPath;

  if (args.resumeRunDir) {
    runDir = path.resolve(args.resumeRunDir);
    const resumeState = await loadResumeState(runDir);
    selectedEntries = resumeState.selectedEntries;
    tasks = resumeState.tasks;
    selectedPromptsPath = resumeState.config.selected_prompts_path;

    if (!selectedEntries.length) {
      throw new Error(`No selected prompts found in resume directory: ${runDir}`);
    }
    if (!tasks.length) {
      throw new Error(`No submitted tasks found in resume directory: ${runDir}`);
    }

    console.log(`Resuming existing run: ${runDir}`);
  } else {
    selectedPromptsPath = path.resolve(args.selectedPromptsPath ?? (await findLatestSelectedPromptsPath()));
    const selectedPrompts = await readJson(selectedPromptsPath);
    const availableEntries = selectedPrompts.selected_prompts ?? [];
    selectedEntries = args.limit ? availableEntries.slice(0, args.limit) : availableEntries;

    if (!selectedEntries.length) {
      throw new Error(`No prompts found in ${selectedPromptsPath}`);
    }

    runDir = path.join(OUTPUT_ROOT, nowStamp());
    await ensureDir(runDir);

    await writeJson(path.join(runDir, 'run_config.json'), {
      created_at: new Date().toISOString(),
      base_url: args.baseUrl,
      model: args.model,
      resolution: args.resolution,
      size: args.size,
      duration: args.duration,
      audio: args.audio,
      watermark: args.watermark,
      selected_prompts_path: selectedPromptsPath,
      selected_count: selectedEntries.length,
    });

    await writeJson(path.join(runDir, 'selected_prompts.json'), {
      created_at: new Date().toISOString(),
      source_path: selectedPromptsPath,
      count: selectedEntries.length,
      selected_prompts: selectedEntries,
    });

    console.log(`Selected ${selectedEntries.length} prompts.`);
    console.log(`Model: ${args.model}`);
    console.log(`Base URL: ${args.baseUrl}`);
    console.log(`Resolution / size / duration: ${args.resolution} / ${args.size} / ${args.duration}s`);
    console.log(`Audio / watermark: ${args.audio} / ${args.watermark}`);
    console.log(`Source prompts: ${selectedPromptsPath}`);
    for (const item of summarizeSelection(selectedEntries)) {
      console.log(`[${item.sample_index}] ${item.macro_domain}`);
      console.log(`    ${item.micro_domain}`);
      console.log(`    ${item.prompt_preview}`);
    }

    const firstSample = selectedEntries[0];
    const probePayload = buildSubmissionPayload(args, firstSample.prompt);

    console.log(`\nSubmitting probe task with sample [${firstSample.sample_index}]...`);
    const probeResponse = await submitTaskWithRetries({
      baseUrl: args.baseUrl,
      apiKey,
      payload: probePayload,
      sampleLabel: `[${firstSample.sample_index}] ${firstSample.macro_domain}`,
    });
    const probeTaskId = extractTaskId(probeResponse);
    if (!probeTaskId) {
      throw new Error(`Probe submission returned no task id: ${JSON.stringify(probeResponse, null, 2)}`);
    }

    await writeJson(path.join(runDir, 'probe_submission.json'), {
      created_at: new Date().toISOString(),
      payload: probePayload,
      response: probeResponse,
    });

    console.log(`Probe task accepted: ${probeTaskId}`);

    tasks = [
      {
        ...firstSample,
        task_id: probeTaskId,
        submission_payload: probePayload,
        submission_response: probeResponse,
        submission_state: 'submitted',
        used_probe_submission: true,
      },
    ];

    await persistSubmittedTasks(
      runDir,
      {
        base_url: args.baseUrl,
        model: args.model,
        resolution: args.resolution,
        size: args.size,
        duration: args.duration,
        audio: args.audio,
        watermark: args.watermark,
      },
      tasks,
    );
  }

  const submittedIndexes = new Set(tasks.map((task) => task.sample_index));

  for (const sample of selectedEntries.filter((entry) => !submittedIndexes.has(entry.sample_index))) {
    const payload = buildSubmissionPayload(args, sample.prompt);

    console.log(`Submitting [${sample.sample_index}] ${sample.macro_domain}`);
    const response = await submitTaskWithRetries({
      baseUrl: args.baseUrl,
      apiKey,
      payload,
      sampleLabel: `[${sample.sample_index}] ${sample.macro_domain}`,
    });
    const taskId = extractTaskId(response);
    if (!taskId) {
      throw new Error(`Submission for sample [${sample.sample_index}] returned no task id: ${JSON.stringify(response, null, 2)}`);
    }

    tasks.push({
      ...sample,
      task_id: taskId,
      submission_payload: payload,
      submission_response: response,
      submission_state: 'submitted',
      used_probe_submission: false,
    });

    await persistSubmittedTasks(
      runDir,
      {
        base_url: args.baseUrl,
        model: args.model,
        resolution: args.resolution,
        size: args.size,
        duration: args.duration,
        audio: args.audio,
        watermark: args.watermark,
      },
      tasks,
    );
  }

  console.log(`\nAll tasks submitted. Run directory: ${runDir}`);

  if (!args.wait) {
    return;
  }

  const startedAt = Date.now();
  let pending = new Set(tasks.map((task) => task.task_id));

  while (pending.size > 0) {
    if (Date.now() - startedAt > args.timeoutMs) {
      break;
    }

    console.log(`\nPolling ${pending.size} pending task(s)...`);
    const currentPending = tasks.filter((task) => pending.has(task.task_id));

    for (const task of currentPending) {
      try {
        const queryResponse = await queryTask({
          baseUrl: args.baseUrl,
          apiKey,
          taskId: task.task_id,
        });
        const status = extractTaskStatus(queryResponse);
        task.last_query_response = queryResponse;
        task.last_polled_at = new Date().toISOString();
        task.task_status = status;

        if (isSucceededStatus(status)) {
          console.log(`  succeeded [${task.sample_index}] ${task.macro_domain}`);
          task.downloads = await saveGeneratedAssets(task, queryResponse, runDir);
          pending.delete(task.task_id);
        } else if (isFailedStatus(status)) {
          console.log(
            `  failed [${task.sample_index}] ${task.macro_domain}: ${extractFailureMessage(queryResponse)}`,
          );
          pending.delete(task.task_id);
        } else {
          console.log(`  ${String(status ?? 'UNKNOWN').toLowerCase()} [${task.sample_index}] ${task.macro_domain}`);
        }
      } catch (error) {
        console.log(`  query error for [${task.sample_index}] ${task.macro_domain}: ${error.message}`);
      }
    }

    await persistSubmittedTasks(
      runDir,
      {
        base_url: args.baseUrl,
        model: args.model,
        resolution: args.resolution,
        size: args.size,
        duration: args.duration,
        audio: args.audio,
        watermark: args.watermark,
      },
      tasks,
    );

    if (pending.size > 0) {
      await sleep(args.pollIntervalMs);
    }
  }

  const summary = {
    completed_at: new Date().toISOString(),
    run_dir: runDir,
    total_tasks: tasks.length,
    succeeded_tasks: tasks.filter((task) => Array.isArray(task.downloads) && task.downloads.length > 0).length,
    failed_tasks: tasks.filter(
      (task) => isFailedStatus(task.task_status) && (!Array.isArray(task.downloads) || task.downloads.length === 0),
    ).length,
    pending_tasks: tasks.filter((task) => !isFailedStatus(task.task_status) && !isSucceededStatus(task.task_status)).map((task) => ({
      sample_index: task.sample_index,
      macro_domain: task.macro_domain,
      task_id: task.task_id,
      task_status: task.task_status ?? null,
    })),
    tasks,
  };

  await writeJson(path.join(runDir, 'run_summary.json'), summary);

  console.log(`\nSummary written to ${path.join(runDir, 'run_summary.json')}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
