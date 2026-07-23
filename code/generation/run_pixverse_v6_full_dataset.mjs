import fs from 'fs/promises';
import path from 'path';
import { pipeline } from 'stream/promises';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/api/v1';
const DEFAULT_MODEL = 'pixverse/pixverse-v6-t2v';
const DEFAULT_SIZE = '1280*720';
const DEFAULT_DURATION = 5;
const DEFAULT_AUDIO = false;
const DEFAULT_WATERMARK = false;
const DEFAULT_POLL_INTERVAL_MS = 20000;
const DEFAULT_TIMEOUT_MS = 6 * 60 * 60 * 1000;
const DEFAULT_SUBMIT_RETRY_DELAY_MS = 30000;
const DEFAULT_SUBMIT_MAX_RETRIES = 8;
const DEFAULT_MAX_INFLIGHT = 20;
const DEFAULT_SUBMIT_BATCH_SIZE = 20;
const DEFAULT_ENTRIES_PATH = path.join(__dirname, '..', 'kling_t2v', 'all_entries_merged_final.json');
const OUTPUT_ROOT = path.join(__dirname, 'outputs', 'pixverse_v6_720p_5s_full');

function parseArgs(argv) {
  const args = {
    wait: true,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
    model: process.env.PIXVERSE_MODEL?.trim() || DEFAULT_MODEL,
    baseUrl: (process.env.DASHSCOPE_BASE_URL?.trim() || DEFAULT_BASE_URL).replace(/\/+$/, ''),
    size: process.env.PIXVERSE_SIZE?.trim() || DEFAULT_SIZE,
    duration: Number(process.env.PIXVERSE_DURATION ?? DEFAULT_DURATION),
    audio:
      process.env.PIXVERSE_AUDIO == null
        ? DEFAULT_AUDIO
        : process.env.PIXVERSE_AUDIO.trim().toLowerCase() === 'true',
    watermark:
      process.env.PIXVERSE_WATERMARK == null
        ? DEFAULT_WATERMARK
        : process.env.PIXVERSE_WATERMARK.trim().toLowerCase() === 'true',
    shotType: process.env.PIXVERSE_SHOT_TYPE?.trim() || undefined,
    entriesPath: process.env.PIXVERSE_ENTRIES_PATH?.trim() || DEFAULT_ENTRIES_PATH,
    maxInflight: Number(process.env.PIXVERSE_MAX_INFLIGHT ?? DEFAULT_MAX_INFLIGHT),
    submitBatchSize: Number(process.env.PIXVERSE_SUBMIT_BATCH_SIZE ?? DEFAULT_SUBMIT_BATCH_SIZE),
    retryFailedCodes: [],
    retryFailedSamples: [],
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
    if (token === '--entries-json') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--entries-json` requires a file path.');
      }
      args.entriesPath = value;
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
    if (token === '--shot-type') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--shot-type` requires a value.');
      }
      args.shotType = value;
      i += 1;
      continue;
    }
    if (token === '--no-shot-type') {
      args.shotType = undefined;
      continue;
    }
    if (token === '--max-inflight') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--max-inflight` must be a positive integer.');
      }
      args.maxInflight = value;
      i += 1;
      continue;
    }
    if (token === '--submit-batch-size') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--submit-batch-size` must be a positive integer.');
      }
      args.submitBatchSize = value;
      i += 1;
      continue;
    }
    if (token === '--retry-failed-codes') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--retry-failed-codes` requires a comma-separated value.');
      }
      args.retryFailedCodes = value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
      i += 1;
      continue;
    }
    if (token === '--retry-failed-samples') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--retry-failed-samples` requires a comma-separated value.');
      }
      args.retryFailedSamples = value
        .split(',')
        .map((item) => Number(item.trim()))
        .filter((item) => Number.isInteger(item) && item > 0);
      i += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  if (!Number.isInteger(args.duration) || args.duration < 1 || args.duration > 15) {
    throw new Error('PixVerse-V6 duration must be an integer between 1 and 15 seconds.');
  }
  if (!Number.isInteger(args.maxInflight) || args.maxInflight <= 0) {
    throw new Error('`maxInflight` must be a positive integer.');
  }
  if (!Number.isInteger(args.submitBatchSize) || args.submitBatchSize <= 0) {
    throw new Error('`submitBatchSize` must be a positive integer.');
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

function previewPrompt(prompt, maxLength = 120) {
  if (!prompt) {
    return '';
  }
  return prompt.length <= maxLength ? prompt : `${prompt.slice(0, maxLength - 3)}...`;
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

function extractFailureCode(response) {
  return response?.output?.code ?? response?.code ?? null;
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

function buildOutputPaths(runDir) {
  return {
    videosDir: path.join(runDir, 'videos'),
    metadataDir: path.join(runDir, 'metadata'),
    batchesDir: path.join(runDir, 'batches'),
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
    const fileBase = `${String(task.sample_index).padStart(3, '0')}_${slugify(task.macro_domain, 48)}_${task.task_id}_${i + 1}`;
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
  const parameters = {
    size: args.size,
    duration: args.duration,
    audio: args.audio,
    watermark: args.watermark,
  };

  if (args.shotType) {
    parameters.shot_type = args.shotType;
  }

  return {
    model: args.model,
    input: {
      prompt,
    },
    parameters,
  };
}

function mapEntriesToSamples(entries, limit) {
  const sliced = limit ? entries.slice(0, limit) : entries;
  return sliced.map((entry, index) => ({
    sample_index: index + 1,
    source_entry_index: index,
    macro_domain: entry?.domain_info?.macro_domain ?? '',
    micro_domain: entry?.domain_info?.micro_domain ?? '',
    prompt: entry?.prompt ?? '',
    complexity: entry?.complexity ?? {},
  }));
}

function summarizeSelection(selectedEntries, maxItems = 5) {
  return selectedEntries.slice(0, maxItems).map((entry) => ({
    sample_index: entry.sample_index,
    macro_domain: entry.macro_domain,
    micro_domain: entry.micro_domain,
    prompt_preview: previewPrompt(entry.prompt, 180),
  }));
}

function countInFlightTasks(tasks) {
  return tasks.filter(
    (task) =>
      !Array.isArray(task.downloads) &&
      !isFailedStatus(task.task_status) &&
      task.submission_state === 'submitted',
  ).length;
}

function getPendingSamples(selectedEntries, tasks) {
  const submittedIndexes = new Set(tasks.map((task) => task.sample_index));
  return selectedEntries.filter((entry) => !submittedIndexes.has(entry.sample_index));
}

function resetTaskForRetry(task) {
  const attemptSnapshot = {
    task_id: task.task_id ?? null,
    task_status: task.task_status ?? null,
    submitted_at: task.submitted_at ?? null,
    submission_batch_index: task.submission_batch_index ?? null,
    submission_response: task.submission_response ?? null,
    last_query_response: task.last_query_response ?? null,
    last_polled_at: task.last_polled_at ?? null,
    failure_message: task.failure_message ?? null,
    failure_code: task.failure_code ?? extractFailureCode(task.last_query_response) ?? null,
    poll_error: task.poll_error ?? null,
    reset_at: new Date().toISOString(),
  };

  if (!Array.isArray(task.attempts)) {
    task.attempts = [];
  }
  task.attempts.push(attemptSnapshot);

  delete task.task_id;
  delete task.submission_batch_index;
  delete task.submission_payload;
  delete task.submission_response;
  delete task.last_query_response;
  delete task.last_polled_at;
  delete task.failure_message;
  delete task.failure_code;
  delete task.poll_error;
  delete task.downloads;
  delete task.submitted_at;

  task.submission_state = 'reset_for_retry';
  task.task_status = null;
}

function prepareRetryTasks(tasks, args) {
  const retryCodes = new Set(args.retryFailedCodes.map((code) => String(code).trim().toUpperCase()));
  const retrySamples = new Set(args.retryFailedSamples);
  const shouldRetryByCode = retryCodes.size > 0;
  const shouldRetryBySample = retrySamples.size > 0;

  if (!shouldRetryByCode && !shouldRetryBySample) {
    return [];
  }

  const retried = [];
  const kept = [];

  for (const task of tasks) {
    const failedWithoutDownloads =
      isFailedStatus(task.task_status) && (!Array.isArray(task.downloads) || task.downloads.length === 0);
    const failureCode = String(task.failure_code ?? extractFailureCode(task.last_query_response) ?? '')
      .trim()
      .toUpperCase();
    const matchesCode = shouldRetryByCode && retryCodes.has(failureCode);
    const matchesSample = shouldRetryBySample && retrySamples.has(task.sample_index);

    if (failedWithoutDownloads && (matchesCode || matchesSample)) {
      resetTaskForRetry(task);
      retried.push({
        sample_index: task.sample_index,
        previous_task_id: task.attempts?.at(-1)?.task_id ?? null,
        failure_code: failureCode || null,
      });
      continue;
    }

    kept.push(task);
  }

  tasks.length = 0;
  tasks.push(...kept);
  return retried;
}

async function persistSubmittedTasks(runDir, context, tasks) {
  await writeJson(path.join(runDir, 'submitted_tasks.json'), {
    created_at: new Date().toISOString(),
    ...context,
    tasks,
  });
}

async function persistRunSummary(runDir, context, selectedEntries, tasks) {
  const summary = {
    updated_at: new Date().toISOString(),
    ...context,
    selected_count: selectedEntries.length,
    total_tasks: tasks.length,
    succeeded_tasks: tasks.filter((task) => Array.isArray(task.downloads) && task.downloads.length > 0).length,
    failed_tasks: tasks.filter(
      (task) => isFailedStatus(task.task_status) && (!Array.isArray(task.downloads) || task.downloads.length === 0),
    ).length,
    inflight_tasks: countInFlightTasks(tasks),
    pending_submission_tasks: getPendingSamples(selectedEntries, tasks).length,
    tasks,
  };
  await writeJson(path.join(runDir, 'run_summary.json'), summary);
}

async function persistBatchSnapshot(runDir, batchIndex, batchTasks) {
  const paths = buildOutputPaths(runDir);
  await ensureDir(paths.batchesDir);
  const filePath = path.join(paths.batchesDir, `batch_${String(batchIndex).padStart(3, '0')}.json`);
  await writeJson(filePath, {
    created_at: new Date().toISOString(),
    batch_index: batchIndex,
    task_count: batchTasks.length,
    tasks: batchTasks,
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

function buildContext(args, entriesPath, submissionBatchIndex) {
  return {
    base_url: args.baseUrl,
    model: args.model,
    size: args.size,
    duration: args.duration,
    audio: args.audio,
    shot_type: args.shotType ?? null,
    watermark: args.watermark,
    entries_path: entriesPath,
    max_inflight: args.maxInflight,
    submit_batch_size: args.submitBatchSize,
    last_submission_batch_index: submissionBatchIndex,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = getRequiredEnv('DASHSCOPE_API_KEY');

  let runDir;
  let selectedEntries;
  let tasks;
  let entriesPath;
  let submissionBatchIndex = 0;

  if (args.resumeRunDir) {
    runDir = path.resolve(args.resumeRunDir);
    const resumeState = await loadResumeState(runDir);
    selectedEntries = resumeState.selectedEntries;
    tasks = resumeState.tasks;
    entriesPath = resumeState.config.entries_path;
    submissionBatchIndex = Number(resumeState.config.last_submission_batch_index ?? 0);

    if (!selectedEntries.length) {
      throw new Error(`No selected prompts found in resume directory: ${runDir}`);
    }

    console.log(`Resuming existing run: ${runDir}`);
    const retried = prepareRetryTasks(tasks, args);
    if (retried.length > 0) {
      console.log(
        `Reset ${retried.length} failed task(s) for retry: ${retried.map((item) => item.sample_index).join(', ')}`,
      );
    }
  } else {
    entriesPath = path.resolve(args.entriesPath);
    const entries = await readJson(entriesPath);
    selectedEntries = mapEntriesToSamples(entries, args.limit);

    if (!selectedEntries.length) {
      throw new Error(`No prompts found in ${entriesPath}`);
    }

    runDir = path.join(OUTPUT_ROOT, nowStamp());
    await ensureDir(runDir);

    await writeJson(path.join(runDir, 'run_config.json'), {
      created_at: new Date().toISOString(),
      ...buildContext(args, entriesPath, submissionBatchIndex),
      docs_reference: {
        model_doc: 'https://help.aliyun.com/zh/model-studio/pixverse-text-to-video-api-reference',
        video_guide: 'https://help.aliyun.com/zh/model-studio/use-video-generation',
        async_tasks_doc: 'https://help.aliyun.com/zh/model-studio/manage-asynchronous-tasks',
        notes: [
          'pixverse/pixverse-v6-t2v supports 720P output using size 1280*720 and duration 1-15 seconds.',
          'The official async task management API documents a 20 QPS limit for task query and batch query operations.',
          'The official PixVerse-V6 documentation does not publish a simultaneous in-flight task limit; this run uses a conservative in-flight window for stability.',
        ],
      },
      selected_count: selectedEntries.length,
    });

    await writeJson(path.join(runDir, 'selected_prompts.json'), {
      created_at: new Date().toISOString(),
      source_path: entriesPath,
      count: selectedEntries.length,
      selected_prompts: selectedEntries,
    });

    console.log(`Selected ${selectedEntries.length} prompts from dataset.`);
    console.log(`Model: ${args.model}`);
    console.log(`Base URL: ${args.baseUrl}`);
    console.log(`Size / duration: ${args.size} / ${args.duration}s`);
    console.log(`Audio / shot_type / watermark: ${args.audio} / ${args.shotType ?? 'none'} / ${args.watermark}`);
    console.log(`Entries source: ${entriesPath}`);
    console.log(`Max in-flight / submit batch size: ${args.maxInflight} / ${args.submitBatchSize}`);
    console.log('Official docs checked:');
    console.log('  - PixVerse text-to-video API: https://help.aliyun.com/zh/model-studio/pixverse-text-to-video-api-reference');
    console.log('  - Video generation guide: https://help.aliyun.com/zh/model-studio/use-video-generation');
    console.log('  - Async task management API: https://help.aliyun.com/zh/model-studio/manage-asynchronous-tasks');
    for (const item of summarizeSelection(selectedEntries)) {
      console.log(`[${item.sample_index}] ${item.macro_domain}`);
      console.log(`    ${item.micro_domain}`);
      console.log(`    ${item.prompt_preview}`);
    }

    tasks = [];
    await persistSubmittedTasks(runDir, buildContext(args, entriesPath, submissionBatchIndex), tasks);
    await persistRunSummary(runDir, buildContext(args, entriesPath, submissionBatchIndex), selectedEntries, tasks);
  }

  const startedAt = Date.now();

  while (true) {
    if (Date.now() - startedAt > args.timeoutMs) {
      console.log('Reached timeout. Current state has been persisted.');
      break;
    }

    const pendingSamples = getPendingSamples(selectedEntries, tasks);
    const inflightCount = countInFlightTasks(tasks);
    const availableSlots = Math.max(0, args.maxInflight - inflightCount);

    if (pendingSamples.length > 0 && availableSlots > 0) {
      const submitCount = Math.min(args.submitBatchSize, availableSlots, pendingSamples.length);
      const batchSamples = pendingSamples.slice(0, submitCount);
      submissionBatchIndex += 1;

      console.log(`\nSubmitting batch ${submissionBatchIndex} with ${batchSamples.length} task(s)...`);
      const batchTasks = [];

      for (const sample of batchSamples) {
        const payload = buildSubmissionPayload(args, sample.prompt);
        console.log(`Submitting [${sample.sample_index}/${selectedEntries.length}] ${sample.macro_domain}`);
        const response = await submitTaskWithRetries({
          baseUrl: args.baseUrl,
          apiKey,
          payload,
          sampleLabel: `[${sample.sample_index}] ${sample.macro_domain}`,
        });
        const taskId = extractTaskId(response);
        if (!taskId) {
          throw new Error(
            `Submission for sample [${sample.sample_index}] returned no task id: ${JSON.stringify(response, null, 2)}`,
          );
        }

        const task = {
          ...sample,
          task_id: taskId,
          submission_batch_index: submissionBatchIndex,
          submission_payload: payload,
          submission_response: response,
          submission_state: 'submitted',
          task_status: response?.output?.task_status ?? null,
          submitted_at: new Date().toISOString(),
        };
        tasks.push(task);
        batchTasks.push(task);

        await persistSubmittedTasks(runDir, buildContext(args, entriesPath, submissionBatchIndex), tasks);
      }

      await persistBatchSnapshot(runDir, submissionBatchIndex, batchTasks);
      await writeJson(path.join(runDir, 'run_config.json'), {
        ...(await readJson(path.join(runDir, 'run_config.json'))),
        last_submission_batch_index: submissionBatchIndex,
        updated_at: new Date().toISOString(),
      });
      await persistRunSummary(runDir, buildContext(args, entriesPath, submissionBatchIndex), selectedEntries, tasks);
    }

    const inflightTasks = tasks.filter(
      (task) =>
        !Array.isArray(task.downloads) &&
        !isFailedStatus(task.task_status) &&
        task.submission_state === 'submitted',
    );

    if (!args.wait) {
      console.log(`\nSubmission pass complete. Run directory: ${runDir}`);
      break;
    }

    if (inflightTasks.length === 0) {
      if (getPendingSamples(selectedEntries, tasks).length === 0) {
        console.log('\nAll tasks have been submitted and processed.');
      } else {
        console.log('\nNo in-flight tasks remain, but some prompts are still pending submission.');
      }
      await persistRunSummary(runDir, buildContext(args, entriesPath, submissionBatchIndex), selectedEntries, tasks);
      if (getPendingSamples(selectedEntries, tasks).length === 0) {
        break;
      }
      continue;
    }

    console.log(`\nPolling ${inflightTasks.length} in-flight task(s)...`);
    for (const task of inflightTasks) {
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
          if (!Array.isArray(task.downloads) || task.downloads.length === 0) {
            console.log(`  succeeded [${task.sample_index}] ${task.macro_domain}`);
            task.downloads = await saveGeneratedAssets(task, queryResponse, runDir);
          }
        } else if (isFailedStatus(status)) {
          task.failure_code = extractFailureCode(queryResponse);
          task.failure_message = extractFailureMessage(queryResponse);
          console.log(`  failed [${task.sample_index}] ${task.macro_domain}: ${task.failure_message}`);
        } else {
          console.log(`  ${String(status ?? 'UNKNOWN').toLowerCase()} [${task.sample_index}] ${task.macro_domain}`);
        }
      } catch (error) {
        task.last_polled_at = new Date().toISOString();
        task.poll_error = {
          message: error instanceof Error ? error.message : String(error),
          status: error?.status ?? null,
          body: error?.body ?? null,
        };
        console.log(`  query error for [${task.sample_index}] ${task.macro_domain}: ${task.poll_error.message}`);
      }
    }

    await persistSubmittedTasks(runDir, buildContext(args, entriesPath, submissionBatchIndex), tasks);
    await persistRunSummary(runDir, buildContext(args, entriesPath, submissionBatchIndex), selectedEntries, tasks);

    if (countInFlightTasks(tasks) > 0 || getPendingSamples(selectedEntries, tasks).length > 0) {
      await sleep(args.pollIntervalMs);
    }
  }

  await persistSubmittedTasks(runDir, buildContext(args, entriesPath, submissionBatchIndex), tasks);
  await persistRunSummary(runDir, buildContext(args, entriesPath, submissionBatchIndex), selectedEntries, tasks);
  console.log(`\nRun summary written to ${path.join(runDir, 'run_summary.json')}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
