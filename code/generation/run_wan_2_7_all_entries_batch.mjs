import fs from 'fs/promises';
import path from 'path';
import { pipeline } from 'stream/promises';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/api/v1';
const DEFAULT_MODEL = 'wan2.7-t2v';
const DEFAULT_RATIO = '16:9';
const DEFAULT_RESOLUTION = '720P';
const DEFAULT_DURATION = 5;
const DEFAULT_POLL_INTERVAL_MS = 15000;
const DEFAULT_TIMEOUT_MS = 12 * 60 * 60 * 1000;
const DEFAULT_SUBMIT_RETRY_DELAY_MS = 30000;
const DEFAULT_SUBMIT_MAX_RETRIES = 8;
const OFFICIAL_MAX_SUBMIT_RPS = 5;
const OFFICIAL_MAX_CONCURRENT_TASKS = 5;
const DEFAULT_BATCH_SIZE = OFFICIAL_MAX_CONCURRENT_TASKS;
const DEFAULT_ENTRIES_PATH = path.join(__dirname, '..', 'kling_t2v', 'all_entries_merged_final.json');
const OUTPUT_ROOT = path.join(__dirname, 'outputs', 'wan_2_7_720p_5s_all_entries');

function parseArgs(argv) {
  const args = {
    wait: true,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
    model: process.env.WAN_MODEL?.trim() || DEFAULT_MODEL,
    baseUrl: (process.env.DASHSCOPE_BASE_URL?.trim() || DEFAULT_BASE_URL).replace(/\/+$/, ''),
    ratio: process.env.WAN_RATIO?.trim() || DEFAULT_RATIO,
    resolution: process.env.WAN_RESOLUTION?.trim() || DEFAULT_RESOLUTION,
    duration: Number(process.env.WAN_DURATION ?? DEFAULT_DURATION),
    promptExtend:
      process.env.WAN_PROMPT_EXTEND == null
        ? false
        : process.env.WAN_PROMPT_EXTEND.trim().toLowerCase() === 'true',
    watermark:
      process.env.WAN_WATERMARK == null
        ? false
        : process.env.WAN_WATERMARK.trim().toLowerCase() === 'true',
    entriesPath: process.env.WAN_ENTRIES_PATH?.trim() || DEFAULT_ENTRIES_PATH,
    maxConcurrent: OFFICIAL_MAX_CONCURRENT_TASKS,
    submitRps: OFFICIAL_MAX_SUBMIT_RPS,
    batchSize: DEFAULT_BATCH_SIZE,
    cliOverrides: {
      pollIntervalMs: false,
      maxConcurrent: false,
      batchSize: false,
    },
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
    if (token === '--entries') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--entries` requires a file path.');
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
      args.cliOverrides.pollIntervalMs = true;
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
    if (token === '--start-index') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--start-index` must be a positive integer.');
      }
      args.startIndex = value;
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
    if (token === '--ratio') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--ratio` requires a value.');
      }
      args.ratio = value;
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
    if (token === '--duration') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--duration` must be a positive integer.');
      }
      args.duration = value;
      i += 1;
      continue;
    }
    if (token === '--prompt-extend') {
      args.promptExtend = true;
      continue;
    }
    if (token === '--no-prompt-extend') {
      args.promptExtend = false;
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
    if (token === '--max-concurrent') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0 || value > OFFICIAL_MAX_CONCURRENT_TASKS) {
        throw new Error(
          `\`--max-concurrent\` must be a positive integer and cannot exceed the official limit ${OFFICIAL_MAX_CONCURRENT_TASKS}.`,
        );
      }
      args.maxConcurrent = value;
      args.cliOverrides.maxConcurrent = true;
      i += 1;
      continue;
    }
    if (token === '--submit-rps') {
      const value = Number(argv[i + 1]);
      if (!Number.isFinite(value) || value <= 0 || value > OFFICIAL_MAX_SUBMIT_RPS) {
        throw new Error(
          `\`--submit-rps\` must be a positive number and cannot exceed the official limit ${OFFICIAL_MAX_SUBMIT_RPS}.`,
        );
      }
      args.submitRps = value;
      i += 1;
      continue;
    }
    if (token === '--batch-size') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--batch-size` must be a positive integer.');
      }
      args.batchSize = value;
      args.cliOverrides.batchSize = true;
      i += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  if (!Number.isInteger(args.duration) || args.duration < 2 || args.duration > 15) {
    throw new Error('Wan 2.7 duration must be an integer between 2 and 15 seconds.');
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function readJson(filePath) {
  const content = await fs.readFile(filePath, 'utf8');
  return JSON.parse(content.replace(/^\uFEFF/, ''));
}

async function writeJson(filePath, data) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

async function writeText(filePath, text) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, text, 'utf8');
}

function previewPrompt(prompt, maxLength = 120) {
  if (!prompt) {
    return '';
  }
  return prompt.length <= maxLength ? prompt : `${prompt.slice(0, maxLength - 3)}...`;
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
  return ['FAILED', 'CANCELED', 'CANCELLED', 'EXPIRED'].includes(normalizeStatus(status));
}

function isAccountAccessDeniedError(error) {
  const message = error instanceof Error ? error.message : String(error ?? '');
  return /access denied|good standing|overdue-payment/i.test(message);
}

function isFinalStatus(status) {
  return isSucceededStatus(status) || isFailedStatus(status);
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

function normalizeEntries(entries) {
  return entries.map((entry, index) => ({
    source_index: entry?.source_index ?? index + 1,
    sample_index: entry?.sample_index ?? index + 1,
    macro_domain:
      entry?.macro_domain ?? entry?.domain_info?.macro_domain ?? 'Unknown Macro Domain',
    micro_domain: entry?.micro_domain ?? entry?.domain_info?.micro_domain ?? '',
    prompt: entry?.prompt ?? '',
    complexity: entry?.complexity ?? {},
    source_entry: entry?.source_entry ?? entry,
  }));
}

function applyEntrySlice(entries, args) {
  let sliced = entries;

  if (args.startIndex) {
    sliced = sliced.slice(args.startIndex - 1);
  }
  if (args.limit) {
    sliced = sliced.slice(0, args.limit);
  }

  return sliced.map((entry, idx) => ({
    ...entry,
    sample_index: args.startIndex || args.limit ? idx + 1 : entry.sample_index,
  }));
}

function summarizeSelection(selectedEntries) {
  return selectedEntries.map((entry) => ({
    sample_index: entry.sample_index,
    source_index: entry.source_index,
    macro_domain: entry.macro_domain,
    micro_domain: entry.micro_domain,
    prompt_preview: previewPrompt(entry.prompt, 180),
  }));
}

function buildSubmissionPayload(args, prompt) {
  return {
    model: args.model,
    input: {
      prompt,
    },
    parameters: {
      resolution: args.resolution,
      ratio: args.ratio,
      duration: args.duration,
      prompt_extend: args.promptExtend,
      watermark: args.watermark,
    },
  };
}

function buildOutputPaths(runDir) {
  return {
    videosDir: path.join(runDir, 'videos'),
    metadataDir: path.join(runDir, 'metadata'),
    batchesDir: path.join(runDir, 'batches'),
  };
}

function getTaskFileBase(task, totalCount) {
  const padWidth = String(totalCount).length;
  return `${String(task.sample_index).padStart(padWidth, '0')}_${slugify(task.macro_domain, 48)}_${task.task_id}_1`;
}

async function saveGeneratedAssets(task, queryResponse, runDir, totalCount) {
  const outputs = extractVideoOutputs(queryResponse);
  if (outputs.length === 0) {
    throw new Error(`Task ${task.task_id} succeeded but returned no downloadable video URL.`);
  }

  const paths = buildOutputPaths(runDir);
  await ensureDir(paths.videosDir);
  await ensureDir(paths.metadataDir);

  const downloaded = [];
  for (let i = 0; i < outputs.length; i += 1) {
    const output = outputs[i];
    const fileBase = outputs.length === 1
      ? getTaskFileBase(task, totalCount)
      : `${getTaskFileBase(task, totalCount).replace(/_1$/, '')}_${i + 1}`;
    const videoPath = path.join(paths.videosDir, `${fileBase}.mp4`);
    const metadataPath = path.join(paths.metadataDir, `${fileBase}.json`);

    await downloadFile(output.url, videoPath);
    await writeJson(metadataPath, {
      task,
      queryResponse,
      downloaded_at: new Date().toISOString(),
      video_index: i + 1,
    });

    downloaded.push({
      ...output,
      video_path: videoPath,
      metadata_path: metadataPath,
    });
  }

  return downloaded;
}

function getBatchFilePath(runDir, batchNumber) {
  return path.join(runDir, 'batches', `batch_${String(batchNumber).padStart(3, '0')}.json`);
}

async function persistBatchFiles(runDir, tasks, totalCount) {
  const batches = new Map();

  for (const task of tasks) {
    if (!task.batch_number) {
      continue;
    }
    if (!batches.has(task.batch_number)) {
      batches.set(task.batch_number, []);
    }
    batches.get(task.batch_number).push({
      sample_index: task.sample_index,
      source_index: task.source_index,
      macro_domain: task.macro_domain,
      micro_domain: task.micro_domain,
      task_id: task.task_id ?? null,
      task_status: task.task_status ?? null,
      submission_state: task.submission_state ?? null,
      has_downloads: Array.isArray(task.downloads) && task.downloads.length > 0,
      prompt_preview: previewPrompt(task.prompt, 180),
      file_base: task.task_id ? getTaskFileBase(task, totalCount) : null,
    });
  }

  for (const [batchNumber, batchTasks] of batches.entries()) {
    await writeJson(getBatchFilePath(runDir, batchNumber), {
      created_at: new Date().toISOString(),
      batch_number: batchNumber,
      count: batchTasks.length,
      tasks: batchTasks,
    });
  }
}

function buildContext(args, entriesPath, totalCount) {
  return {
    base_url: args.baseUrl,
    model: args.model,
    ratio: args.ratio,
    resolution: args.resolution,
    duration: args.duration,
    prompt_extend: args.promptExtend,
    watermark: args.watermark,
    entries_path: entriesPath,
    selected_count: totalCount,
    official_limits: {
      submit_rps: OFFICIAL_MAX_SUBMIT_RPS,
      max_concurrent_tasks: OFFICIAL_MAX_CONCURRENT_TASKS,
    },
    configured_limits: {
      submit_rps: args.submitRps,
      max_concurrent_tasks: args.maxConcurrent,
      batch_size: args.batchSize,
      poll_interval_ms: args.pollIntervalMs,
    },
  };
}

function buildRunSummary(runDir, tasks, context) {
  const succeeded = tasks.filter((task) => isSucceededStatus(task.task_status)).length;
  const failed = tasks.filter((task) => isFailedStatus(task.task_status)).length;
  const submitted = tasks.filter((task) => task.task_id).length;
  const pendingSubmission = tasks.filter((task) => !task.task_id).length;
  const active = tasks.filter((task) => task.task_id && !isFinalStatus(task.task_status)).length;
  const downloaded = tasks.filter((task) => Array.isArray(task.downloads) && task.downloads.length > 0).length;

  return {
    created_at: new Date().toISOString(),
    run_dir: runDir,
    ...context,
    totals: {
      tasks: tasks.length,
      submitted,
      pending_submission: pendingSubmission,
      active,
      succeeded,
      failed,
      downloaded,
      complete: succeeded + failed,
    },
    pending_task_ids: tasks
      .filter((task) => task.task_id && !isFinalStatus(task.task_status))
      .map((task) => task.task_id),
    tasks,
  };
}

async function persistRunState(runDir, tasks, context) {
  await writeJson(path.join(runDir, 'submitted_tasks.json'), {
    created_at: new Date().toISOString(),
    ...context,
    tasks,
  });
  await persistBatchFiles(runDir, tasks, context.selected_count);
  await writeJson(path.join(runDir, 'run_summary.json'), buildRunSummary(runDir, tasks, context));
}

async function loadResumeState(runDir) {
  const config = await readJson(path.join(runDir, 'run_config.json'));
  const selectedEntriesPayload = await readJson(path.join(runDir, 'selected_entries.json'));
  const submittedTasks = await readJson(path.join(runDir, 'submitted_tasks.json'));

  return {
    config,
    selectedEntries: selectedEntriesPayload.selected_entries ?? [],
    tasks: submittedTasks.tasks ?? [],
  };
}

function createTaskSkeletons(selectedEntries) {
  return selectedEntries.map((entry) => ({
    ...entry,
    task_id: null,
    batch_number: null,
    batch_position: null,
    submission_payload: null,
    submission_response: null,
    submission_state: 'pending_submission',
    task_status: null,
    last_query_response: null,
    last_polled_at: null,
    failure_message: null,
    downloads: [],
    poll_error: null,
  }));
}

function getActiveTasks(tasks) {
  return tasks.filter((task) => task.task_id && !isFinalStatus(task.task_status));
}

function getPendingSubmissionTasks(tasks) {
  return tasks.filter((task) => !task.task_id);
}

function countSubmittedTasks(tasks) {
  return tasks.filter((task) => task.task_id).length;
}

async function ensureCompletedDownloads(runDir, task, totalCount) {
  if (!isSucceededStatus(task.task_status) || (task.downloads && task.downloads.length > 0)) {
    return false;
  }

  if (!task.last_query_response) {
    return false;
  }

  task.downloads = await saveGeneratedAssets(task, task.last_query_response, runDir, totalCount);
  return true;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = getRequiredEnv('DASHSCOPE_API_KEY');

  let runDir;
  let selectedEntries;
  let tasks;
  let context;

  if (args.resumeRunDir) {
    runDir = path.resolve(args.resumeRunDir);
    const resumeState = await loadResumeState(runDir);
    selectedEntries = resumeState.selectedEntries;
    tasks = resumeState.tasks;
    context = buildContext(
      {
        ...args,
        ...resumeState.config,
        baseUrl: resumeState.config.base_url,
        promptExtend: resumeState.config.prompt_extend,
        watermark: resumeState.config.watermark,
        submitRps: resumeState.config.configured_limits?.submit_rps ?? args.submitRps,
        pollIntervalMs: args.cliOverrides.pollIntervalMs
          ? args.pollIntervalMs
          : (resumeState.config.configured_limits?.poll_interval_ms ?? args.pollIntervalMs),
        maxConcurrent: args.cliOverrides.maxConcurrent
          ? args.maxConcurrent
          : (resumeState.config.configured_limits?.max_concurrent_tasks ?? args.maxConcurrent),
        batchSize: args.cliOverrides.batchSize
          ? args.batchSize
          : (resumeState.config.configured_limits?.batch_size ?? args.batchSize),
      },
      resumeState.config.entries_path,
      selectedEntries.length,
    );

    if (!selectedEntries.length) {
      throw new Error(`No selected entries found in resume directory: ${runDir}`);
    }

    console.log(`Resuming existing run: ${runDir}`);
    await writeText(path.join(OUTPUT_ROOT, 'latest_run.txt'), `${runDir}\n`);
  } else {
    const entriesPath = path.resolve(args.entriesPath);
    const allEntries = await readJson(entriesPath);

    if (!Array.isArray(allEntries) || allEntries.length === 0) {
      throw new Error(`Input entries file does not contain a non-empty array: ${entriesPath}`);
    }

    selectedEntries = applyEntrySlice(normalizeEntries(allEntries), args);
    if (!selectedEntries.length) {
      throw new Error(`No prompts selected from ${entriesPath}`);
    }

    runDir = path.join(OUTPUT_ROOT, `${nowStamp()}_all-entries-${selectedEntries.length}`);
    await ensureDir(runDir);
    await ensureDir(path.join(runDir, 'batches'));
    await writeText(path.join(OUTPUT_ROOT, 'latest_run.txt'), `${runDir}\n`);

    context = buildContext(args, entriesPath, selectedEntries.length);
    tasks = createTaskSkeletons(selectedEntries);

    await writeJson(path.join(runDir, 'run_config.json'), {
      created_at: new Date().toISOString(),
      ...context,
    });

    await writeJson(path.join(runDir, 'selected_entries.json'), {
      created_at: new Date().toISOString(),
      source_path: entriesPath,
      count: selectedEntries.length,
      selected_entries: selectedEntries,
    });

    console.log(`Run directory: ${runDir}`);
    console.log(`Selected ${selectedEntries.length} prompts from ${entriesPath}`);
    console.log(`Model: ${args.model}`);
    console.log(`Base URL: ${args.baseUrl}`);
    console.log(`Resolution / duration / ratio: ${args.resolution} / ${args.duration}s / ${args.ratio}`);
    console.log(`prompt_extend / watermark: ${args.promptExtend} / ${args.watermark}`);
    console.log(
      `Official limit / configured limit: concurrency ${OFFICIAL_MAX_CONCURRENT_TASKS}/${args.maxConcurrent}, submit RPS ${OFFICIAL_MAX_SUBMIT_RPS}/${args.submitRps}`,
    );

    for (const item of summarizeSelection(selectedEntries).slice(0, 5)) {
      console.log(
        `[${item.sample_index}/${selectedEntries.length}] source #${item.source_index} ${item.macro_domain}`,
      );
      console.log(`    ${item.micro_domain}`);
      console.log(`    ${item.prompt_preview}`);
    }
    if (selectedEntries.length > 5) {
      console.log(`    ... and ${selectedEntries.length - 5} more prompts`);
    }

    await persistRunState(runDir, tasks, context);
  }

  for (const task of tasks) {
    if (await ensureCompletedDownloads(runDir, task, selectedEntries.length)) {
      console.log(`Recovered downloads for [${task.sample_index}] ${task.macro_domain}`);
    }
  }
  await persistRunState(runDir, tasks, context);

  const submitDelayMs = Math.ceil(1000 / context.configured_limits.submit_rps);
  const startedAt = Date.now();
  let submissionBlockedError = null;

  while (true) {
    if (Date.now() - startedAt > args.timeoutMs) {
      console.log('\nTimeout reached; current state has been saved.');
      break;
    }

    let activeTasks = getActiveTasks(tasks);
    const pendingSubmissionTasks = getPendingSubmissionTasks(tasks);

    while (
      activeTasks.length < context.configured_limits.max_concurrent_tasks &&
      pendingSubmissionTasks.length > 0
    ) {
      const task = pendingSubmissionTasks.shift();
      const submissionOrder = countSubmittedTasks(tasks) + 1;
      const batchNumber = Math.floor((submissionOrder - 1) / context.configured_limits.batch_size) + 1;
      const batchPosition = ((submissionOrder - 1) % context.configured_limits.batch_size) + 1;
      const payload = buildSubmissionPayload(args, task.prompt);

      console.log(
        `Submitting [${task.sample_index}/${selectedEntries.length}] batch ${batchNumber} pos ${batchPosition} ${task.macro_domain}`,
      );
      let response;
      try {
        response = await submitTaskWithRetries({
          baseUrl: context.base_url,
          apiKey,
          payload,
          sampleLabel: `[${task.sample_index}] ${task.macro_domain}`,
        });
        submissionBlockedError = null;
      } catch (error) {
        if (!isAccountAccessDeniedError(error)) {
          throw error;
        }
        submissionBlockedError = error instanceof Error ? error : new Error(String(error));
        console.log(
          `Submission blocked for [${task.sample_index}] ${task.macro_domain} -> ${submissionBlockedError.message}`,
        );
        break;
      }

      const taskId = extractTaskId(response);
      if (!taskId) {
        throw new Error(
          `Submission for sample [${task.sample_index}] returned no task id: ${JSON.stringify(response, null, 2)}`,
        );
      }

      task.task_id = taskId;
      task.batch_number = batchNumber;
      task.batch_position = batchPosition;
      task.submission_payload = payload;
      task.submission_response = response;
      task.submission_state = 'submitted';
      task.task_status = extractTaskStatus(response);

      activeTasks = getActiveTasks(tasks);
      await persistRunState(runDir, tasks, context);
      await sleep(submitDelayMs);
    }

    activeTasks = getActiveTasks(tasks);
    if (activeTasks.length === 0) {
      if (submissionBlockedError && getPendingSubmissionTasks(tasks).length > 0) {
        console.log(
          `\nStopping after active tasks drained because new submissions are blocked: ${submissionBlockedError.message}`,
        );
        break;
      }
      if (getPendingSubmissionTasks(tasks).length === 0) {
        break;
      }
      continue;
    }

    console.log(
      `\nPolling ${activeTasks.length} active task(s); submitted ${countSubmittedTasks(tasks)}/${selectedEntries.length}, completed ${tasks.filter((task) => isFinalStatus(task.task_status)).length}/${selectedEntries.length}`,
    );

    for (const task of activeTasks) {
      try {
        const queryResponse = await queryTask({
          baseUrl: context.base_url,
          apiKey,
          taskId: task.task_id,
        });
        const status = extractTaskStatus(queryResponse);
        task.last_query_response = queryResponse;
        task.last_polled_at = new Date().toISOString();
        task.task_status = status;
        task.poll_error = null;

        if (isSucceededStatus(status)) {
          console.log(`  succeeded [${task.sample_index}] ${task.macro_domain}`);
          task.downloads = await saveGeneratedAssets(task, queryResponse, runDir, selectedEntries.length);
        } else if (isFailedStatus(status)) {
          console.log(`  failed    [${task.sample_index}] ${task.macro_domain}`);
          task.failure_message = extractFailureMessage(queryResponse);
        } else {
          console.log(`  ${String(status ?? 'unknown').padEnd(9)} [${task.sample_index}] ${task.macro_domain}`);
        }
      } catch (error) {
        task.poll_error = {
          message: error instanceof Error ? error.message : String(error),
          status: error?.status ?? null,
          body: error?.body ?? null,
        };
        task.last_polled_at = new Date().toISOString();
        console.log(`  error     [${task.sample_index}] ${task.macro_domain} -> ${task.poll_error.message}`);
      }
    }

    await persistRunState(runDir, tasks, context);

    if (!args.wait && getPendingSubmissionTasks(tasks).length === 0) {
      break;
    }

    if (getActiveTasks(tasks).length > 0) {
      await sleep(context.configured_limits.poll_interval_ms);
    }
  }

  await persistRunState(runDir, tasks, context);

  const summary = buildRunSummary(runDir, tasks, context);
  console.log('\nBatch summary');
  console.log(`  total:      ${summary.totals.tasks}`);
  console.log(`  submitted:  ${summary.totals.submitted}`);
  console.log(`  succeeded:  ${summary.totals.succeeded}`);
  console.log(`  failed:     ${summary.totals.failed}`);
  console.log(`  active:     ${summary.totals.active}`);
  console.log(`  pending:    ${summary.totals.pending_submission}`);
  console.log(`  downloaded: ${summary.totals.downloaded}`);
  console.log(`  summary:    ${path.join(runDir, 'run_summary.json')}`);
}

main().catch((error) => {
  console.error('\nWan 2.7 all-entries batch run failed.');
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
