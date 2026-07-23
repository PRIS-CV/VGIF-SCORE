import fs from 'fs/promises';
import path from 'path';
import { pipeline } from 'stream/promises';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3';
const DEFAULT_MODEL = 'doubao-seedance-2-0-260128';
const DEFAULT_RATIO = '16:9';
const DEFAULT_RESOLUTION = '720p';
const DEFAULT_DURATION = 5;
const DEFAULT_POLL_INTERVAL_MS = 20000;
const DEFAULT_TIMEOUT_MS = 45 * 60 * 1000;
const DEFAULT_MAX_ACTIVE_TASKS = 5;
const DEFAULT_SUBMIT_CONCURRENCY = 2;
const DEFAULT_QUERY_CONCURRENCY = 5;
const DEFAULT_SUBMIT_RETRIES = 5;
const DEFAULT_RETRY_DELAY_MS = 15000;
const DEFAULT_RUN_LABEL = 'all_entries';

const KLING_OUTPUT_ROOT = path.join(__dirname, '..', 'kling_t2v', 'outputs', 'kling_v3_720p_5s');
const ALL_ENTRIES_PATH = path.join(__dirname, '..', 'kling_t2v', 'all_entries_merged_final.json');
const OUTPUT_ROOT = path.join(__dirname, 'outputs', 'seedance_2_0_720p_5s');

function parsePositiveInteger(value, flagName) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`\`${flagName}\` must be a positive integer.`);
  }
  return parsed;
}

function parsePositiveNumber(value, flagName) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`\`${flagName}\` must be a positive number.`);
  }
  return parsed;
}

function parseArgs(argv) {
  const args = {
    wait: true,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
    model: process.env.SEEDANCE_MODEL?.trim() || DEFAULT_MODEL,
    baseUrl: process.env.ARK_BASE_URL?.trim() || DEFAULT_BASE_URL,
    ratio: process.env.SEEDANCE_RATIO?.trim() || DEFAULT_RATIO,
    resolution: process.env.SEEDANCE_RESOLUTION?.trim() || DEFAULT_RESOLUTION,
    duration: Number(process.env.SEEDANCE_DURATION ?? DEFAULT_DURATION),
    maxActiveTasks: parsePositiveInteger(process.env.SEEDANCE_MAX_ACTIVE_TASKS ?? DEFAULT_MAX_ACTIVE_TASKS, 'SEEDANCE_MAX_ACTIVE_TASKS'),
    submitConcurrency: parsePositiveInteger(process.env.SEEDANCE_SUBMIT_CONCURRENCY ?? DEFAULT_SUBMIT_CONCURRENCY, 'SEEDANCE_SUBMIT_CONCURRENCY'),
    queryConcurrency: parsePositiveInteger(process.env.SEEDANCE_QUERY_CONCURRENCY ?? DEFAULT_QUERY_CONCURRENCY, 'SEEDANCE_QUERY_CONCURRENCY'),
    submitRetries: parsePositiveInteger(process.env.SEEDANCE_SUBMIT_RETRIES ?? DEFAULT_SUBMIT_RETRIES, 'SEEDANCE_SUBMIT_RETRIES'),
    retryDelayMs: parsePositiveInteger(process.env.SEEDANCE_RETRY_DELAY_MS ?? DEFAULT_RETRY_DELAY_MS, 'SEEDANCE_RETRY_DELAY_MS'),
    outputRoot: process.env.SEEDANCE_OUTPUT_ROOT?.trim() || OUTPUT_ROOT,
    runLabel: process.env.SEEDANCE_RUN_LABEL?.trim() || DEFAULT_RUN_LABEL,
    retrySubmitFailures: false,
    retryOverdueBalanceFailures: false,
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
    if (token === '--entries-file') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--entries-file` requires a file path.');
      }
      args.entriesFile = value;
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
      const minutes = parsePositiveNumber(argv[i + 1], '--timeout-minutes');
      args.timeoutMs = Math.round(minutes * 60 * 1000);
      i += 1;
      continue;
    }
    if (token === '--retry-submit-failures') {
      args.retrySubmitFailures = true;
      continue;
    }
    if (token === '--retry-overdue-balance-failures') {
      args.retryOverdueBalanceFailures = true;
      continue;
    }
    if (token === '--poll-seconds') {
      const seconds = parsePositiveNumber(argv[i + 1], '--poll-seconds');
      args.pollIntervalMs = Math.round(seconds * 1000);
      i += 1;
      continue;
    }
    if (token === '--limit') {
      args.limit = parsePositiveInteger(argv[i + 1], '--limit');
      i += 1;
      continue;
    }
    if (token === '--start-index') {
      args.startIndex = parsePositiveInteger(argv[i + 1], '--start-index');
      i += 1;
      continue;
    }
    if (token === '--end-index') {
      args.endIndex = parsePositiveInteger(argv[i + 1], '--end-index');
      i += 1;
      continue;
    }
    if (token === '--model') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--model` requires a model id.');
      }
      args.model = value;
      i += 1;
      continue;
    }
    if (token === '--base-url') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--base-url` requires a URL.');
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
      args.duration = parsePositiveNumber(argv[i + 1], '--duration');
      i += 1;
      continue;
    }
    if (token === '--max-active-tasks') {
      args.maxActiveTasks = parsePositiveInteger(argv[i + 1], '--max-active-tasks');
      i += 1;
      continue;
    }
    if (token === '--submit-concurrency') {
      args.submitConcurrency = parsePositiveInteger(argv[i + 1], '--submit-concurrency');
      i += 1;
      continue;
    }
    if (token === '--query-concurrency') {
      args.queryConcurrency = parsePositiveInteger(argv[i + 1], '--query-concurrency');
      i += 1;
      continue;
    }
    if (token === '--submit-retries') {
      args.submitRetries = parsePositiveInteger(argv[i + 1], '--submit-retries');
      i += 1;
      continue;
    }
    if (token === '--retry-delay-ms') {
      args.retryDelayMs = parsePositiveInteger(argv[i + 1], '--retry-delay-ms');
      i += 1;
      continue;
    }
    if (token === '--output-root') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--output-root` requires a directory path.');
      }
      args.outputRoot = value;
      i += 1;
      continue;
    }
    if (token === '--run-label') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('`--run-label` requires a value.');
      }
      args.runLabel = value;
      i += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  if (!Number.isFinite(args.duration) || args.duration <= 0) {
    throw new Error('Duration must be a positive number.');
  }

  if (args.startIndex && args.endIndex && args.startIndex > args.endIndex) {
    throw new Error('`--start-index` cannot be greater than `--end-index`.');
  }

  if (args.submitConcurrency > args.maxActiveTasks) {
    args.submitConcurrency = args.maxActiveTasks;
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
  const normalized = String(text ?? '')
    .normalize('NFKD')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();

  return normalized.slice(0, maxLength) || 'run';
}

function previewPrompt(prompt, maxLength = 120) {
  if (!prompt) {
    return '';
  }
  return prompt.length <= maxLength ? prompt : `${prompt.slice(0, maxLength - 3)}...`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function padSampleIndex(sampleIndex, totalCount) {
  const width = Math.max(2, String(totalCount).length);
  return String(sampleIndex).padStart(width, '0');
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
      // Keep scanning.
    }
  }

  throw new Error(`Unable to find any selected_prompts.json under ${KLING_OUTPUT_ROOT}`);
}

function makeHeaders(apiKey) {
  return {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };
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
        : body?.error?.message ?? body?.message ?? JSON.stringify(body);
    const error = new Error(`HTTP ${response.status} for ${url}: ${message}`);
    error.status = response.status;
    error.body = body;
    throw error;
  }

  return body;
}

function isRetriableStatus(status) {
  return status === 408 || status === 409 || status === 425 || status === 429 || status >= 500;
}

function isRetriableError(error) {
  const status = error?.status ?? null;
  if (status && isRetriableStatus(status)) {
    return true;
  }
  return /fetch failed|network|timeout|ECONNRESET|ETIMEDOUT|socket hang up/i.test(String(error?.message ?? ''));
}

function serializeError(error) {
  return {
    message: error instanceof Error ? error.message : String(error),
    status: error?.status ?? null,
    body: error?.body ?? null,
    stack: error instanceof Error ? error.stack ?? null : null,
    at: new Date().toISOString(),
  };
}

async function withRetries(operation, { retries, delayMs, label }) {
  let attempt = 0;
  let lastError = null;

  while (attempt < retries) {
    attempt += 1;
    try {
      return await operation(attempt);
    } catch (error) {
      lastError = error;
      const canRetry = attempt < retries && isRetriableError(error);
      if (!canRetry) {
        break;
      }
      const waitMs = delayMs * attempt;
      console.log(`  retry     ${label} -> ${error.message} (attempt ${attempt}/${retries}, wait ${waitMs}ms)`);
      await sleep(waitMs);
    }
  }

  throw lastError;
}

async function submitTask({ baseUrl, apiKey, payload }) {
  return fetchJson(`${baseUrl}/contents/generations/tasks`, {
    method: 'POST',
    headers: makeHeaders(apiKey),
    body: JSON.stringify(payload),
  });
}

async function queryTask({ baseUrl, apiKey, taskId }) {
  return fetchJson(`${baseUrl}/contents/generations/tasks/${taskId}`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
  });
}

function extractTaskId(response) {
  return response?.id ?? response?.data?.id ?? null;
}

function extractTaskStatus(response) {
  return response?.status ?? response?.data?.status ?? null;
}

function isSucceededStatus(status) {
  return status === 'succeeded' || status === 'succeed';
}

function isFailedStatus(status) {
  return status === 'failed' || status === 'cancelled' || status === 'expired';
}

function extractFailureMessage(response) {
  return (
    response?.error?.message ??
    response?.data?.error?.message ??
    response?.message ??
    response?.data?.message ??
    'Unknown error'
  );
}

function extractVideoOutputs(response) {
  const candidates = [];

  const videoUrl = response?.content?.video_url ?? response?.data?.content?.video_url;
  if (videoUrl) {
    candidates.push({
      url: videoUrl,
      last_frame_url: response?.content?.last_frame_url ?? response?.data?.content?.last_frame_url ?? null,
      id: response?.content?.id ?? response?.data?.content?.id ?? null,
      duration: response?.content?.duration ?? response?.data?.content?.duration ?? null,
    });
  }

  const videoList =
    response?.content?.videos ??
    response?.data?.content?.videos ??
    response?.task_result?.videos ??
    response?.data?.task_result?.videos ??
    [];

  if (Array.isArray(videoList)) {
    for (const item of videoList) {
      if (item?.url) {
        candidates.push({
          url: item.url,
          last_frame_url: item.last_frame_url ?? null,
          id: item.id ?? null,
          duration: item.duration ?? null,
        });
      }
    }
  }

  return candidates;
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
  };
}

async function saveGeneratedAssets(task, queryResponse, runDir, totalCount) {
  if (Array.isArray(task.downloads) && task.downloads.length > 0) {
    return task.downloads;
  }

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
    const fileBase = `${padSampleIndex(task.sample_index, totalCount)}_${slugify(task.macro_domain, 48)}_${task.task_id}_${i + 1}`;
    const videoPath = path.join(paths.videosDir, `${fileBase}.mp4`);
    const metadataPath = path.join(paths.metadataDir, `${fileBase}.json`);

    try {
      await fs.access(videoPath);
    } catch {
      await downloadFile(output.url, videoPath);
    }

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

function buildSubmissionPayload(args, prompt, externalTaskId) {
  return {
    model: args.model,
    content: [
      {
        type: 'text',
        text: prompt,
      },
    ],
    resolution: args.resolution,
    ratio: args.ratio,
    duration: args.duration,
    watermark: false,
    external_task_id: externalTaskId,
  };
}

function mapAllEntriesToSamples(entries) {
  return entries.map((entry, index) => ({
    sample_index: index + 1,
    macro_domain: entry?.domain_info?.macro_domain ?? 'Unknown Macro Domain',
    micro_domain: entry?.domain_info?.micro_domain ?? 'Unknown Micro Domain',
    prompt: entry?.prompt ?? '',
    complexity: entry?.complexity ?? null,
    qa_pair_count: Array.isArray(entry?.vlm_qa_pairs) ? entry.vlm_qa_pairs.length : null,
    source_kind: 'all_entries_merged_final',
  }));
}

function applyEntryFilters(entries, args) {
  let filtered = [...entries];

  if (args.startIndex) {
    filtered = filtered.filter((entry) => entry.sample_index >= args.startIndex);
  }
  if (args.endIndex) {
    filtered = filtered.filter((entry) => entry.sample_index <= args.endIndex);
  }
  if (args.limit) {
    filtered = filtered.slice(0, args.limit);
  }

  return filtered;
}

async function loadSelectedEntries(args) {
  if (args.selectedPromptsPath) {
    const selectedPromptsPath = path.resolve(args.selectedPromptsPath);
    const selectedPrompts = await readJson(selectedPromptsPath);
    const availableEntries = selectedPrompts.selected_prompts ?? [];
    return {
      sourcePath: selectedPromptsPath,
      sourceKind: 'selected_prompts',
      selectedEntries: applyEntryFilters(availableEntries, args),
    };
  }

  const entriesPath = path.resolve(args.entriesFile ?? ALL_ENTRIES_PATH);
  const allEntries = await readJson(entriesPath);
  const mappedEntries = mapAllEntriesToSamples(allEntries);

  return {
    sourcePath: entriesPath,
    sourceKind: 'all_entries',
    selectedEntries: applyEntryFilters(mappedEntries, args),
  };
}

function summarizeSelection(selectedEntries) {
  return selectedEntries.map((entry) => ({
    sample_index: entry.sample_index,
    macro_domain: entry.macro_domain,
    micro_domain: entry.micro_domain,
    prompt_preview: previewPrompt(entry.prompt, 180),
  }));
}

function findTask(tasks, sampleIndex) {
  return tasks.find((task) => task.sample_index === sampleIndex) ?? null;
}

function upsertTask(tasks, sample, patch) {
  const existing = findTask(tasks, sample.sample_index);
  const base = {
    sample_index: sample.sample_index,
    macro_domain: sample.macro_domain,
    micro_domain: sample.micro_domain,
    prompt: sample.prompt,
    complexity: sample.complexity ?? null,
    qa_pair_count: sample.qa_pair_count ?? null,
    source_kind: sample.source_kind ?? null,
  };

  if (existing) {
    Object.assign(existing, patch);
    return existing;
  }

  const created = { ...base, ...patch };
  tasks.push(created);
  tasks.sort((left, right) => left.sample_index - right.sample_index);
  return created;
}

function hasDownloads(task) {
  return Array.isArray(task.downloads) && task.downloads.length > 0;
}

function isOverdueBalanceMessage(message) {
  return /overdue balance/i.test(String(message ?? ''));
}

function isTaskTerminal(task) {
  if (!task) {
    return false;
  }
  if (isFailedStatus(task.task_status)) {
    return true;
  }
  if (task.submission_state === 'submit_failed_terminal') {
    return true;
  }
  return isSucceededStatus(task.task_status) && hasDownloads(task);
}

function isTaskActive(task) {
  if (!task?.task_id) {
    return false;
  }
  return !isTaskTerminal(task);
}

function buildPendingSampleList(selectedEntries, tasks, args) {
  return selectedEntries.filter((sample) => {
    const task = findTask(tasks, sample.sample_index);
    if (!task) {
      return true;
    }
    if (
      args.retryOverdueBalanceFailures &&
      isFailedStatus(task.task_status) &&
      isOverdueBalanceMessage(task.failure_message)
    ) {
      return true;
    }
    if (task.task_id) {
      return false;
    }
    if (task.submission_state === 'submit_failed_terminal') {
      return args.retrySubmitFailures;
    }
    return true;
  });
}

function buildRuntimeSummary(selectedEntries, tasks, args, runDir, extra = {}) {
  const activeTasks = tasks.filter(isTaskActive);
  const succeeded = tasks.filter((task) => isSucceededStatus(task.task_status) && hasDownloads(task)).length;
  const generationSucceededNotDownloaded = tasks.filter(
    (task) => isSucceededStatus(task.task_status) && !hasDownloads(task),
  ).length;
  const failed = tasks.filter((task) => isFailedStatus(task.task_status)).length;
  const submitFailed = tasks.filter((task) => task.submission_state === 'submit_failed_terminal').length;
  const submitted = tasks.filter((task) => !!task.task_id).length;
  const notYetSubmitted = buildPendingSampleList(selectedEntries, tasks, args).length;

  return {
    updated_at: new Date().toISOString(),
    run_dir: runDir,
    total_selected: selectedEntries.length,
    submitted,
    succeeded,
    generation_succeeded_not_downloaded: generationSucceededNotDownloaded,
    failed,
    submit_failed: submitFailed,
    active: activeTasks.length,
    not_yet_submitted: notYetSubmitted,
    pending_task_ids: activeTasks.map((task) => task.task_id),
    config: {
      model: args.model,
      resolution: args.resolution,
      ratio: args.ratio,
      duration: args.duration,
      max_active_tasks: args.maxActiveTasks,
      submit_concurrency: args.submitConcurrency,
      query_concurrency: args.queryConcurrency,
      poll_interval_ms: args.pollIntervalMs,
    },
    tasks,
    ...extra,
  };
}

async function persistRunState(runDir, context, selectedEntries, tasks, args, extra = {}) {
  await writeJson(path.join(runDir, 'submitted_tasks.json'), {
    updated_at: new Date().toISOString(),
    context,
    tasks,
  });

  await writeJson(path.join(runDir, 'run_summary.json'), buildRuntimeSummary(selectedEntries, tasks, args, runDir, extra));
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

async function runWithConcurrency(items, concurrency, worker) {
  if (!items.length) {
    return;
  }

  let cursor = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (true) {
      const currentIndex = cursor;
      cursor += 1;
      if (currentIndex >= items.length) {
        return;
      }
      await worker(items[currentIndex], currentIndex);
    }
  });

  await Promise.all(runners);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = getRequiredEnv('ARK_API_KEY');

  let runDir;
  let selectedEntries;
  let tasks;
  let sourcePath;
  let sourceKind;

  if (args.resumeRunDir) {
    runDir = path.resolve(args.resumeRunDir);
    const resumeState = await loadResumeState(runDir);
    selectedEntries = resumeState.selectedEntries;
    tasks = resumeState.tasks;
    sourcePath = resumeState.config.source_path ?? resumeState.config.selected_prompts_path ?? null;
    sourceKind = resumeState.config.source_kind ?? 'unknown';

    if (!selectedEntries.length) {
      throw new Error(`No selected prompts found in resume directory: ${runDir}`);
    }

    console.log(`Resuming existing run: ${runDir}`);
  } else {
    const loadResult = await loadSelectedEntries(args);
    sourcePath = loadResult.sourcePath;
    sourceKind = loadResult.sourceKind;
    selectedEntries = loadResult.selectedEntries;

    if (!selectedEntries.length) {
      throw new Error(`No prompts found in ${sourcePath}`);
    }

    runDir = path.join(path.resolve(args.outputRoot), `${nowStamp()}_${slugify(args.runLabel, 32)}`);
    await ensureDir(runDir);

    const runConfig = {
      created_at: new Date().toISOString(),
      base_url: args.baseUrl,
      model: args.model,
      ratio: args.ratio,
      resolution: args.resolution,
      duration: args.duration,
      source_kind: sourceKind,
      source_path: sourcePath,
      selected_count: selectedEntries.length,
      start_index: args.startIndex ?? null,
      end_index: args.endIndex ?? null,
      limit: args.limit ?? null,
      max_active_tasks: args.maxActiveTasks,
      submit_concurrency: args.submitConcurrency,
      query_concurrency: args.queryConcurrency,
      submit_retries: args.submitRetries,
      retry_delay_ms: args.retryDelayMs,
      official_docs: {
        create_task: 'https://www.volcengine.com/docs/82379/1520757?lang=zh',
        query_task: 'https://www.volcengine.com/docs/82379/1521309?lang=zh',
        notes: [
          '官方文档明确给出创建接口 /contents/generations/tasks 与查询接口 /contents/generations/tasks/{id}。',
          '官方文档明确给出任务状态 queued/running/cancelled/succeeded/failed/expired。',
          '官方文档明确给出生成结果中的 content.video_url 有效期为 24 小时，因此脚本在任务成功后立即下载并持久化。',
          '在已核对的上述页面中，没有检索到 Seedance 2.0 单次视频生成任务的明确数值型并发上限，因此本次采用保守的可配置活动任务窗口。',
        ],
      },
    };

    await writeJson(path.join(runDir, 'run_config.json'), runConfig);
    await writeJson(path.join(runDir, 'selected_prompts.json'), {
      created_at: new Date().toISOString(),
      source_kind: sourceKind,
      source_path: sourcePath,
      count: selectedEntries.length,
      selected_prompts: selectedEntries,
    });

    tasks = [];

    console.log(`Selected ${selectedEntries.length} prompts.`);
    console.log(`Model: ${args.model}`);
    console.log(`Base URL: ${args.baseUrl}`);
    console.log(`Resolution / duration / ratio: ${args.resolution} / ${args.duration}s / ${args.ratio}`);
    console.log(`Max active tasks: ${args.maxActiveTasks}`);
    console.log(`Source prompts: ${sourcePath}`);
    for (const item of summarizeSelection(selectedEntries.slice(0, 12))) {
      console.log(`[${item.sample_index}] ${item.macro_domain}`);
      console.log(`    ${item.micro_domain}`);
      console.log(`    ${item.prompt_preview}`);
    }
    if (selectedEntries.length > 12) {
      console.log(`... (${selectedEntries.length - 12} more prompts omitted from console preview)`);
    }
  }

  const context = {
    base_url: args.baseUrl,
    model: args.model,
    ratio: args.ratio,
    resolution: args.resolution,
    duration: args.duration,
    source_kind: sourceKind,
    source_path: sourcePath,
  };

  let persistQueue = Promise.resolve();
  const queuePersist = (extra = {}) => {
    persistQueue = persistQueue.then(() => persistRunState(runDir, context, selectedEntries, tasks, args, extra));
    return persistQueue;
  };

  await queuePersist({ wait_mode: args.wait });

  const startedAt = Date.now();
  let timedOut = false;

  while (true) {
    if (Date.now() - startedAt > args.timeoutMs) {
      timedOut = true;
      break;
    }

    const activeTasks = tasks.filter(isTaskActive);
    const pendingSamples = buildPendingSampleList(selectedEntries, tasks, args);
    const availableSlots = args.wait ? Math.max(0, args.maxActiveTasks - activeTasks.length) : pendingSamples.length;

    if (availableSlots > 0 && pendingSamples.length > 0) {
      const samplesToSubmit = pendingSamples.slice(0, availableSlots);
      console.log(`\nSubmitting up to ${samplesToSubmit.length} task(s). Active window: ${activeTasks.length}/${args.maxActiveTasks}`);

      await runWithConcurrency(samplesToSubmit, Math.min(args.submitConcurrency, samplesToSubmit.length), async (sample) => {
        const externalTaskId = `batch-${path.basename(runDir)}-${padSampleIndex(sample.sample_index, selectedEntries.length)}-${slugify(sample.macro_domain, 24)}`;
        const payload = buildSubmissionPayload(args, sample.prompt, externalTaskId);

        upsertTask(tasks, sample, {
          submission_state: 'submitting',
          submission_attempts: (findTask(tasks, sample.sample_index)?.submission_attempts ?? 0) + 1,
          last_submission_attempt_at: new Date().toISOString(),
          submission_payload: payload,
          task_id: null,
          task_status: null,
          failure_message: null,
          submission_response: null,
          submission_error: null,
          last_query_response: null,
          last_polled_at: null,
          poll_error: null,
          download_error: null,
        });
        await queuePersist();

        console.log(`Submitting [${sample.sample_index}] ${sample.macro_domain}`);

        try {
          const response = await withRetries(
            () => submitTask({ baseUrl: args.baseUrl, apiKey, payload }),
            {
              retries: args.submitRetries,
              delayMs: args.retryDelayMs,
              label: `[${sample.sample_index}] ${sample.macro_domain}`,
            },
          );

          const taskId = extractTaskId(response);
          if (!taskId) {
            throw new Error(`Submission returned no task id: ${JSON.stringify(response, null, 2)}`);
          }

          upsertTask(tasks, sample, {
            task_id: taskId,
            submission_payload: payload,
            submission_response: response,
            submission_state: 'submitted',
            submission_error: null,
            task_status: 'queued',
            last_submitted_at: new Date().toISOString(),
          });
          console.log(`  accepted  [${sample.sample_index}] ${sample.macro_domain} -> ${taskId}`);
        } catch (error) {
          upsertTask(tasks, sample, {
            submission_state: 'submit_failed_terminal',
            submission_error: serializeError(error),
            task_status: 'submit_failed',
          });
          console.log(`  rejected  [${sample.sample_index}] ${sample.macro_domain} -> ${error.message}`);
        }

        await queuePersist();
      });

      await persistQueue;
    }

    if (!args.wait) {
      const stillPendingToSubmit = buildPendingSampleList(selectedEntries, tasks, args);
      if (!stillPendingToSubmit.length) {
        break;
      }
      continue;
    }

    const currentActiveTasks = tasks.filter(isTaskActive);
    if (!currentActiveTasks.length) {
      const remainingSamples = buildPendingSampleList(selectedEntries, tasks, args);
      if (!remainingSamples.length) {
        break;
      }
      await sleep(Math.min(args.pollIntervalMs, 5000));
      continue;
    }

    console.log(`\nPolling ${currentActiveTasks.length} active task(s)...`);
    await runWithConcurrency(currentActiveTasks, Math.min(args.queryConcurrency, currentActiveTasks.length), async (task) => {
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
        task.poll_error = null;

        if (isSucceededStatus(status)) {
          if (!hasDownloads(task)) {
            try {
              task.downloads = await saveGeneratedAssets(task, queryResponse, runDir, selectedEntries.length);
              task.download_error = null;
              console.log(`  downloaded [${task.sample_index}] ${task.macro_domain}`);
            } catch (error) {
              task.download_error = serializeError(error);
              console.log(`  dl-error  [${task.sample_index}] ${task.macro_domain} -> ${error.message}`);
            }
          } else {
            console.log(`  succeeded [${task.sample_index}] ${task.macro_domain}`);
          }
        } else if (isFailedStatus(status)) {
          task.failure_message = extractFailureMessage(queryResponse);
          console.log(`  failed    [${task.sample_index}] ${task.macro_domain}`);
        } else {
          console.log(`  ${String(status ?? 'unknown').padEnd(9)} [${task.sample_index}] ${task.macro_domain}`);
        }
      } catch (error) {
        task.poll_error = serializeError(error);
        task.last_polled_at = new Date().toISOString();
        console.log(`  error     [${task.sample_index}] ${task.macro_domain} -> ${task.poll_error.message}`);
      }

      await queuePersist();
    });

    await persistQueue;

    const remainingSamples = buildPendingSampleList(selectedEntries, tasks, args);
    const stillActive = tasks.filter(isTaskActive);
    if (!remainingSamples.length && !stillActive.length) {
      break;
    }

    await sleep(args.pollIntervalMs);
  }

  await queuePersist({ wait_mode: args.wait, timed_out: timedOut });
  await persistQueue;

  const summary = buildRuntimeSummary(selectedEntries, tasks, args, runDir, {
    wait_mode: args.wait,
    timed_out: timedOut,
  });

  console.log('\nBatch summary');
  console.log(`  selected:           ${summary.total_selected}`);
  console.log(`  submitted:          ${summary.submitted}`);
  console.log(`  succeeded:          ${summary.succeeded}`);
  console.log(`  success-not-saved:  ${summary.generation_succeeded_not_downloaded}`);
  console.log(`  failed:             ${summary.failed}`);
  console.log(`  submit-failed:      ${summary.submit_failed}`);
  console.log(`  active:             ${summary.active}`);
  console.log(`  not-yet-submitted:  ${summary.not_yet_submitted}`);
  console.log(`  summary:            ${path.join(runDir, 'run_summary.json')}`);
}

main().catch((error) => {
  console.error('\nSeedance 2.0 batch run failed.');
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
