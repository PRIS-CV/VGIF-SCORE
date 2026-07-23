import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';
import jwt from 'jsonwebtoken';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_BASE_URL = 'https://api-beijing.klingai.com';
const INPUT_JSON = path.join(__dirname, 'all_entries_merged_final.json');
const OUTPUT_ROOT = path.join(__dirname, 'outputs', 'kling_v3_720p_5s');
const POLL_INTERVAL_MS = 20000;
const DEFAULT_TIMEOUT_MS = 45 * 60 * 1000;
const DEFAULT_MAX_ACTIVE_TASKS = 3;
const DEFAULT_SUBMIT_CONCURRENCY = 2;
const DEFAULT_QUERY_CONCURRENCY = 3;
const TOKEN_VALIDITY_SECONDS = 1800;
const CLOCK_SKEW_SECONDS = 5;
const DEFAULT_SUBMIT_RETRY_DELAY_MS = 30000;
const DEFAULT_SUBMIT_MAX_RETRIES = 8;

function parseArgs(argv) {
  const args = {
    wait: true,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    inputJson: INPUT_JSON,
    outputRoot: OUTPUT_ROOT,
    pollIntervalMs: POLL_INTERVAL_MS,
    maxActiveTasks: DEFAULT_MAX_ACTIVE_TASKS,
    submitConcurrency: DEFAULT_SUBMIT_CONCURRENCY,
    queryConcurrency: DEFAULT_QUERY_CONCURRENCY,
    representativeOnly: false,
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
    if (token === '--timeout-minutes') {
      const minutes = Number(argv[i + 1]);
      if (!Number.isFinite(minutes) || minutes <= 0) {
        throw new Error('`--timeout-minutes` must be a positive number.');
      }
      args.timeoutMs = Math.round(minutes * 60 * 1000);
      i += 1;
      continue;
    }
    if (token === '--limit') {
      const limit = Number(argv[i + 1]);
      if (!Number.isInteger(limit) || limit <= 0) {
        throw new Error('`--limit` must be a positive integer.');
      }
      args.limit = limit;
      i += 1;
      continue;
    }
    if (token === '--entries-file') {
      const inputJson = argv[i + 1];
      if (!inputJson) {
        throw new Error('`--entries-file` requires a file path.');
      }
      args.inputJson = inputJson;
      i += 1;
      continue;
    }
    if (token === '--output-root') {
      const outputRoot = argv[i + 1];
      if (!outputRoot) {
        throw new Error('`--output-root` requires a directory path.');
      }
      args.outputRoot = outputRoot;
      i += 1;
      continue;
    }
    if (token === '--run-label') {
      const runLabel = argv[i + 1];
      if (!runLabel) {
        throw new Error('`--run-label` requires a label.');
      }
      args.runLabel = runLabel;
      i += 1;
      continue;
    }
    if (token === '--poll-seconds') {
      const seconds = Number(argv[i + 1]);
      if (!Number.isFinite(seconds) || seconds <= 0) {
        throw new Error('`--poll-seconds` must be a positive number.');
      }
      args.pollIntervalMs = Math.round(seconds * 1000);
      i += 1;
      continue;
    }
    if (token === '--max-active-tasks') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--max-active-tasks` must be a positive integer.');
      }
      args.maxActiveTasks = value;
      i += 1;
      continue;
    }
    if (token === '--submit-concurrency') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--submit-concurrency` must be a positive integer.');
      }
      args.submitConcurrency = value;
      i += 1;
      continue;
    }
    if (token === '--query-concurrency') {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error('`--query-concurrency` must be a positive integer.');
      }
      args.queryConcurrency = value;
      i += 1;
      continue;
    }
    if (token === '--representative-only') {
      args.representativeOnly = true;
      continue;
    }
    if (token === '--resume-run-dir') {
      const runDir = argv[i + 1];
      if (!runDir) {
        throw new Error('`--resume-run-dir` requires a directory path.');
      }
      args.resumeRunDir = runDir;
      i += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }

  if (args.submitConcurrency > args.maxActiveTasks) {
    args.submitConcurrency = args.maxActiveTasks;
  }
  if (args.queryConcurrency > args.maxActiveTasks) {
    args.queryConcurrency = args.maxActiveTasks;
  }

  return args;
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
  return text
    .normalize('NFKD')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, maxLength)
    .toLowerCase();
}

function getRequiredEnv(name) {
  const value = process.env[name];
  if (!value || !value.trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value.trim();
}

function getBaseUrl() {
  const value = process.env.KLING_BASE_URL?.trim();
  return value || DEFAULT_BASE_URL;
}

function buildToken(accessKey, secretKey) {
  const now = Math.floor(Date.now() / 1000);
  return jwt.sign(
    {
      iss: accessKey,
      exp: now + TOKEN_VALIDITY_SECONDS,
      nbf: now - CLOCK_SKEW_SECONDS,
    },
    secretKey,
    {
      algorithm: 'HS256',
      noTimestamp: true,
      header: {
        alg: 'HS256',
        typ: 'JWT',
      },
    },
  );
}

function createApiClient(accessKey, secretKey) {
  const client = axios.create({
    baseURL: getBaseUrl(),
    timeout: 120000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  client.interceptors.request.use((config) => {
    const headers = config.headers ?? {};
    headers.Authorization = `Bearer ${buildToken(accessKey, secretKey)}`;
    config.headers = headers;
    return config;
  });

  return client;
}

async function readJson(filePath) {
  const content = await fs.readFile(filePath, 'utf8');
  return JSON.parse(content);
}

function rankEntry(entry) {
  const complexityRank = {
    low: 0,
    medium: 1,
    high: 2,
  };

  return [
    complexityRank[entry.complexity?.level] ?? 99,
    entry.complexity?.num_entities ?? 999,
    entry.complexity?.num_primary_actions ?? 999,
    entry.complexity?.num_state_changes ?? 999,
    entry.prompt?.length ?? 9999,
  ];
}

function compareRanks(a, b) {
  const rankA = rankEntry(a);
  const rankB = rankEntry(b);
  for (let i = 0; i < rankA.length; i += 1) {
    if (rankA[i] !== rankB[i]) {
      return rankA[i] - rankB[i];
    }
  }
  return 0;
}

function selectRepresentativeEntries(entries, limit) {
  const groups = new Map();

  for (const entry of entries) {
    const macroDomain = entry?.domain_info?.macro_domain;
    if (!macroDomain) {
      continue;
    }
    if (!groups.has(macroDomain)) {
      groups.set(macroDomain, []);
    }
    groups.get(macroDomain).push(entry);
  }

  const selected = [...groups.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([macroDomain, groupEntries], index) => {
      const best = [...groupEntries].sort(compareRanks)[0];
      return {
        sample_index: index + 1,
        macro_domain: macroDomain,
        micro_domain: best.domain_info?.micro_domain ?? '',
        prompt: best.prompt,
        complexity: best.complexity ?? {},
      };
    });

  return limit ? selected.slice(0, limit) : selected;
}

function selectAllEntries(entries, limit) {
  const normalized = entries.map((entry, index) => ({
    sample_index: index + 1,
    macro_domain: entry?.domain_info?.macro_domain ?? 'Unknown Macro Domain',
    micro_domain: entry?.domain_info?.micro_domain ?? '',
    prompt: entry?.prompt ?? '',
    complexity: entry?.complexity ?? {},
    qa_pair_count: Array.isArray(entry?.qa_pairs) ? entry.qa_pairs.length : null,
    source_kind: 'all_entries_merged_final',
  }));

  return limit ? normalized.slice(0, limit) : normalized;
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

async function writeJson(filePath, data) {
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

function isRetryableSubmissionError(error) {
  if (!axios.isAxiosError(error)) {
    return false;
  }

  const status = error.response?.status;
  return status === 429 || status === 500 || status === 503 || status === 504;
}

function buildProbeVariants() {
  const modelNames = ['kling-v3', 'kling-v3-master'];
  const variants = [];

  for (const modelName of modelNames) {
    variants.push({
      label: `${modelName} + resolution + sound + string duration`,
      basePayload: {
        model_name: modelName,
        duration: '5',
        resolution: '720p',
        sound: 'off',
      },
    });
    variants.push({
      label: `${modelName} + resolution + string duration`,
      basePayload: {
        model_name: modelName,
        duration: '5',
        resolution: '720p',
      },
    });
    variants.push({
      label: `${modelName} + video_resolution + sound + string duration`,
      basePayload: {
        model_name: modelName,
        duration: '5',
        video_resolution: '720p',
        sound: 'off',
      },
    });
    variants.push({
      label: `${modelName} + sound + string duration`,
      basePayload: {
        model_name: modelName,
        duration: '5',
        sound: 'off',
      },
    });
    variants.push({
      label: `${modelName} + resolution + numeric duration`,
      basePayload: {
        model_name: modelName,
        duration: 5,
        resolution: '720p',
      },
    });
    variants.push({
      label: `${modelName} + numeric duration`,
      basePayload: {
        model_name: modelName,
        duration: 5,
      },
    });
    variants.push({
      label: `${modelName} + string duration`,
      basePayload: {
        model_name: modelName,
        duration: '5',
      },
    });
  }

  return variants;
}

async function submitTextToVideoTask(client, payload) {
  const response = await client.post('/v1/videos/text2video', payload);
  return response.data;
}

async function submitTextToVideoTaskWithRetries(client, payload, sampleLabel) {
  let lastError = null;

  for (let attempt = 1; attempt <= DEFAULT_SUBMIT_MAX_RETRIES; attempt += 1) {
    try {
      return await submitTextToVideoTask(client, payload);
    } catch (error) {
      lastError = error;
      if (!isRetryableSubmissionError(error) || attempt === DEFAULT_SUBMIT_MAX_RETRIES) {
        throw error;
      }

      const delayMs = DEFAULT_SUBMIT_RETRY_DELAY_MS * attempt;
      const described = describeAxiosError(error);
      console.log(
        `  retrying submission for ${sampleLabel} after ${Math.round(delayMs / 1000)}s because of ${described.status ?? 'error'} ${described.data?.code ?? ''}`.trim(),
      );
      await sleep(delayMs);
    }
  }

  throw lastError ?? new Error(`Submission failed for ${sampleLabel}`);
}

async function queryTextToVideoTask(client, taskId) {
  const response = await client.get(`/v1/videos/text2video/${taskId}`);
  return response.data;
}

function extractTaskId(response) {
  return response?.data?.task_id ?? response?.task_id ?? null;
}

function extractTaskStatus(response) {
  return response?.data?.task_status ?? response?.task_status ?? null;
}

function extractTaskResult(response) {
  return response?.data?.task_result ?? response?.task_result ?? null;
}

function describeAxiosError(error) {
  if (!axios.isAxiosError(error)) {
    return {
      message: error instanceof Error ? error.message : String(error),
    };
  }

  return {
    message: error.message,
    status: error.response?.status ?? null,
    data: error.response?.data ?? null,
  };
}

async function probeWorkingPayload(client, firstSample, runDir) {
  const variants = buildProbeVariants();
  const attempts = [];

  for (const variant of variants) {
    const externalTaskId = `probe-${nowStamp()}-${slugify(firstSample.macro_domain, 24)}`;
    const payload = {
      ...variant.basePayload,
      prompt: firstSample.prompt,
      external_task_id: externalTaskId,
    };

    try {
      const response = await submitTextToVideoTask(client, payload);
      const taskId = extractTaskId(response);
      if (!taskId) {
        throw new Error(`Probe succeeded without task_id. Raw response: ${JSON.stringify(response)}`);
      }

      await writeJson(path.join(runDir, 'probe_success.json'), {
        variant,
        payload,
        response,
      });

      return {
        variant,
        payload,
        response,
        taskId,
      };
    } catch (error) {
      const described = describeAxiosError(error);
      attempts.push({
        variant: variant.label,
        payload,
        error: described,
      });
    }
  }

  await writeJson(path.join(runDir, 'probe_failures.json'), attempts);
  const last = attempts.at(-1);
  throw new Error(
    `Unable to find a working Kling v3 payload variant. Last failure: ${JSON.stringify(last, null, 2)}`,
  );
}

function buildOutputPaths(runDir) {
  return {
    videosDir: path.join(runDir, 'videos'),
    metadataDir: path.join(runDir, 'metadata'),
  };
}

async function downloadFile(url, destination) {
  const response = await axios.get(url, {
    responseType: 'stream',
    timeout: 300000,
  });

  await ensureDir(path.dirname(destination));

  const { createWriteStream } = await import('fs');
  await new Promise((resolve, reject) => {
    const stream = response.data;
    const writer = createWriteStream(destination);
    stream.pipe(writer);
    writer.on('finish', resolve);
    writer.on('error', reject);
    stream.on('error', reject);
  });
}

async function saveTaskVideos(task, queryResponse, runDir) {
  const taskResult = extractTaskResult(queryResponse);
  const videos = taskResult?.videos ?? [];
  const outputPaths = buildOutputPaths(runDir);

  await ensureDir(outputPaths.videosDir);
  await ensureDir(outputPaths.metadataDir);

  const downloaded = [];
  for (let i = 0; i < videos.length; i += 1) {
    const video = videos[i];
    const fileBase = `${String(task.sample_index).padStart(2, '0')}_${slugify(task.macro_domain, 48)}_${task.task_id}_${i + 1}`;
    const videoPath = path.join(outputPaths.videosDir, `${fileBase}.mp4`);
    const metadataPath = path.join(outputPaths.metadataDir, `${fileBase}.json`);

    await downloadFile(video.url, videoPath);
    await writeJson(metadataPath, {
      task,
      queryResponse,
      downloaded_at: new Date().toISOString(),
      video_index: i + 1,
    });

    downloaded.push({
      url: video.url,
      video_path: videoPath,
      metadata_path: metadataPath,
    });
  }

  return downloaded;
}

function summarizeSelection(selectedEntries) {
  return selectedEntries.map((entry) => ({
    sample_index: entry.sample_index,
    macro_domain: entry.macro_domain,
    micro_domain: entry.micro_domain,
    complexity: entry.complexity,
    prompt_preview: previewPrompt(entry.prompt, 180),
  }));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadResumeState(runDir) {
  const selectedPromptsPath = path.join(runDir, 'selected_prompts.json');
  const submittedTasksPath = path.join(runDir, 'submitted_tasks.json');
  const selectedPrompts = await readJson(selectedPromptsPath);
  const submittedTasks = await readJson(submittedTasksPath);

  return {
    selectedEntries: selectedPrompts.selected_prompts ?? [],
    reusableBasePayload: submittedTasks.reusable_base_payload ?? null,
    tasks: submittedTasks.tasks ?? [],
  };
}

async function persistSubmittedTasks(runDir, reusableBasePayload, tasks) {
  await writeJson(path.join(runDir, 'submitted_tasks.json'), {
    created_at: new Date().toISOString(),
    reusable_base_payload: reusableBasePayload,
    tasks,
  });
}

function summarizeTasks(tasks) {
  const submitted = tasks.filter((task) => task.submission_state === 'submitted').length;
  const succeeded = tasks.filter((task) => task.task_status === 'succeed').length;
  const failed = tasks.filter((task) => task.task_status === 'failed').length;
  const active = tasks.filter((task) => task.submission_state === 'submitted' && !['succeed', 'failed'].includes(task.task_status)).length;

  return {
    submitted,
    succeeded,
    failed,
    active,
  };
}

async function persistRunSummary(runDir, selectedEntries, reusableBasePayload, tasks, pendingTaskIds, args) {
  const counts = summarizeTasks(tasks);
  await writeJson(path.join(runDir, 'run_summary.json'), {
    updated_at: new Date().toISOString(),
    run_dir: runDir,
    total_selected: selectedEntries.length,
    submitted: counts.submitted,
    succeeded: counts.succeeded,
    failed: counts.failed,
    active: counts.active,
    not_yet_submitted: selectedEntries.length - counts.submitted,
    pending_task_ids: pendingTaskIds,
    config: {
      base_url: getBaseUrl(),
      input_json: path.resolve(args.inputJson),
      max_active_tasks: args.maxActiveTasks,
      submit_concurrency: args.submitConcurrency,
      query_concurrency: args.queryConcurrency,
      poll_interval_ms: args.pollIntervalMs,
      wait: args.wait,
      representative_only: args.representativeOnly,
    },
    reusable_base_payload: reusableBasePayload,
    tasks,
  });
}

async function runWithConcurrency(items, concurrency, worker) {
  if (!items.length) {
    return;
  }

  let cursor = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (cursor < items.length) {
      const currentIndex = cursor;
      cursor += 1;
      await worker(items[currentIndex], currentIndex);
    }
  });

  await Promise.all(runners);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const accessKey = getRequiredEnv('KLING_ACCESS_KEY');
  const secretKey = getRequiredEnv('KLING_SECRET_KEY');
  const client = createApiClient(accessKey, secretKey);

  let selectedEntries;
  let runDir;
  let tasks;
  let reusableBasePayload;

  if (args.resumeRunDir) {
    runDir = path.resolve(args.resumeRunDir);
    const resumeState = await loadResumeState(runDir);
    selectedEntries = resumeState.selectedEntries;
    tasks = resumeState.tasks;
    reusableBasePayload = resumeState.reusableBasePayload;

    if (!selectedEntries.length) {
      throw new Error(`No selected prompts found in resume directory: ${runDir}`);
    }
    if (!tasks.length) {
      throw new Error(`No submitted tasks found in resume directory: ${runDir}`);
    }
    if (!reusableBasePayload) {
      throw new Error(`No reusable base payload found in resume directory: ${runDir}`);
    }

    console.log(`Resuming existing run: ${runDir}`);
  } else {
    const allEntries = await readJson(path.resolve(args.inputJson));
    if (!Array.isArray(allEntries) || allEntries.length === 0) {
      throw new Error('Input JSON does not contain a non-empty array.');
    }

    selectedEntries = args.representativeOnly
      ? selectRepresentativeEntries(allEntries, args.limit)
      : selectAllEntries(allEntries, args.limit);
    if (selectedEntries.length === 0) {
      throw new Error('No prompt samples were selected.');
    }

    const runStamp = nowStamp();
    const suffix = args.runLabel ? slugify(args.runLabel, 64) : `${args.representativeOnly ? 'representative' : 'all-entries'}-${selectedEntries.length}`;
    runDir = path.join(path.resolve(args.outputRoot), `${runStamp}_${suffix}`);
    await ensureDir(runDir);
    await writeJson(path.join(runDir, 'selected_prompts.json'), {
      created_at: new Date().toISOString(),
      base_url: getBaseUrl(),
      input_json: path.resolve(args.inputJson),
      count: selectedEntries.length,
      selected_prompts: selectedEntries,
    });

    console.log(`Selected ${selectedEntries.length} prompt(s).`);
    console.log(`Base URL: ${getBaseUrl()}`);
    for (const item of summarizeSelection(selectedEntries.slice(0, Math.min(selectedEntries.length, 12)))) {
      console.log(`[${item.sample_index}] ${item.macro_domain}`);
      console.log(`    ${item.micro_domain}`);
      console.log(`    ${item.prompt_preview}`);
    }
    if (selectedEntries.length > 12) {
      console.log(`    ... plus ${selectedEntries.length - 12} more prompt(s)`);
    }

    const firstSample = selectedEntries[0];
    console.log(`\nProbing a working Kling v3 payload with sample [${firstSample.sample_index}]...`);
    const probe = await probeWorkingPayload(client, firstSample, runDir);
    console.log(`Probe accepted with variant: ${probe.variant.label}`);
    console.log(`Probe task_id: ${probe.taskId}`);

    tasks = [
      {
        ...firstSample,
        task_id: probe.taskId,
        submission_payload: probe.payload,
        submission_response: probe.response,
        submission_state: 'submitted',
        used_probe_submission: true,
      },
    ];

    reusableBasePayload = { ...probe.variant.basePayload };
    await persistSubmittedTasks(runDir, reusableBasePayload, tasks);
  }

  const startedAt = Date.now();
  while (true) {
    if (Date.now() - startedAt > args.timeoutMs) {
      break;
    }

    const submittedIndexes = new Set(tasks.map((task) => task.sample_index));
    const activeTasks = tasks.filter((task) => task.submission_state === 'submitted' && !['succeed', 'failed'].includes(task.task_status));
    const pendingSamples = selectedEntries.filter((entry) => !submittedIndexes.has(entry.sample_index));

    if (!args.wait && pendingSamples.length === 0) {
      break;
    }

    const availableSlots = args.wait ? Math.max(0, args.maxActiveTasks - activeTasks.length) : pendingSamples.length;
    const samplesToSubmit = pendingSamples.slice(0, availableSlots);

    if (samplesToSubmit.length > 0) {
      console.log(`\nSubmitting up to ${samplesToSubmit.length} task(s). Active window: ${activeTasks.length}/${args.maxActiveTasks}`);
      await runWithConcurrency(samplesToSubmit, args.submitConcurrency, async (sample) => {
        const payload = {
          ...reusableBasePayload,
          prompt: sample.prompt,
          external_task_id: `batch-${path.basename(runDir)}-${String(sample.sample_index).padStart(3, '0')}-${slugify(sample.macro_domain, 24)}`,
        };

        console.log(`Submitting [${sample.sample_index}] ${sample.macro_domain}`);
        const response = await submitTextToVideoTaskWithRetries(
          client,
          payload,
          `[${sample.sample_index}] ${sample.macro_domain}`,
        );
        const taskId = extractTaskId(response);
        if (!taskId) {
          throw new Error(`Submission for sample [${sample.sample_index}] returned no task_id: ${JSON.stringify(response)}`);
        }

        tasks.push({
          ...sample,
          task_id: taskId,
          submission_payload: payload,
          submission_response: response,
          submission_state: 'submitted',
          used_probe_submission: false,
          last_submitted_at: new Date().toISOString(),
        });
      });

      await persistSubmittedTasks(runDir, reusableBasePayload, tasks);
    }

    if (!args.wait) {
      await persistRunSummary(runDir, selectedEntries, reusableBasePayload, tasks, [], args);
      break;
    }

    const currentActiveTasks = tasks.filter((task) => task.submission_state === 'submitted' && !['succeed', 'failed'].includes(task.task_status));
    if (currentActiveTasks.length === 0 && pendingSamples.length === 0) {
      break;
    }

    if (currentActiveTasks.length > 0) {
      console.log(`\nPolling ${currentActiveTasks.length} active task(s)...`);
      await runWithConcurrency(currentActiveTasks, args.queryConcurrency, async (task) => {
        try {
          const queryResponse = await queryTextToVideoTask(client, task.task_id);
          const status = extractTaskStatus(queryResponse);
          task.last_query_response = queryResponse;
          task.last_polled_at = new Date().toISOString();
          task.task_status = status;

          if (status === 'succeed') {
            console.log(`  downloaded [${task.sample_index}] ${task.macro_domain}`);
            task.downloads = await saveTaskVideos(task, queryResponse, runDir);
          } else if (status === 'failed') {
            console.log(`  failed     [${task.sample_index}] ${task.macro_domain}`);
            task.failure_message = queryResponse?.data?.task_status_msg ?? queryResponse?.task_status_msg ?? 'Unknown error';
          } else {
            console.log(`  ${String(status ?? 'unknown').padEnd(10)} [${task.sample_index}] ${task.macro_domain}`);
          }
        } catch (error) {
          const described = describeAxiosError(error);
          task.poll_error = described;
          task.last_polled_at = new Date().toISOString();
          console.log(`  error      [${task.sample_index}] ${task.macro_domain} -> ${described.message}`);
        }
      });

      await persistSubmittedTasks(runDir, reusableBasePayload, tasks);
    }

    const pendingTaskIds = tasks
      .filter((task) => task.submission_state === 'submitted' && !['succeed', 'failed'].includes(task.task_status))
      .map((task) => task.task_id);

    await persistRunSummary(runDir, selectedEntries, reusableBasePayload, tasks, pendingTaskIds, args);

    if (pendingTaskIds.length === 0 && selectedEntries.length === tasks.length) {
      break;
    }

    await sleep(args.pollIntervalMs);
  }

  const pendingTaskIds = tasks
    .filter((task) => task.submission_state === 'submitted' && !['succeed', 'failed'].includes(task.task_status))
    .map((task) => task.task_id);

  const summary = {
    updated_at: new Date().toISOString(),
    run_dir: runDir,
    total_selected: selectedEntries.length,
    submitted: tasks.filter((task) => task.submission_state === 'submitted').length,
    succeeded,
    failed,
    active: pendingTaskIds.length,
    not_yet_submitted: selectedEntries.length - tasks.length,
    reusable_base_payload: reusableBasePayload,
    timed_out: pendingTaskIds.length > 0,
    pending_task_ids: pendingTaskIds,
    config: {
      base_url: getBaseUrl(),
      input_json: path.resolve(args.inputJson),
      max_active_tasks: args.maxActiveTasks,
      submit_concurrency: args.submitConcurrency,
      query_concurrency: args.queryConcurrency,
      poll_interval_ms: args.pollIntervalMs,
      wait: args.wait,
      representative_only: args.representativeOnly,
    },
    tasks,
  };

  await writeJson(path.join(runDir, 'run_summary.json'), summary);

  const succeeded = tasks.filter((task) => task.task_status === 'succeed').length;
  const failed = tasks.filter((task) => task.task_status === 'failed').length;
  const stillPending = tasks.length - succeeded - failed;

  console.log('\nBatch summary');
  console.log(`  succeeded: ${succeeded}`);
  console.log(`  failed:    ${failed}`);
  console.log(`  pending:   ${stillPending}`);
  console.log(`  summary:   ${path.join(runDir, 'run_summary.json')}`);
}

main().catch((error) => {
  console.error('\nBatch run failed.');
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
