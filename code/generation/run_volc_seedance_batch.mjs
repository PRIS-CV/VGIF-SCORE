import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3';
const DEFAULT_MODEL = 'doubao-seedance-2-0-260128';
const DEFAULT_INPUT_JSON = path.join(__dirname, 'all_entries_merged_final.json');
const DEFAULT_OUTPUT_ROOT = path.join(__dirname, 'outputs', 'volc_seedance_2_0_720p_5s');
const DEFAULT_CONCURRENCY = 3;
const DEFAULT_BATCH_SIZE = 20;
const DEFAULT_POLL_INTERVAL_MS = 30000;
const DEFAULT_SUBMIT_TIMEOUT_MS = 120000;
const DEFAULT_STATUS_TIMEOUT_MS = 60000;
const DEFAULT_SUBMIT_MAX_RETRIES = 8;

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT_JSON,
    outputRoot: DEFAULT_OUTPUT_ROOT,
    baseUrl: process.env.ARK_BASE_URL?.trim() || DEFAULT_BASE_URL,
    model: process.env.ARK_MODEL?.trim() || DEFAULT_MODEL,
    resolution: '720p',
    duration: 5,
    ratio: 'adaptive',
    watermark: false,
    generateAudio: false,
    serviceTier: 'default',
    concurrency: DEFAULT_CONCURRENCY,
    batchSize: DEFAULT_BATCH_SIZE,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
    executionExpiresAfter: 172800,
    submitMaxRetries: DEFAULT_SUBMIT_MAX_RETRIES,
    noDownload: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    const next = argv[i + 1];

    if (token === '--input') {
      args.input = path.resolve(next);
      i += 1;
      continue;
    }
    if (token === '--output-root') {
      args.outputRoot = path.resolve(next);
      i += 1;
      continue;
    }
    if (token === '--resume-run-dir') {
      args.resumeRunDir = path.resolve(next);
      i += 1;
      continue;
    }
    if (token === '--base-url') {
      args.baseUrl = next;
      i += 1;
      continue;
    }
    if (token === '--model') {
      args.model = next;
      i += 1;
      continue;
    }
    if (token === '--resolution') {
      args.resolution = next;
      i += 1;
      continue;
    }
    if (token === '--duration') {
      args.duration = Number(next);
      i += 1;
      continue;
    }
    if (token === '--ratio') {
      args.ratio = next;
      i += 1;
      continue;
    }
    if (token === '--service-tier') {
      args.serviceTier = next;
      i += 1;
      continue;
    }
    if (token === '--execution-expires-after') {
      args.executionExpiresAfter = Number(next);
      i += 1;
      continue;
    }
    if (token === '--concurrency') {
      args.concurrency = Number(next);
      i += 1;
      continue;
    }
    if (token === '--batch-size') {
      args.batchSize = Number(next);
      i += 1;
      continue;
    }
    if (token === '--poll-interval-sec') {
      args.pollIntervalMs = Number(next) * 1000;
      i += 1;
      continue;
    }
    if (token === '--submit-max-retries') {
      args.submitMaxRetries = Number(next);
      i += 1;
      continue;
    }
    if (token === '--generate-audio') {
      args.generateAudio = true;
      continue;
    }
    if (token === '--silent-video') {
      args.generateAudio = false;
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
    if (token === '--no-download') {
      args.noDownload = true;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }

  if (!Number.isInteger(args.concurrency) || args.concurrency <= 0) {
    throw new Error('`--concurrency` must be a positive integer.');
  }
  if (!Number.isInteger(args.batchSize) || args.batchSize <= 0) {
    throw new Error('`--batch-size` must be a positive integer.');
  }
  if (!Number.isFinite(args.duration) || args.duration <= 0) {
    throw new Error('`--duration` must be a positive number.');
  }
  if (!Number.isFinite(args.executionExpiresAfter) || args.executionExpiresAfter < 3600) {
    throw new Error('`--execution-expires-after` must be at least 3600.');
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

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
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

function previewPrompt(prompt, maxLength = 80) {
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
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

function createClient(apiKey, baseUrl) {
  return axios.create({
    baseURL: baseUrl,
    timeout: DEFAULT_SUBMIT_TIMEOUT_MS,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
  });
}

function isRetryable(error) {
  if (!axios.isAxiosError(error)) {
    return false;
  }
  const status = error.response?.status;
  return status === 408 || status === 409 || status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

function buildInitialTasks(entries, runDir) {
  return entries.map((entry, index) => {
    const macroDomain = entry?.domain_info?.macro_domain ?? '';
    const microDomain = entry?.domain_info?.micro_domain ?? '';
    const prompt = entry?.prompt ?? '';
    const stem = `${String(index + 1).padStart(3, '0')}_${slugify(`${macroDomain} ${microDomain}` || prompt || `item-${index + 1}`) || `item-${index + 1}`}`;

    return {
      index,
      item_no: index + 1,
      macro_domain: macroDomain,
      micro_domain: microDomain,
      prompt,
      prompt_preview: previewPrompt(prompt),
      remote_id: null,
      status: 'pending',
      error: null,
      submitted_at: null,
      updated_at: null,
      completed_at: null,
      batch_no: null,
      attempts: 0,
      output: {
        video_url: null,
        last_frame_url: null,
        local_video_path: path.join(runDir, 'videos', `${stem}.mp4`),
      },
    };
  });
}

async function saveState(paths, runMeta, tasks) {
  const counts = tasks.reduce((acc, task) => {
    acc[task.status] = (acc[task.status] || 0) + 1;
    return acc;
  }, {});

  const summary = {
    run: runMeta,
    counts,
    total: tasks.length,
    pending: counts.pending || 0,
    submitted: counts.submitted || 0,
    queued: counts.queued || 0,
    running: counts.running || 0,
    succeeded: counts.succeeded || 0,
    failed: counts.failed || 0,
    expired: counts.expired || 0,
    cancelled: counts.cancelled || 0,
    downloaded: tasks.filter((task) => task.output?.local_video_path && task.status === 'succeeded').length,
    updated_at: new Date().toISOString(),
  };

  await writeJson(paths.runMetaPath, runMeta);
  await writeJson(paths.tasksPath, tasks);
  await writeJson(paths.summaryPath, summary);
}

async function downloadFile(url, targetPath) {
  await ensureDir(path.dirname(targetPath));
  const response = await axios.get(url, {
    responseType: 'arraybuffer',
    timeout: 300000,
  });
  await fs.writeFile(targetPath, response.data);
}

async function submitOne(client, task, args) {
  let lastError = null;

  for (let attempt = 1; attempt <= args.submitMaxRetries; attempt += 1) {
    try {
      const response = await client.post(
        '/contents/generations/tasks',
        {
          model: args.model,
          content: [
            {
              type: 'text',
              text: task.prompt,
            },
          ],
          resolution: args.resolution,
          duration: args.duration,
          ratio: args.ratio,
          watermark: args.watermark,
          generate_audio: args.generateAudio,
          service_tier: args.serviceTier,
          execution_expires_after: args.executionExpiresAfter,
        },
        {
          timeout: DEFAULT_SUBMIT_TIMEOUT_MS,
        },
      );

      return response.data?.id;
    } catch (error) {
      lastError = error;
      if (!isRetryable(error) || attempt === args.submitMaxRetries) {
        break;
      }
      const delayMs = Math.min(60000, 2000 * 2 ** (attempt - 1));
      await sleep(delayMs);
    }
  }

  throw lastError;
}

async function queryOne(client, remoteId) {
  const response = await client.get(`/contents/generations/tasks/${encodeURIComponent(remoteId)}`, {
    timeout: DEFAULT_STATUS_TIMEOUT_MS,
  });
  return response.data;
}

async function runWithConcurrency(items, limit, worker) {
  let cursor = 0;
  const workers = new Array(Math.min(limit, items.length)).fill(null).map(async () => {
    while (cursor < items.length) {
      const current = items[cursor];
      cursor += 1;
      await worker(current);
    }
  });
  await Promise.all(workers);
}

function isTerminalStatus(status) {
  return status === 'succeeded' || status === 'failed' || status === 'expired' || status === 'cancelled';
}

function collectBatchCandidates(tasks) {
  return tasks.filter((task) => task.status === 'pending' || task.status === 'submitted' || task.status === 'queued' || task.status === 'running');
}

function splitIntoBatches(tasks, batchSize) {
  const batches = [];
  for (let i = 0; i < tasks.length; i += batchSize) {
    batches.push(tasks.slice(i, i + batchSize));
  }
  return batches;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = getRequiredEnv('ARK_API_KEY');

  let runDir;
  let runMeta;
  let tasks;

  if (args.resumeRunDir) {
    runDir = args.resumeRunDir;
    const paths = {
      runMetaPath: path.join(runDir, 'run_meta.json'),
      tasksPath: path.join(runDir, 'tasks.json'),
      summaryPath: path.join(runDir, 'summary.json'),
    };
    runMeta = await readJson(paths.runMetaPath);
    tasks = await readJson(paths.tasksPath);
  } else {
    runDir = path.join(args.outputRoot, nowStamp());
    await ensureDir(runDir);
    await ensureDir(path.join(runDir, 'videos'));

    const entries = await readJson(args.input);
    tasks = buildInitialTasks(entries, runDir);
    runMeta = {
      created_at: new Date().toISOString(),
      run_dir: runDir,
      input_json: args.input,
      model: args.model,
      base_url: args.baseUrl,
      total_prompts: tasks.length,
      requested_settings: {
        resolution: args.resolution,
        duration: args.duration,
        ratio: args.ratio,
        watermark: args.watermark,
        generate_audio: args.generateAudio,
        service_tier: args.serviceTier,
        execution_expires_after: args.executionExpiresAfter,
      },
      scheduler: {
        concurrency: args.concurrency,
        batch_size: args.batchSize,
        poll_interval_ms: args.pollIntervalMs,
      },
      notes: [
        'Official Seedance 2.0 online quota in the model list: personal max concurrency 3, enterprise max concurrency 10.',
        'This runner defaults to concurrency 3 for safety and allows manual override.',
      ],
    };
  }

  const paths = {
    runMetaPath: path.join(runDir, 'run_meta.json'),
    tasksPath: path.join(runDir, 'tasks.json'),
    summaryPath: path.join(runDir, 'summary.json'),
  };

  const client = createClient(apiKey, args.baseUrl);
  await ensureDir(path.join(runDir, 'videos'));
  await saveState(paths, runMeta, tasks);

  const activeCandidates = collectBatchCandidates(tasks);
  const batches = splitIntoBatches(activeCandidates, args.batchSize);

  for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
    const batchNo = batchIndex + 1;
    const batchTasks = batches[batchIndex];

    for (const task of batchTasks) {
      if (task.batch_no == null) {
        task.batch_no = batchNo;
      }
    }
    await saveState(paths, runMeta, tasks);

    const toSubmit = batchTasks.filter((task) => task.status === 'pending');
    if (toSubmit.length > 0) {
      await runWithConcurrency(toSubmit, args.concurrency, async (task) => {
        task.attempts += 1;
        task.status = 'submitting';
        task.updated_at = new Date().toISOString();
        await saveState(paths, runMeta, tasks);

        try {
          const remoteId = await submitOne(client, task, args);
          task.remote_id = remoteId;
          task.status = 'submitted';
          task.submitted_at = new Date().toISOString();
          task.updated_at = task.submitted_at;
          task.error = null;
        } catch (error) {
          task.status = 'failed';
          task.updated_at = new Date().toISOString();
          task.completed_at = task.updated_at;
          task.error = axios.isAxiosError(error)
            ? {
                message: error.message,
                status: error.response?.status ?? null,
                data: error.response?.data ?? null,
              }
            : {
                message: String(error),
              };
        }

        await saveState(paths, runMeta, tasks);
      });
    }

    const activeRemoteTasks = batchTasks.filter((task) => task.remote_id && !isTerminalStatus(task.status));
    while (activeRemoteTasks.some((task) => !isTerminalStatus(task.status))) {
      await runWithConcurrency(
        activeRemoteTasks.filter((task) => !isTerminalStatus(task.status)),
        args.concurrency,
        async (task) => {
          try {
            const result = await queryOne(client, task.remote_id);
            task.status = result?.status || task.status;
            task.updated_at = new Date().toISOString();
            if (isTerminalStatus(task.status)) {
              task.completed_at = task.updated_at;
            }
            task.output.video_url = result?.content?.video_url ?? task.output.video_url;
            task.output.last_frame_url = result?.content?.last_frame_url ?? task.output.last_frame_url;
            task.error = result?.error ?? null;
          } catch (error) {
            if (!isRetryable(error)) {
              task.status = 'failed';
              task.updated_at = new Date().toISOString();
              task.completed_at = task.updated_at;
              task.error = axios.isAxiosError(error)
                ? {
                    message: error.message,
                    status: error.response?.status ?? null,
                    data: error.response?.data ?? null,
                  }
                : {
                    message: String(error),
                  };
            }
          }
        },
      );

      await saveState(paths, runMeta, tasks);

      if (activeRemoteTasks.every((task) => isTerminalStatus(task.status))) {
        break;
      }
      await sleep(args.pollIntervalMs);
    }

    if (!args.noDownload) {
      const toDownload = batchTasks.filter((task) => task.status === 'succeeded' && task.output?.video_url);
      await runWithConcurrency(toDownload, Math.min(args.concurrency, 4), async (task) => {
        try {
          await fs.access(task.output.local_video_path);
          return;
        } catch {
          // fall through
        }

        try {
          await downloadFile(task.output.video_url, task.output.local_video_path);
          task.updated_at = new Date().toISOString();
        } catch (error) {
          task.error = {
            message: `Video download failed: ${error.message}`,
          };
          task.updated_at = new Date().toISOString();
        }
        await saveState(paths, runMeta, tasks);
      });
    }
  }

  await saveState(paths, runMeta, tasks);

  const counts = tasks.reduce((acc, task) => {
    acc[task.status] = (acc[task.status] || 0) + 1;
    return acc;
  }, {});

  console.log(`Run directory: ${runDir}`);
  console.log(`Total: ${tasks.length}`);
  console.log(`Succeeded: ${counts.succeeded || 0}`);
  console.log(`Failed: ${counts.failed || 0}`);
  console.log(`Expired: ${counts.expired || 0}`);
  console.log(`Downloaded: ${tasks.filter((task) => task.status === 'succeeded' && task.output?.local_video_path).length}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
