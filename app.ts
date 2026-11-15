// The forge's HTTP surface, M1: REST v3 repositories, branches, commits,
// contents, issues, labels and comments, plus the protocol rules.
//
// Auth: every /api/v3 route needs `Authorization: Bearer <token>` where the token
// is the one in this process's MAKEFUN_SHOP_TOKEN (the suite presents the same
// value). No token, or a wrong one, is 401 with a JSON {"message"}. A repository
// belongs to the login the token names (scenario.ts's VIEWER_LOGIN); the same name
// under another owner is 404, never 403. /healthz takes no token.
//
// Protocol, on every reply: application/json (never HTML), a non-empty
// X-GitHub-Request-Id (a per-process counter, deterministic for a given request
// sequence), unknown routes are JSON 404s, and a malformed JSON body is 400.
//
// Determinism: the store holds no clock and no randomness; responses are pure
// functions of the requests so far, so two runs on the same seed export identical
// state. (M2 added git smart HTTP, M3 pull requests with squash merge, M4 GraphQL
// v4 in graphql.ts and MCP in mcp.ts.)

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { VIEWER_LOGIN } from './scenario.ts';
import { Forge, type Issue, type Pull, type Repo } from './store.ts';
import { executeGraphQL } from './graphql.ts';
import { handleMcp } from './mcp.ts';
import {
  GitHttpError,
  handleInfoRefs,
  handleLsRefs,
  handleReceivePack,
  handleUploadPack,
  type GitService,
} from './gitproto.ts';

const JSON_TYPE = 'application/json; charset=utf-8';
const SHOP_TOKEN = process.env.MAKEFUN_SHOP_TOKEN ?? '';
const MAX_BODY_BYTES = 64 * 1024 * 1024;

type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value !== '';
}

function isValidBase64(value: string): boolean {
  const clean = value.replace(/[\r\n\t ]+/g, '');
  return clean.length % 4 === 0 && /^[A-Za-z0-9+/]*={0,2}$/.test(clean);
}

export function createApp(): Server {
  const forge = new Forge();
  let requestSeq = 0;

  function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
    requestSeq += 1;
    const text = `${JSON.stringify(body)}\n`;
    res.writeHead(status, {
      'content-type': JSON_TYPE,
      'content-length': Buffer.byteLength(text),
      'x-github-request-id': `req-${String(requestSeq).padStart(8, '0')}`,
      ...extraHeaders,
    });
    res.end(text);
  }

  function sendNoContent(res: ServerResponse, status: number): void {
    requestSeq += 1;
    res.writeHead(status, { 'x-github-request-id': `req-${String(requestSeq).padStart(8, '0')}` });
    res.end();
  }

  /** A binary (non-JSON) reply: the git smart-HTTP content types, with a request id like every reply. */
  function sendRaw(res: ServerResponse, status: number, contentType: string, body: Buffer): void {
    requestSeq += 1;
    res.writeHead(status, {
      'content-type': contentType,
      'content-length': body.length,
      'x-github-request-id': `req-${String(requestSeq).padStart(8, '0')}`,
    });
    res.end(body);
  }

  function sendError(res: ServerResponse, status: number, message: string): void {
    sendJson(res, status, { message });
  }

  function bearerToken(req: IncomingMessage): string | null {
    const header = req.headers.authorization;
    if (header === undefined) return null;
    const match = /^Bearer (.+)$/.exec(header);
    return match === null ? null : (match[1] as string);
  }

  function isAuthed(req: IncomingMessage): boolean {
    const token = bearerToken(req);
    if (token === null || token === '') return false;
    if (SHOP_TOKEN === '') return true; // local dev: any non-empty token (the trial always sets it)
    return token === SHOP_TOKEN;
  }

  function readBody(req: IncomingMessage): Promise<Buffer> {
    return new Promise<Buffer>((resolve, reject) => {
      const chunks: Buffer[] = [];
      let size = 0;
      req.on('data', (chunk: Buffer) => {
        size += chunk.length;
        if (size > MAX_BODY_BYTES) {
          req.destroy();
          reject(new Error('request body too large'));
        } else {
          chunks.push(chunk);
        }
      });
      req.on('end', () => resolve(Buffer.concat(chunks)));
      req.on('error', reject);
    });
  }

  /** A request body as a JSON object; empty body is {}; malformed JSON is null. */
  async function readJsonBody(req: IncomingMessage): Promise<JsonObject | null> {
    const buffer = await readBody(req);
    const text = buffer.toString('utf8').trim();
    if (text === '') return {};
    try {
      const value: unknown = JSON.parse(text);
      return isObject(value) ? value : null;
    } catch {
      return null;
    }
  }

  // -- views -------------------------------------------------------------------

  function repoView(repo: Repo): JsonObject {
    return {
      id: repo.id,
      name: repo.name,
      full_name: `${repo.owner}/${repo.name}`,
      private: repo.private,
      description: repo.description,
      default_branch: repo.defaultBranch,
      owner: { login: repo.owner, id: 1000, type: 'User' },
    };
  }

  function labelView(repo: Repo, name: string): JsonObject {
    const label = repo.labels.get(name);
    return { name, color: label === undefined ? 'ededed' : label.color };
  }

  function issueView(repo: Repo, issue: Issue): JsonObject {
    return {
      id: issue.number,
      number: issue.number,
      title: issue.title,
      body: issue.body,
      state: issue.state,
      labels: issue.labels.map((name) => labelView(repo, name)),
    };
  }

  function pullView(pull: Pull): JsonObject {
    return {
      id: pull.number,
      number: pull.number,
      state: pull.state,
      title: pull.title,
      body: pull.body,
      head: { ref: pull.headRef, sha: pull.headSha },
      base: { ref: pull.baseRef, sha: pull.baseSha },
      merged: pull.merged,
      merged_at: pull.mergedAt,
      merge_commit_sha: pull.mergeCommitSha,
    };
  }

  // -- line diff (for GET /pulls/{n}/files) -------------------------------------

  function splitLines(text: string): string[] {
    if (text === '') return [];
    const parts = text.split('\n');
    if (parts.length > 0 && parts[parts.length - 1] === '') parts.pop();
    return parts;
  }

  /** Added/removed line counts between two file versions, from an LCS on lines. */
  function lineStats(oldText: string, newText: string): { additions: number; deletions: number } {
    const oldLines = splitLines(oldText);
    const newLines = splitLines(newText);
    const m = oldLines.length;
    const n = newLines.length;
    let prev = new Array<number>(n + 1).fill(0);
    let curr = new Array<number>(n + 1).fill(0);
    for (let i = 1; i <= m; i += 1) {
      for (let j = 1; j <= n; j += 1) {
        curr[j] = oldLines[i - 1] === newLines[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], curr[j - 1]);
      }
      const tmp = prev;
      prev = curr;
      curr = tmp;
      curr.fill(0);
    }
    const lcs = prev[n] ?? 0;
    return { additions: n - lcs, deletions: m - lcs };
  }

  /** The files a pull request changes: diff of merge-base tree vs head tree. */
  function pullFiles(forge: Forge, pull: Pull): JsonObject[] {
    const mergeBase = forge.objects.mergeBase(pull.headSha, pull.baseSha);
    const baseTree = mergeBase === null ? null : (forge.objects.readCommit(mergeBase)?.tree ?? null);
    const headCommit = forge.objects.readCommit(pull.headSha);
    const headTree = headCommit?.tree ?? null;
    return forge.objects.diffTrees(baseTree, headTree).map((diff) => {
      const oldBlob = diff.oldSha === null ? null : (forge.objects.readBlob(diff.oldSha) ?? null);
      const newBlob = diff.newSha === null ? null : (forge.objects.readBlob(diff.newSha) ?? null);
      const stats = lineStats(oldBlob === null ? '' : oldBlob.toString('utf8'), newBlob === null ? '' : newBlob.toString('utf8'));
      return {
        filename: diff.path,
        status: diff.oldSha === null ? 'added' : diff.newSha === null ? 'removed' : 'modified',
        additions: stats.additions,
        deletions: stats.deletions,
        changes: stats.additions + stats.deletions,
      };
    });
  }

  // -- pagination ---------------------------------------------------------------

  function paginate<T>(items: readonly T[], query: URLSearchParams, origin: string, path: string): { page: T[]; link: string } {
    const rawPerPage = Number.parseInt(query.get('per_page') ?? '30', 10);
    const perPage = Number.isFinite(rawPerPage) ? Math.min(100, Math.max(1, rawPerPage)) : 30;
    const rawPage = Number.parseInt(query.get('page') ?? '1', 10);
    const page = Number.isFinite(rawPage) ? Math.max(1, rawPage) : 1;
    const totalPages = Math.max(1, Math.ceil(items.length / perPage));
    const start = (page - 1) * perPage;
    const slice = items.slice(start, start + perPage);
    const urlFor = (p: number): string => `${origin}${path}?per_page=${perPage}&page=${p}`;
    const links: string[] = [];
    if (page < totalPages) links.push(`<${urlFor(page + 1)}>; rel="next"`);
    if (page > 1) links.push(`<${urlFor(page - 1)}>; rel="prev"`);
    links.push(`<${urlFor(1)}>; rel="first"`, `<${urlFor(totalPages)}>; rel="last"`);
    return { page: slice, link: links.join(', ') };
  }

  // -- handlers -----------------------------------------------------------------

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    let url: URL;
    try {
      url = new URL(req.url ?? '/', 'http://entry');
    } catch {
      sendError(res, 400, 'Bad Request');
      return;
    }
    const method = req.method ?? 'GET';
    const rawSegments = url.pathname.split('/').filter((part) => part !== '');
    const segments: string[] = [];
    for (const part of rawSegments) {
      try {
        segments.push(decodeURIComponent(part));
      } catch {
        sendError(res, 400, 'Bad Request');
        return;
      }
    }
    const origin = `http://${req.headers.host ?? 'entry'}`;
    const path = url.pathname;

    // The env's boot probe: 200 {"ok":true}, no token needed.
    if (method === 'GET' && segments.length === 1 && segments[0] === 'healthz') {
      sendJson(res, 200, { ok: true });
      return;
    }

    const isApi = segments.length >= 2 && segments[0] === 'api' && segments[1] === 'v3';
    if (!isApi) {
      // git smart HTTP (M2): /{owner}/{repo}.git/info/refs and git-{upload,receive}-pack.
      if (segments.length >= 3 && segments[1].endsWith('.git')) {
        const owner = segments[0] as string;
        const name = segments[1].slice(0, -4);
        await handleGitRoute(req, res, method, url, owner, name, segments.slice(2));
        return;
      }
      // GraphQL v4 (M4): POST /api/graphql — auth first, then the executor.
      if (segments.length === 2 && segments[0] === 'api' && segments[1] === 'graphql') {
        if (method !== 'POST') {
          sendError(res, 404, 'Not Found');
          return;
        }
        if (!isAuthed(req)) {
          sendError(res, 401, 'Requires authentication');
          return;
        }
        const raw = await readBody(req);
        let parsed: unknown;
        try {
          parsed = JSON.parse(raw.toString('utf8'));
        } catch {
          sendJson(res, 400, { errors: [{ message: 'Malformed JSON body' }] });
          return;
        }
        const result = executeGraphQL(forge, VIEWER_LOGIN, parsed);
        sendJson(res, result.status, result.body);
        return;
      }
      // MCP (M4): POST /mcp (streamable HTTP, plain JSON answers); DELETE ends a session.
      if (segments.length === 1 && segments[0] === 'mcp') {
        if (method === 'DELETE') {
          sendNoContent(res, 204);
          return;
        }
        if (method !== 'POST') {
          sendError(res, 404, 'Not Found');
          return;
        }
        const raw = await readBody(req);
        const result = handleMcp(forge, VIEWER_LOGIN, raw.toString('utf8'), isAuthed(req));
        if (result.body === null) {
          sendNoContent(res, result.status);
          return;
        }
        sendJson(res, result.status, result.body);
        return;
      }
      sendError(res, 404, 'Not Found');
      return;
    }

    if (!isAuthed(req)) {
      sendError(res, 401, 'Requires authentication');
      return;
    }

    const rest = segments.slice(2);

    // GET /api/v3/ — the API root.
    if (rest.length === 0) {
      if (method !== 'GET') {
        sendError(res, 404, 'Not Found');
        return;
      }
      sendJson(
        res,
        200,
        { current_user_url: '/api/v3/user', rate_limit_url: '/api/v3/rate_limit' },
        { 'x-oauth-scopes': 'repo, read:org' },
      );
      return;
    }

    // GET /api/v3/meta
    if (rest.length === 1 && rest[0] === 'meta') {
      if (method !== 'GET') {
        sendError(res, 404, 'Not Found');
        return;
      }
      sendJson(res, 200, { installed_version: '1.0.0' });
      return;
    }

    // /api/v3/user/repos
    if (rest.length === 2 && rest[0] === 'user' && rest[1] === 'repos') {
      if (method === 'POST') {
        const body = await readJsonBody(req);
        if (body === null) {
          sendError(res, 400, 'Malformed JSON body');
          return;
        }
        if (!isNonEmptyString(body.name)) {
          sendError(res, 422, 'name is required');
          return;
        }
        const repo = forge.createRepo(
          VIEWER_LOGIN,
          body.name,
          body.private === true,
          typeof body.description === 'string' ? body.description : '',
        );
        if (repo === undefined) {
          sendError(res, 422, 'Repository already exists');
          return;
        }
        sendJson(res, 201, repoView(repo));
        return;
      }
      if (method === 'GET') {
        const { page, link } = paginate(forge.listRepos(VIEWER_LOGIN), url.searchParams, origin, path);
        sendJson(res, 200, page.map((repo) => repoView(repo)), { link });
        return;
      }
      sendError(res, 404, 'Not Found');
      return;
    }

    // /api/v3/repos/{owner}/{name}[...]
    if (rest.length >= 3 && rest[0] === 'repos') {
      const owner = rest[1] as string;
      const name = rest[2] as string;
      const tail = rest.slice(3);
      await handleRepoRoute(req, res, method, url, origin, path, owner, name, tail);
      return;
    }

    sendError(res, 404, 'Not Found');
  }

  // -- git smart HTTP (M2) ------------------------------------------------------

  async function handleGitRoute(
    req: IncomingMessage,
    res: ServerResponse,
    method: string,
    url: URL,
    owner: string,
    name: string,
    tail: string[],
  ): Promise<void> {
    if (!isAuthed(req)) {
      sendError(res, 401, 'Requires authentication');
      return;
    }
    try {
      // GET /{owner}/{repo}.git/info/refs?service=git-upload-pack|git-receive-pack
      if (method === 'GET' && tail.length === 2 && tail[0] === 'info' && tail[1] === 'refs') {
        const service = url.searchParams.get('service');
        if (service !== 'git-upload-pack' && service !== 'git-receive-pack') {
          sendError(res, 400, 'service is required');
          return;
        }
        const wantV2 = req.headers['git-protocol'] === 'version=2';
        const result = handleInfoRefs(forge, owner, name, service as GitService, wantV2);
        sendRaw(res, result.status, result.contentType, result.body);
        return;
      }
      // POST /{owner}/{repo}.git/git-upload-pack | git-receive-pack
      if (method === 'POST' && tail.length === 1 && (tail[0] === 'git-upload-pack' || tail[0] === 'git-receive-pack')) {
        const body = await readBody(req);
        const wantV2 = req.headers['git-protocol'] === 'version=2';
        const result =
          tail[0] === 'git-upload-pack'
            ? handleUploadPack(forge, owner, name, body, wantV2)
            : handleReceivePack(forge, owner, name, body);
        sendRaw(res, result.status, result.contentType, result.body);
        return;
      }
      sendError(res, 404, 'Not Found');
    } catch (err) {
      if (err instanceof GitHttpError) {
        sendError(res, err.status, err.message);
        return;
      }
      throw err;
    }
  }

  async function handleRepoRoute(
    req: IncomingMessage,
    res: ServerResponse,
    method: string,
    url: URL,
    origin: string,
    path: string,
    owner: string,
    name: string,
    tail: string[],
  ): Promise<void> {
    const repo = forge.getRepo(owner, name);
    if (repo === undefined) {
      sendError(res, 404, 'Not Found');
      return;
    }

    // /repos/{o}/{r} itself.
    if (tail.length === 0) {
      if (method === 'GET') {
        sendJson(res, 200, repoView(repo));
        return;
      }
      if (method === 'PATCH') {
        const body = await readJsonBody(req);
        if (body === null) {
          sendError(res, 400, 'Malformed JSON body');
          return;
        }
        if (body.description !== undefined) {
          if (typeof body.description !== 'string') {
            sendError(res, 400, 'description must be a string');
            return;
          }
          repo.description = body.description;
        }
        if (body.private !== undefined) {
          if (typeof body.private !== 'boolean') {
            sendError(res, 400, 'private must be a boolean');
            return;
          }
          repo.private = body.private;
        }
        sendJson(res, 200, repoView(repo));
        return;
      }
      if (method === 'DELETE') {
        forge.deleteRepo(owner, name);
        sendNoContent(res, 204);
        return;
      }
      sendError(res, 404, 'Not Found');
      return;
    }

    // GET /repos/{o}/{r}/branches
    if (tail.length === 1 && tail[0] === 'branches') {
      if (method !== 'GET') {
        sendError(res, 404, 'Not Found');
        return;
      }
      sendJson(
        res,
        200,
        forge.listBranches(repo).map((branch) => ({ name: branch.name, commit: { sha: branch.sha } })),
      );
      return;
    }

    // GET /repos/{o}/{r}/commits?sha=<ref>
    if (tail.length === 1 && tail[0] === 'commits') {
      if (method !== 'GET') {
        sendError(res, 404, 'Not Found');
        return;
      }
      const ref = url.searchParams.get('sha') ?? undefined;
      const tip = forge.resolveRef(repo, ref);
      if (tip === null) {
        if (ref !== undefined) {
          sendError(res, 404, 'Not Found');
          return;
        }
        sendJson(res, 200, []);
        return;
      }
      const commits = forge.objects.walkCommits(tip).map((sha) => {
        const commit = forge.objects.readCommit(sha);
        return { sha, commit: { message: commit === undefined ? '' : commit.message } };
      });
      sendJson(res, 200, commits);
      return;
    }

    // POST /repos/{o}/{r}/git/refs
    if (tail.length === 2 && tail[0] === 'git' && tail[1] === 'refs') {
      if (method !== 'POST') {
        sendError(res, 404, 'Not Found');
        return;
      }
      const body = await readJsonBody(req);
      if (body === null) {
        sendError(res, 400, 'Malformed JSON body');
        return;
      }
      if (!isNonEmptyString(body.ref) || !body.ref.startsWith('refs/')) {
        sendError(res, 422, 'ref is required');
        return;
      }
      if (!isNonEmptyString(body.sha) || !/^[0-9a-f]{40}$/i.test(body.sha)) {
        sendError(res, 422, 'sha is required');
        return;
      }
      if (repo.refs.has(body.ref)) {
        sendError(res, 422, 'Reference already exists');
        return;
      }
      if (forge.objects.readCommit(body.sha) === undefined) {
        sendError(res, 422, 'Invalid commit sha');
        return;
      }
      repo.refs.set(body.ref, body.sha.toLowerCase());
      sendJson(res, 201, { ref: body.ref, object: { sha: body.sha.toLowerCase() } });
      return;
    }

    // /repos/{o}/{r}/contents/{path...}
    if (tail.length >= 1 && tail[0] === 'contents') {
      const filePath = tail.slice(1).join('/');
      await handleContents(req, res, method, url, repo, filePath);
      return;
    }

    // POST /repos/{o}/{r}/labels
    if (tail.length === 1 && tail[0] === 'labels') {
      if (method !== 'POST') {
        sendError(res, 404, 'Not Found');
        return;
      }
      const body = await readJsonBody(req);
      if (body === null) {
        sendError(res, 400, 'Malformed JSON body');
        return;
      }
      if (!isNonEmptyString(body.name)) {
        sendError(res, 422, 'name is required');
        return;
      }
      const color = typeof body.color === 'string' && body.color !== '' ? body.color : 'ededed';
      repo.labels.set(body.name, { name: body.name, color });
      sendJson(res, 201, { name: body.name, color });
      return;
    }

    // /repos/{o}/{r}/issues[...]
    if (tail.length >= 1 && tail[0] === 'issues') {
      await handleIssues(req, res, method, url, origin, path, repo, tail.slice(1));
      return;
    }

    // /repos/{o}/{r}/pulls[...]
    if (tail.length >= 1 && tail[0] === 'pulls') {
      await handlePulls(req, res, method, url, origin, path, repo, tail.slice(1));
      return;
    }

    sendError(res, 404, 'Not Found');
  }

  async function handlePulls(
    req: IncomingMessage,
    res: ServerResponse,
    method: string,
    url: URL,
    origin: string,
    path: string,
    repo: Repo,
    tail: string[],
  ): Promise<void> {
    // /pulls — list or create.
    if (tail.length === 0) {
      if (method === 'GET') {
        const state = url.searchParams.get('state') ?? 'open';
        let list = [...repo.pulls];
        if (state !== 'all') list = list.filter((pull) => pull.state === state);
        list.sort((a, b) => b.number - a.number);
        const { page, link } = paginate(list, url.searchParams, origin, path);
        sendJson(res, 200, page.map((pull) => pullView(pull)), { link });
        return;
      }
      if (method === 'POST') {
        const body = await readJsonBody(req);
        if (body === null) {
          sendError(res, 400, 'Malformed JSON body');
          return;
        }
        if (!isNonEmptyString(body.title)) {
          sendError(res, 422, 'title is required');
          return;
        }
        if (!isNonEmptyString(body.head) || !isNonEmptyString(body.base)) {
          sendError(res, 422, 'head and base are required');
          return;
        }
        const headSha = forge.resolveRef(repo, body.head);
        const baseSha = forge.resolveRef(repo, body.base);
        if (headSha === null || baseSha === null) {
          sendError(res, 422, 'head or base branch not found');
          return;
        }
        const pull = forge.createPull(
          repo,
          body.title,
          typeof body.body === 'string' ? body.body : '',
          body.head,
          body.base,
          headSha,
          baseSha,
        );
        sendJson(res, 201, pullView(pull));
        return;
      }
      sendError(res, 404, 'Not Found');
      return;
    }

    // /pulls/{n}[...]
    const number = Number.parseInt(tail[0] as string, 10);
    if (!Number.isSafeInteger(number) || number < 1) {
      sendError(res, 404, 'Not Found');
      return;
    }
    const pull = forge.getPull(repo, number);
    if (pull === undefined) {
      sendError(res, 404, 'Not Found');
      return;
    }

    if (tail.length === 1) {
      if (method === 'GET') {
        sendJson(res, 200, pullView(pull));
        return;
      }
      sendError(res, 404, 'Not Found');
      return;
    }

    if (tail.length === 2 && tail[1] === 'files') {
      if (method !== 'GET') {
        sendError(res, 404, 'Not Found');
        return;
      }
      sendJson(res, 200, pullFiles(forge, pull));
      return;
    }

    if (tail.length === 2 && tail[1] === 'merge') {
      if (method !== 'PUT') {
        sendError(res, 404, 'Not Found');
        return;
      }
      const body = await readJsonBody(req);
      if (body === null) {
        sendError(res, 400, 'Malformed JSON body');
        return;
      }
      const result = forge.squashMerge(repo, pull);
      if (!result.ok) {
        if (result.reason === 'already-merged') {
          sendError(res, 405, 'Pull request has already been merged');
          return;
        }
        if (result.reason === 'conflict') {
          sendError(res, 405, 'Merge conflict');
          return;
        }
        sendError(res, 404, 'Not Found');
        return;
      }
      sendJson(res, 200, { merged: true, sha: result.commitSha, message: 'Pull request successfully merged' });
      return;
    }

    sendError(res, 404, 'Not Found');
  }

  async function handleContents(
    req: IncomingMessage,
    res: ServerResponse,
    method: string,
    url: URL,
    repo: Repo,
    filePath: string,
  ): Promise<void> {
    if (method === 'GET') {
      const ref = url.searchParams.get('ref') ?? undefined;
      const tip = forge.resolveRef(repo, ref);
      if (tip === null) {
        sendError(res, 404, 'Not Found');
        return;
      }
      const commit = forge.objects.readCommit(tip);
      if (commit === undefined) {
        sendError(res, 404, 'Not Found');
        return;
      }
      const found = forge.objects.readPath(commit.tree, filePath);
      if (found === null) {
        sendError(res, 404, 'Not Found');
        return;
      }
      if (found.kind === 'file') {
        const blob = forge.objects.readBlob(found.sha);
        if (blob === undefined) {
          sendError(res, 404, 'Not Found');
          return;
        }
        const baseName = filePath.split('/').pop() ?? filePath;
        sendJson(res, 200, {
          type: 'file',
          encoding: 'base64',
          size: blob.length,
          name: baseName,
          path: filePath,
          sha: found.sha,
          content: blob.toString('base64'),
        });
        return;
      }
      const entries = found.entries.map((entry) => {
        let size = 0;
        if (!entry.isTree) {
          const blob = forge.objects.readBlob(entry.sha);
          size = blob === undefined ? 0 : blob.length;
        }
        return {
          type: entry.isTree ? 'dir' : 'file',
          size,
          name: entry.name,
          path: filePath === '' ? entry.name : `${filePath}/${entry.name}`,
          sha: entry.sha,
        };
      });
      sendJson(res, 200, entries);
      return;
    }

    if (method === 'PUT') {
      const body = await readJsonBody(req);
      if (body === null) {
        sendError(res, 400, 'Malformed JSON body');
        return;
      }
      if (!isNonEmptyString(body.content) || !isValidBase64(body.content)) {
        sendError(res, 400, 'content must be base64');
        return;
      }
      const message = typeof body.message === 'string' && body.message !== '' ? body.message : 'Update file';
      const branch = typeof body.branch === 'string' && body.branch !== '' ? body.branch : repo.defaultBranch;
      const branchRef = `refs/heads/${branch}`;
      const tip = repo.refs.get(branchRef);
      if (tip === undefined && repo.refs.size > 0) {
        sendError(res, 404, 'Branch not found');
        return;
      }
      const content = Buffer.from(body.content.replace(/[\r\n\t ]+/g, ''), 'base64');
      const currentSha = forge.fileShaAt(repo, branch, filePath);
      const givenSha = typeof body.sha === 'string' ? body.sha : undefined;
      if (givenSha !== undefined) {
        if (currentSha === null || currentSha.toLowerCase() !== givenSha.toLowerCase()) {
          sendError(res, 409, 'sha does not match the current file');
          return;
        }
        const updated = forge.commitFile(repo, branch, filePath, content, message);
        if (updated === null) {
          sendError(res, 404, 'Branch not found');
          return;
        }
        sendJson(res, 200, {
          content: { name: filePath.split('/').pop() ?? filePath, path: filePath, sha: updated.blobSha, size: content.length, encoding: 'base64' },
          commit: { sha: updated.commitSha },
        });
        return;
      }
      if (currentSha !== null) {
        sendError(res, 422, 'file already exists');
        return;
      }
      const created = forge.commitFile(repo, branch, filePath, content, message);
      if (created === null) {
        sendError(res, 404, 'Branch not found');
        return;
      }
      sendJson(res, 201, {
        content: { name: filePath.split('/').pop() ?? filePath, path: filePath, sha: created.blobSha, size: content.length, encoding: 'base64' },
        commit: { sha: created.commitSha },
      });
      return;
    }

    if (method === 'DELETE') {
      const body = await readJsonBody(req);
      if (body === null) {
        sendError(res, 400, 'Malformed JSON body');
        return;
      }
      const message = typeof body.message === 'string' && body.message !== '' ? body.message : 'Delete file';
      const currentSha = forge.fileShaAt(repo, repo.defaultBranch, filePath);
      if (currentSha === null) {
        sendError(res, 404, 'Not Found');
        return;
      }
      if (typeof body.sha === 'string' && body.sha !== '' && body.sha.toLowerCase() !== currentSha.toLowerCase()) {
        sendError(res, 409, 'sha does not match the current file');
        return;
      }
      const removed = forge.deleteFile(repo, repo.defaultBranch, filePath, message);
      if (removed === null) {
        sendError(res, 404, 'Not Found');
        return;
      }
      sendJson(res, 200, { commit: { sha: removed.commitSha } });
      return;
    }

    sendError(res, 404, 'Not Found');
  }

  async function handleIssues(
    req: IncomingMessage,
    res: ServerResponse,
    method: string,
    url: URL,
    origin: string,
    path: string,
    repo: Repo,
    tail: string[],
  ): Promise<void> {
    // /issues — list or create.
    if (tail.length === 0) {
      if (method === 'GET') {
        const state = url.searchParams.get('state') ?? 'open';
        const direction = url.searchParams.get('direction') ?? 'desc';
        let list = [...repo.issues];
        if (state !== 'all') list = list.filter((issue) => issue.state === state);
        list.sort((a, b) => (direction === 'asc' ? a.number - b.number : b.number - a.number));
        const { page, link } = paginate(list, url.searchParams, origin, path);
        sendJson(res, 200, page.map((issue) => issueView(repo, issue)), { link });
        return;
      }
      if (method === 'POST') {
        const body = await readJsonBody(req);
        if (body === null) {
          sendError(res, 400, 'Malformed JSON body');
          return;
        }
        if (!isNonEmptyString(body.title)) {
          sendError(res, 422, 'title is required');
          return;
        }
        const labels: string[] = [];
        if (body.labels !== undefined) {
          if (!Array.isArray(body.labels) || body.labels.some((label) => typeof label !== 'string')) {
            sendError(res, 422, 'labels must be an array of strings');
            return;
          }
          for (const label of body.labels as string[]) {
            if (!repo.labels.has(label)) {
              sendError(res, 422, `label "${label}" does not exist`);
              return;
            }
            labels.push(label);
          }
        }
        const issue = forge.createIssue(repo, body.title, typeof body.body === 'string' ? body.body : '', labels);
        sendJson(res, 201, issueView(repo, issue));
        return;
      }
      sendError(res, 404, 'Not Found');
      return;
    }

    // /issues/{n} and /issues/{n}/comments
    const number = Number.parseInt(tail[0] as string, 10);
    if (!Number.isSafeInteger(number) || number < 1) {
      sendError(res, 404, 'Not Found');
      return;
    }
    const issue = forge.getIssue(repo, number);
    if (issue === undefined) {
      sendError(res, 404, 'Not Found');
      return;
    }

    if (tail.length === 1) {
      if (method === 'GET') {
        sendJson(res, 200, issueView(repo, issue));
        return;
      }
      if (method === 'PATCH') {
        const body = await readJsonBody(req);
        if (body === null) {
          sendError(res, 400, 'Malformed JSON body');
          return;
        }
        if (body.state !== undefined) {
          if (body.state !== 'open' && body.state !== 'closed') {
            sendError(res, 422, 'state must be open or closed');
            return;
          }
          issue.state = body.state;
        }
        if (body.title !== undefined) {
          if (typeof body.title !== 'string') {
            sendError(res, 400, 'title must be a string');
            return;
          }
          issue.title = body.title;
        }
        if (body.body !== undefined) {
          if (typeof body.body !== 'string') {
            sendError(res, 400, 'body must be a string');
            return;
          }
          issue.body = body.body;
        }
        if (body.labels !== undefined) {
          if (!Array.isArray(body.labels) || body.labels.some((label) => typeof label !== 'string')) {
            sendError(res, 422, 'labels must be an array of strings');
            return;
          }
          for (const label of body.labels as string[]) {
            if (!repo.labels.has(label)) {
              sendError(res, 422, `label "${label}" does not exist`);
              return;
            }
          }
          issue.labels = [...(body.labels as string[])];
        }
        sendJson(res, 200, issueView(repo, issue));
        return;
      }
      sendError(res, 404, 'Not Found');
      return;
    }

    if (tail.length === 2 && tail[1] === 'comments') {
      if (method === 'GET') {
        sendJson(
          res,
          200,
          issue.comments.map((comment) => ({ id: comment.id, body: comment.body })),
        );
        return;
      }
      if (method === 'POST') {
        const body = await readJsonBody(req);
        if (body === null) {
          sendError(res, 400, 'Malformed JSON body');
          return;
        }
        if (!isNonEmptyString(body.body)) {
          sendError(res, 422, 'body is required');
          return;
        }
        const comment = forge.addComment(repo, issue, body.body);
        sendJson(res, 201, { id: comment.id, body: comment.body });
        return;
      }
      sendError(res, 404, 'Not Found');
      return;
    }

    sendError(res, 404, 'Not Found');
  }

  return createServer((req: IncomingMessage, res: ServerResponse) => {
    void handle(req, res).catch((err: unknown) => {
      if (!res.headersSent) {
        try {
          sendError(res, 500, err instanceof Error ? err.message : 'Internal Server Error');
        } catch {
          res.destroy();
        }
      } else {
        res.destroy();
      }
    });
  });
}
