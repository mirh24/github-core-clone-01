// MCP (M4): Model Context Protocol over streamable HTTP at POST /mcp.
//
// Two protocol eras, one endpoint (kit sdk/conformance/mcp-client.ts):
//   modern 2026-07-28: `server/discover` returns identity, capabilities and
//           supportedVersions in one request; no handshake, no session.
//   legacy 2025-11-25 and earlier: `initialize` + `notifications/initialized`.
// The suite probes server/discover first and falls back to the handshake, so
// both are implemented; a plain JSON body answers every request (no SSE).
//
// Tool results carry the object twice: as a text block (JSON) and as
// structuredContent, because the suite reads either. Tools mutate the same
// Forge the REST and GraphQL surfaces read, so an object created over MCP is
// the same object over REST. Auth is decided before any method runs: no (or a
// wrong) token is HTTP 401. A body that is not JSON is HTTP 400 with
// JSON-RPC -32700. Unknown tools and missing required arguments are JSON-RPC
// -32602; an unknown method is -32601. No clock, no randomness anywhere.

import type { Forge, Repo } from './store.ts';

type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const SERVER_NAME = 'makefun-forge';
const SERVER_VERSION = '1.0.0';
const SUPPORTED_VERSIONS = ['2026-07-28', '2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];

interface ToolDef {
  name: string;
  description: string;
  inputSchema: JsonObject;
}

function stringProp(description: string): JsonObject {
  return { type: 'string', description };
}

function booleanProp(description: string): JsonObject {
  return { type: 'boolean', description };
}

/** The six tools, in a fixed order: tools/list must be byte-identical every call. */
const TOOLS: readonly ToolDef[] = Object.freeze([
  {
    name: 'create_repo',
    description: 'Create a repository owned by the authenticated user.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: stringProp('The login that will own the repository.'),
        name: stringProp('The repository name.'),
        description: stringProp('A short description of the repository.'),
        private: booleanProp('Whether the repository is private.'),
      },
      required: ['name'],
    },
  },
  {
    name: 'get_repo',
    description: 'Read one repository by owner and name.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: stringProp('The repository owner login.'),
        repo: stringProp('The repository name.'),
      },
      required: ['owner', 'repo'],
    },
  },
  {
    name: 'create_issue',
    description: 'Create an issue in a repository.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: stringProp('The repository owner login.'),
        repo: stringProp('The repository name.'),
        title: stringProp('The issue title.'),
        body: stringProp('The issue body.'),
      },
      required: ['owner', 'repo', 'title'],
    },
  },
  {
    name: 'list_issues',
    description: 'List the issues of a repository.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: stringProp('The repository owner login.'),
        repo: stringProp('The repository name.'),
        state: stringProp('Filter by state: open, closed or all.'),
      },
      required: ['owner', 'repo'],
    },
  },
  {
    name: 'create_pull_request',
    description: 'Open a pull request from a head branch into a base branch.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: stringProp('The repository owner login.'),
        repo: stringProp('The repository name.'),
        title: stringProp('The pull request title.'),
        body: stringProp('The pull request body.'),
        head: stringProp('The head branch name.'),
        base: stringProp('The base branch name.'),
      },
      required: ['owner', 'repo', 'title', 'head', 'base'],
    },
  },
  {
    name: 'merge_pull_request',
    description: 'Squash-merge a pull request onto its base branch.',
    inputSchema: {
      type: 'object',
      properties: {
        owner: stringProp('The repository owner login.'),
        repo: stringProp('The repository name.'),
        pull_number: stringProp('The pull request number.'),
      },
      required: ['owner', 'repo', 'pull_number'],
    },
  },
]);

// -- views -----------------------------------------------------------------------

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

/** The repo the tools read: the named owner first, the token owner as fallback. */
function findRepo(forge: Forge, viewerLogin: string, owner: unknown, name: unknown): Repo | undefined {
  if (typeof name !== 'string' || name === '') return undefined;
  if (typeof owner === 'string' && owner !== '') {
    const direct = forge.getRepo(owner, name);
    if (direct !== undefined) return direct;
  }
  return forge.getRepo(viewerLogin, name);
}

// -- JSON-RPC ----------------------------------------------------------------------

function rpcResult(id: unknown, result: unknown): JsonObject {
  return { jsonrpc: '2.0', id: id ?? null, result };
}

function rpcError(id: unknown, code: number, message: string): JsonObject {
  return { jsonrpc: '2.0', id: id ?? null, error: { code, message } };
}

function toolResult(value: unknown): JsonObject {
  const text = JSON.stringify(value);
  return { content: [{ type: 'text', text }], structuredContent: value, isError: false };
}

function toolError(message: string): JsonObject {
  return { content: [{ type: 'text', text: message }], isError: true };
}

// -- tools -------------------------------------------------------------------------

const TOOL_REQUIRED: Record<string, readonly string[]> = {
  create_repo: ['name'],
  get_repo: ['owner', 'repo'],
  create_issue: ['owner', 'repo', 'title'],
  list_issues: ['owner', 'repo'],
  create_pull_request: ['owner', 'repo', 'title', 'head', 'base'],
  merge_pull_request: ['owner', 'repo', 'pull_number'],
};

function callTool(forge: Forge, viewerLogin: string, name: string, args: JsonObject): JsonObject {
  const required = TOOL_REQUIRED[name];
  if (required === undefined) {
    return rpcError(null, -32602, `Unknown tool: ${name}`);
  }
  const missing = required.filter((key) => args[key] === undefined || args[key] === null || args[key] === '');
  if (missing.length > 0) {
    return rpcError(null, -32602, `Missing required argument: ${missing.join(', ')}`);
  }
  switch (name) {
    case 'create_repo': {
      const repoName = args.name as string;
      const repo = forge.createRepo(
        viewerLogin,
        repoName,
        args.private === true,
        typeof args.description === 'string' ? args.description : '',
      );
      if (repo === undefined) return toolError(`Repository ${viewerLogin}/${repoName} already exists`);
      return toolResult(repoView(repo));
    }
    case 'get_repo': {
      const repo = findRepo(forge, viewerLogin, args.owner, args.repo);
      if (repo === undefined) return toolError(`Repository ${String(args.owner)}/${String(args.repo)} not found`);
      return toolResult(repoView(repo));
    }
    case 'create_issue': {
      const repo = findRepo(forge, viewerLogin, args.owner, args.repo);
      if (repo === undefined) return toolError(`Repository ${String(args.owner)}/${String(args.repo)} not found`);
      const issue = forge.createIssue(
        repo,
        args.title as string,
        typeof args.body === 'string' ? args.body : '',
        [],
      );
      return toolResult({ id: issue.number, number: issue.number, title: issue.title, body: issue.body, state: issue.state });
    }
    case 'list_issues': {
      const repo = findRepo(forge, viewerLogin, args.owner, args.repo);
      if (repo === undefined) return toolError(`Repository ${String(args.owner)}/${String(args.repo)} not found`);
      const state = typeof args.state === 'string' ? args.state : 'open';
      const list = state === 'all' ? [...repo.issues] : repo.issues.filter((issue) => issue.state === state);
      return toolResult(list.map((issue) => ({ id: issue.number, number: issue.number, title: issue.title, body: issue.body, state: issue.state })));
    }
    case 'create_pull_request': {
      const repo = findRepo(forge, viewerLogin, args.owner, args.repo);
      if (repo === undefined) return toolError(`Repository ${String(args.owner)}/${String(args.repo)} not found`);
      const head = args.head as string;
      const base = args.base as string;
      const headSha = forge.resolveRef(repo, head);
      const baseSha = forge.resolveRef(repo, base);
      if (headSha === null || baseSha === null) return toolError(`Head or base branch not found: ${head} / ${base}`);
      const pull = forge.createPull(
        repo,
        args.title as string,
        typeof args.body === 'string' ? args.body : '',
        head,
        base,
        headSha,
        baseSha,
      );
      return toolResult({
        id: pull.number,
        number: pull.number,
        title: pull.title,
        body: pull.body,
        state: pull.state,
        head: { ref: pull.headRef, sha: pull.headSha },
        base: { ref: pull.baseRef, sha: pull.baseSha },
      });
    }
    case 'merge_pull_request': {
      const repo = findRepo(forge, viewerLogin, args.owner, args.repo);
      if (repo === undefined) return toolError(`Repository ${String(args.owner)}/${String(args.repo)} not found`);
      const number = typeof args.pull_number === 'number' ? args.pull_number : Number.parseInt(String(args.pull_number), 10);
      if (!Number.isSafeInteger(number) || number < 1) return toolError(`Invalid pull request number: ${String(args.pull_number)}`);
      const pull = forge.getPull(repo, number);
      if (pull === undefined) return toolError(`Pull request #${number} not found`);
      const result = forge.squashMerge(repo, pull);
      if (!result.ok) return toolError(`Cannot merge pull request #${number}: ${result.reason}`);
      return toolResult({ merged: true, number: pull.number, state: pull.state, sha: result.commitSha });
    }
    default:
      return toolError(`Unknown tool: ${name}`);
  }
}

// -- dispatcher ----------------------------------------------------------------------

function dispatch(forge: Forge, viewerLogin: string, request: JsonObject): JsonObject | null {
  const id = 'id' in request ? request.id : undefined;
  const method = request.method;
  if (typeof method !== 'string') return rpcError(id, -32600, 'Invalid Request: method is required');
  // Notifications (no id) get no JSON-RPC response.
  if (id === undefined) return null;

  if (method === 'server/discover') {
    return rpcResult(id, {
      resultType: 'discover',
      supportedVersions: [...SUPPORTED_VERSIONS],
      capabilities: { tools: {} },
      serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
      _meta: { 'io.modelcontextprotocol/serverInfo': { name: SERVER_NAME, version: SERVER_VERSION } },
    });
  }

  if (method === 'initialize') {
    const params = isObject(request.params) ? request.params : {};
    const offered = typeof params.protocolVersion === 'string' ? params.protocolVersion : null;
    const negotiated = offered !== null && SUPPORTED_VERSIONS.includes(offered) ? offered : SUPPORTED_VERSIONS[0];
    return rpcResult(id, {
      protocolVersion: negotiated,
      capabilities: { tools: {} },
      serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
    });
  }

  if (method === 'ping') {
    return rpcResult(id, {});
  }

  if (method === 'tools/list') {
    return rpcResult(id, {
      tools: TOOLS.map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
      })),
    });
  }

  if (method === 'tools/call') {
    const params = isObject(request.params) ? request.params : {};
    const toolName = params.name;
    const args = isObject(params.arguments) ? params.arguments : {};
    if (typeof toolName !== 'string' || toolName === '') {
      return rpcError(id, -32602, 'tools/call requires a tool name');
    }
    const outcome = callTool(forge, viewerLogin, toolName, args);
    if (isObject(outcome.error)) return { jsonrpc: '2.0', id, error: outcome.error };
    return rpcResult(id, outcome);
  }

  return rpcError(id, -32601, `Method not found: ${method}`);
}

/**
 * Handle one POST /mcp body. Returns the HTTP status and the JSON body (null
 * body means an empty reply, e.g. a notification). Auth is checked before any
 * method runs; a non-JSON body is HTTP 400 with JSON-RPC -32700.
 */
export function handleMcp(
  forge: Forge,
  viewerLogin: string,
  rawBody: string,
  authed: boolean,
): { status: number; body: unknown } {
  let parsed: unknown;
  try {
    parsed = rawBody.trim() === '' ? null : JSON.parse(rawBody);
  } catch {
    return { status: 400, body: rpcError(null, -32700, 'Parse error') };
  }
  if (!authed) {
    const id = isObject(parsed) && 'id' in parsed ? parsed.id : null;
    return { status: 401, body: rpcError(id, -32000, 'Requires authentication') };
  }
  if (parsed === null) {
    return { status: 400, body: rpcError(null, -32600, 'Invalid Request: empty body') };
  }
  if (Array.isArray(parsed)) {
    const responses: unknown[] = [];
    for (const entry of parsed) {
      if (!isObject(entry)) continue;
      const response = dispatch(forge, viewerLogin, entry);
      if (response !== null) responses.push(response);
    }
    return { status: 200, body: responses };
  }
  if (!isObject(parsed)) {
    return { status: 400, body: rpcError(null, -32600, 'Invalid Request: expected a JSON-RPC object') };
  }
  const response = dispatch(forge, viewerLogin, parsed);
  if (response === null) return { status: 202, body: null };
  return { status: 200, body: response };
}
