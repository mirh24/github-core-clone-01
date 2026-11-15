// GraphQL v4 (M4): a hand-rolled parser + executor for the fixed documents gh
// and github-mcp-server send (kit sdk/conformance/checks-graphql.ts). No general
// engine: the documents are bounded — named query/mutation operations, variables,
// inline literals (lists, input objects, enums, null), aliases and inline
// fragments — so a small tokenizer + recursive-descent parser + a static schema
// table covers them. Validation errors (unknown field/argument) are a 200 with
// an `errors` array, per the GraphQL wire; only unauthenticated requests are
// HTTP 401 (handled in app.ts, before this module runs).
//
// Global ids are opaque and stable: base64 of "Type:<repoDbId>:<number>" —
// never the plain number, never a clock. All timestamps in responses are the
// fixed epoch string, so two runs on the same requests export identical state.

import type { Forge, Issue, Pull, Repo } from './store.ts';

export type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// -- tokenizer -----------------------------------------------------------------

type Token =
  | { kind: 'punct'; value: string }
  | { kind: 'name'; value: string }
  | { kind: 'int'; value: string }
  | { kind: 'float'; value: string }
  | { kind: 'string'; value: string }
  | { kind: 'eof' };

const PUNCTS = new Set(['{', '}', '(', ')', '[', ']', ':', '!', '$', '=', '@', '|', '&']);

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const n = source.length;
  while (i < n) {
    const ch = source[i] as string;
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r' || ch === ',') {
      i += 1;
      continue;
    }
    if (ch === '#') {
      while (i < n && source[i] !== '\n') i += 1;
      continue;
    }
    if (ch === '.' && source.slice(i, i + 3) === '...') {
      tokens.push({ kind: 'punct', value: '...' });
      i += 3;
      continue;
    }
    if (PUNCTS.has(ch)) {
      tokens.push({ kind: 'punct', value: ch });
      i += 1;
      continue;
    }
    if (ch === '"') {
      let value = '';
      i += 1;
      while (i < n && source[i] !== '"') {
        const c = source[i] as string;
        if (c === '\\' && i + 1 < n) {
          const esc = source[i + 1] as string;
          if (esc === 'n') value += '\n';
          else if (esc === 't') value += '\t';
          else if (esc === 'r') value += '\r';
          else if (esc === 'u' && i + 5 < n) {
            value += String.fromCharCode(Number.parseInt(source.slice(i + 2, i + 6), 16));
            i += 4;
          } else value += esc;
          i += 2;
        } else {
          value += c;
          i += 1;
        }
      }
      if (i >= n) throw new Error('unterminated string in GraphQL document');
      i += 1;
      tokens.push({ kind: 'string', value });
      continue;
    }
    if (/[0-9-]/.test(ch)) {
      let j = i;
      if (source[j] === '-') j += 1;
      while (j < n && /[0-9]/.test(source[j] as string)) j += 1;
      let isFloat = false;
      if (j < n && source[j] === '.') {
        isFloat = true;
        j += 1;
        while (j < n && /[0-9]/.test(source[j] as string)) j += 1;
      }
      if (j < n && (source[j] === 'e' || source[j] === 'E')) {
        isFloat = true;
        j += 1;
        if (j < n && (source[j] === '+' || source[j] === '-')) j += 1;
        while (j < n && /[0-9]/.test(source[j] as string)) j += 1;
      }
      const text = source.slice(i, j);
      tokens.push(isFloat ? { kind: 'float', value: text } : { kind: 'int', value: text });
      i = j;
      continue;
    }
    if (/[_A-Za-z]/.test(ch)) {
      let j = i;
      while (j < n && /[_A-Za-z0-9]/.test(source[j] as string)) j += 1;
      tokens.push({ kind: 'name', value: source.slice(i, j) });
      i = j;
      continue;
    }
    throw new Error(`unexpected character ${JSON.stringify(ch)} in GraphQL document`);
  }
  tokens.push({ kind: 'eof' });
  return tokens;
}

// -- AST -----------------------------------------------------------------------

export type ValueNode =
  | { kind: 'variable'; name: string }
  | { kind: 'string'; value: string }
  | { kind: 'int'; value: number }
  | { kind: 'float'; value: number }
  | { kind: 'boolean'; value: boolean }
  | { kind: 'null' }
  | { kind: 'enum'; value: string }
  | { kind: 'list'; values: ValueNode[] }
  | { kind: 'object'; fields: { name: string; value: ValueNode }[] };

export interface FieldNode {
  kind: 'field';
  alias: string | null;
  name: string;
  args: { name: string; value: ValueNode }[];
  selections: SelectionNode[] | null;
}

export interface InlineFragmentNode {
  kind: 'inlineFragment';
  typeCondition: string | null;
  selections: SelectionNode[];
}

export type SelectionNode = FieldNode | InlineFragmentNode;

export interface OperationNode {
  operation: 'query' | 'mutation' | 'subscription';
  name: string | null;
  variableDefinitions: { name: string; defaultValue: ValueNode | null }[];
  selections: SelectionNode[];
}

// -- parser --------------------------------------------------------------------

class Parser {
  private readonly tokens: Token[];
  private pos = 0;

  constructor(tokens: Token[]) {
    this.tokens = tokens;
  }

  private peek(): Token {
    return this.tokens[this.pos] as Token;
  }

  private next(): Token {
    const token = this.tokens[this.pos] as Token;
    if (this.pos < this.tokens.length - 1) this.pos += 1;
    return token;
  }

  private expectPunct(value: string): void {
    const token = this.next();
    if (token.kind !== 'punct' || token.value !== value) {
      throw new Error(`expected "${value}" in GraphQL document, got ${token.kind} ${'value' in token ? token.value : ''}`);
    }
  }

  private expectName(what = 'name'): string {
    const token = this.next();
    if (token.kind !== 'name') throw new Error(`expected ${what} in GraphQL document, got ${token.kind}`);
    return token.value;
  }

  private isPunct(value: string): boolean {
    const token = this.peek();
    return token.kind === 'punct' && token.value === value;
  }

  parseDocument(): OperationNode[] {
    const operations: OperationNode[] = [];
    while (this.peek().kind !== 'eof') {
      operations.push(this.parseOperation());
    }
    if (operations.length === 0) throw new Error('empty GraphQL document');
    return operations;
  }

  private parseOperation(): OperationNode {
    let operation: 'query' | 'mutation' | 'subscription' = 'query';
    let name: string | null = null;
    const token = this.peek();
    if (token.kind === 'name' && (token.value === 'query' || token.value === 'mutation' || token.value === 'subscription')) {
      operation = token.value;
      this.next();
      const next = this.peek();
      if (next.kind === 'name') name = this.next().value as string;
    } else if (token.kind === 'name' && token.value === 'fragment') {
      throw new Error('named fragments are not supported by this GraphQL executor');
    }
    const variableDefinitions: { name: string; defaultValue: ValueNode | null }[] = [];
    if (this.isPunct('(')) {
      this.next();
      while (!this.isPunct(')')) {
        this.expectPunct('$');
        const varName = this.expectName('variable name');
        this.expectPunct(':');
        this.parseType();
        let defaultValue: ValueNode | null = null;
        if (this.isPunct('=')) {
          this.next();
          defaultValue = this.parseValue();
        }
        variableDefinitions.push({ name: varName, defaultValue });
        if (this.peek().kind === 'eof') throw new Error('unterminated variable definitions');
      }
      this.expectPunct(')');
    }
    // Directives on the operation (none in the fixed documents): skip @name(args).
    while (this.isPunct('@')) this.skipDirective();
    const selections = this.parseSelectionSet();
    return { operation, name, variableDefinitions, selections };
  }

  /** A type reference: Named, [Named], Named!, [Named!]! — kept as text, unused. */
  private parseType(): string {
    let text = '';
    const token = this.next();
    if (token.kind === 'punct' && token.value === '[') {
      text += '[' + this.parseType() + ']';
    } else if (token.kind === 'name') {
      text += token.value;
    } else {
      throw new Error('expected a type in a variable definition');
    }
    if (this.isPunct('!')) {
      this.next();
      text += '!';
    }
    return text;
  }

  private skipDirective(): void {
    this.expectPunct('@');
    this.expectName('directive name');
    if (this.isPunct('(')) {
      this.next();
      let depth = 1;
      while (depth > 0) {
        const token = this.next();
        if (token.kind === 'eof') throw new Error('unterminated directive');
        if (token.kind === 'punct' && token.value === '(') depth += 1;
        if (token.kind === 'punct' && token.value === ')') depth -= 1;
      }
    }
  }

  private parseSelectionSet(): SelectionNode[] {
    this.expectPunct('{');
    const selections: SelectionNode[] = [];
    while (!this.isPunct('}')) {
      if (this.peek().kind === 'eof') throw new Error('unterminated selection set');
      selections.push(this.parseSelection());
    }
    this.expectPunct('}');
    return selections;
  }

  private parseSelection(): SelectionNode {
    if (this.isPunct('...')) {
      this.next();
      let typeCondition: string | null = null;
      const token = this.peek();
      if (token.kind === 'name' && token.value === 'on') {
        this.next();
        typeCondition = this.expectName('type condition');
      }
      while (this.isPunct('@')) this.skipDirective();
      return { kind: 'inlineFragment', typeCondition, selections: this.parseSelectionSet() };
    }
    const first = this.expectName('field name');
    let alias: string | null = null;
    let name = first;
    if (this.isPunct(':')) {
      this.next();
      alias = first;
      name = this.expectName('field name');
    }
    const args: { name: string; value: ValueNode }[] = [];
    if (this.isPunct('(')) {
      this.next();
      while (!this.isPunct(')')) {
        const argName = this.expectName('argument name');
        this.expectPunct(':');
        args.push({ name: argName, value: this.parseValue() });
        if (this.peek().kind === 'eof') throw new Error('unterminated arguments');
      }
      this.expectPunct(')');
    }
    while (this.isPunct('@')) this.skipDirective();
    const selections = this.isPunct('{') ? this.parseSelectionSet() : null;
    return { kind: 'field', alias, name, args, selections };
  }

  private parseValue(): ValueNode {
    const token = this.next();
    if (token.kind === 'punct' && token.value === '$') {
      return { kind: 'variable', name: this.expectName('variable name') };
    }
    if (token.kind === 'punct' && token.value === '[') {
      const values: ValueNode[] = [];
      while (!this.isPunct(']')) {
        values.push(this.parseValue());
        if (this.peek().kind === 'eof') throw new Error('unterminated list value');
      }
      this.expectPunct(']');
      return { kind: 'list', values };
    }
    if (token.kind === 'punct' && token.value === '{') {
      const fields: { name: string; value: ValueNode }[] = [];
      while (!this.isPunct('}')) {
        const name = this.expectName('object field name');
        this.expectPunct(':');
        fields.push({ name, value: this.parseValue() });
        if (this.peek().kind === 'eof') throw new Error('unterminated object value');
      }
      this.expectPunct('}');
      return { kind: 'object', fields };
    }
    if (token.kind === 'string') return { kind: 'string', value: token.value };
    if (token.kind === 'int') return { kind: 'int', value: Number.parseInt(token.value, 10) };
    if (token.kind === 'float') return { kind: 'float', value: Number.parseFloat(token.value) };
    if (token.kind === 'name') {
      if (token.value === 'true') return { kind: 'boolean', value: true };
      if (token.value === 'false') return { kind: 'boolean', value: false };
      if (token.value === 'null') return { kind: 'null' };
      return { kind: 'enum', value: token.value };
    }
    throw new Error(`unexpected token in a GraphQL value: ${token.kind}`);
  }
}

function parseDocument(source: string): OperationNode[] {
  return new Parser(tokenize(source)).parseDocument();
}

// -- static schema (validation + introspection) ----------------------------------

/** The fields each type carries; __typename is allowed on every type. */
const FIELDS: Record<string, ReadonlySet<string>> = {
  Query: new Set(['viewer', 'repository', 'node', '__type', '__schema']),
  Mutation: new Set(['createIssue', 'closeIssue', 'createPullRequest', 'mergePullRequest', 'addComment']),
  Viewer: new Set(['login']),
  User: new Set(['login', 'id']),
  Repository: new Set([
    'id', 'databaseId', 'name', 'nameWithOwner', 'owner', 'defaultBranchRef', 'viewerPermission',
    'hasIssuesEnabled', 'issue', 'issues', 'pullRequests', 'pullRequest', 'description', 'private',
    'isPrivate', 'url', 'resourcePath', 'createdAt', 'updatedAt',
  ]),
  Ref: new Set(['name', 'target']),
  Issue: new Set([
    'id', 'number', 'title', 'body', 'state', 'author', 'createdAt', 'updatedAt', 'comments',
    'labels', 'url', 'resourcePath', 'databaseId', 'repository',
  ]),
  IssueConnection: new Set(['nodes', 'pageInfo', 'totalCount', 'edges']),
  PageInfo: new Set(['hasNextPage', 'hasPreviousPage', 'startCursor', 'endCursor']),
  PullRequest: new Set([
    'id', 'number', 'title', 'body', 'state', 'merged', 'mergeable', 'headRefName', 'baseRefName',
    'headRefOid', 'baseRefOid', 'files', 'author', 'createdAt', 'updatedAt', 'url', 'resourcePath',
    'databaseId',
  ]),
  PullRequestConnection: new Set(['nodes', 'pageInfo', 'totalCount']),
  PullRequestChangedFileConnection: new Set(['nodes', 'pageInfo', 'totalCount']),
  PullRequestChangedFile: new Set(['path', 'additions', 'deletions', 'changeType']),
  Comment: new Set(['id', 'body', 'author']),
  CommentConnection: new Set(['nodes', 'pageInfo', 'totalCount']),
  CreateIssuePayload: new Set(['issue']),
  CloseIssuePayload: new Set(['issue']),
  CreatePullRequestPayload: new Set(['pullRequest']),
  MergePullRequestPayload: new Set(['pullRequest']),
  AddCommentPayload: new Set(['commentEdge']),
  IssueCommentEdge: new Set(['node']),
  Label: new Set(['name', 'color']),
  LabelConnection: new Set(['nodes']),
  __Type: new Set(['name', 'kind', 'fields', 'enumValues', 'description']),
  __Field: new Set(['name', 'description']),
  __EnumValue: new Set(['name', 'description']),
  __Schema: new Set(['types', 'queryType', 'mutationType']),
};

/** The arguments each field accepts (fields not listed take none). */
const FIELD_ARGS: Record<string, Record<string, ReadonlySet<string>>> = {
  Query: {
    repository: new Set(['owner', 'name']),
    node: new Set(['id']),
    __type: new Set(['name']),
  },
  Repository: {
    issue: new Set(['number']),
    issues: new Set(['first', 'states', 'orderBy', 'after']),
    pullRequests: new Set(['first', 'after']),
    pullRequest: new Set(['number']),
  },
  PullRequest: {
    files: new Set(['first', 'after']),
  },
  __Type: {
    fields: new Set(['includeDeprecated']),
    enumValues: new Set(['includeDeprecated']),
  },
  Mutation: {
    createIssue: new Set(['input']),
    closeIssue: new Set(['input']),
    createPullRequest: new Set(['input']),
    mergePullRequest: new Set(['input']),
    addComment: new Set(['input']),
  },
};

/** __type fields per object type (what gh feature-detects). */
const TYPE_FIELDS: Record<string, readonly string[]> = {
  Issue: ['id', 'number', 'title', 'body', 'state', 'author', 'createdAt', 'updatedAt', 'comments', 'labels', 'url', 'resourcePath', 'databaseId', 'repository'],
  PullRequest: ['id', 'number', 'title', 'body', 'state', 'merged', 'mergeable', 'headRefName', 'baseRefName', 'headRefOid', 'baseRefOid', 'files', 'author', 'createdAt', 'updatedAt', 'url', 'resourcePath', 'databaseId'],
  Repository: ['id', 'databaseId', 'name', 'nameWithOwner', 'owner', 'defaultBranchRef', 'viewerPermission', 'hasIssuesEnabled', 'issue', 'issues', 'pullRequests', 'pullRequest', 'description', 'private', 'isPrivate', 'url', 'resourcePath', 'createdAt', 'updatedAt'],
  User: ['login', 'id'],
  Viewer: ['login'],
  Ref: ['name', 'target'],
  IssueConnection: ['nodes', 'pageInfo', 'totalCount'],
  PageInfo: ['hasNextPage', 'hasPreviousPage', 'startCursor', 'endCursor'],
  PullRequestConnection: ['nodes', 'pageInfo', 'totalCount'],
  PullRequestChangedFile: ['path', 'additions', 'deletions', 'changeType'],
  Comment: ['id', 'body', 'author'],
  __Type: ['name', 'kind', 'fields', 'enumValues', 'description'],
  __Field: ['name', 'description'],
  __EnumValue: ['name', 'description'],
  __Schema: ['types', 'queryType', 'mutationType'],
  Query: ['viewer', 'repository', 'node', '__type', '__schema'],
  Mutation: ['createIssue', 'closeIssue', 'createPullRequest', 'mergePullRequest', 'addComment'],
};

const TYPE_ENUM_VALUES: Record<string, readonly string[]> = {
  SearchType: ['COMMIT', 'DISCUSSION', 'ISSUE', 'REPOSITORY', 'USER', 'CODE'],
  IssueState: ['OPEN', 'CLOSED'],
  PullRequestState: ['OPEN', 'CLOSED', 'MERGED'],
  MergeableState: ['MERGEABLE', 'CONFLICTING', 'UNKNOWN'],
  OrderDirection: ['ASC', 'DESC'],
  IssueOrderField: ['CREATED_AT', 'UPDATED_AT', 'COMMENTS'],
  PullRequestMergeMethod: ['MERGE', 'SQUASH', 'REBASE'],
  RepositoryPermission: ['ADMIN', 'MAINTAIN', 'WRITE', 'TRIAGE', 'READ'],
};

const FIXED_TIME = '1970-01-01T00:00:00Z';

// -- global ids ------------------------------------------------------------------

function encodeGlobalId(kind: string, ...parts: (number | string)[]): string {
  return Buffer.from([kind, ...parts.map(String)].join(':'), 'utf8').toString('base64');
}

function decodeGlobalId(id: string): { kind: string; parts: string[] } | null {
  try {
    const text = Buffer.from(id, 'base64').toString('utf8');
    const parts = text.split(':');
    const kind = parts[0];
    if (kind === undefined || kind === '' || parts.length < 2) return null;
    if (text !== [kind, ...parts.slice(1)].join(':')) return null;
    return { kind, parts: parts.slice(1) };
  } catch {
    return null;
  }
}

function repoGlobalId(repo: Repo): string {
  return encodeGlobalId('Repository', repo.id);
}

function issueGlobalId(repo: Repo, issue: Issue): string {
  return encodeGlobalId('Issue', repo.id, issue.number);
}

function pullGlobalId(repo: Repo, pull: Pull): string {
  return encodeGlobalId('PullRequest', repo.id, pull.number);
}

function commentGlobalId(repo: Repo, issue: Issue, commentId: number): string {
  return encodeGlobalId('IssueComment', repo.id, issue.number, commentId);
}

// -- runtime values ---------------------------------------------------------------

type Runtime =
  | { __typename: 'Query' }
  | { __typename: 'Mutation' }
  | { __typename: 'Viewer'; login: string }
  | { __typename: 'User'; login: string }
  | { __typename: 'Repository'; repo: Repo }
  | { __typename: 'Ref'; name: string }
  | { __typename: 'Issue'; repo: Repo; issue: Issue }
  | { __typename: 'IssueConnection'; repo: Repo; issues: Issue[]; pageInfo: PageInfoValue }
  | { __typename: 'PageInfo'; value: PageInfoValue }
  | { __typename: 'PullRequest'; repo: Repo; pull: Pull }
  | { __typename: 'PullRequestConnection'; nodes: Pull[]; repo: Repo; pageInfo: PageInfoValue }
  | { __typename: 'PullRequestChangedFileConnection'; files: { path: string; additions: number; deletions: number; changeType: string }[] }
  | { __typename: 'PullRequestChangedFile'; path: string; additions: number; deletions: number; changeType: string }
  | { __typename: 'Comment'; repo: Repo; issue: Issue; id: number; body: string }
  | { __typename: 'CommentConnection'; repo: Repo; issue: Issue; nodes: { id: number; body: string }[] }
  | { __typename: 'CreateIssuePayload'; repo: Repo; issue: Issue }
  | { __typename: 'CloseIssuePayload'; repo: Repo; issue: Issue }
  | { __typename: 'CreatePullRequestPayload'; repo: Repo; pull: Pull }
  | { __typename: 'MergePullRequestPayload'; repo: Repo; pull: Pull }
  | { __typename: 'AddCommentPayload'; repo: Repo; issue: Issue; commentId: number; commentBody: string }
  | { __typename: 'IssueCommentEdge'; repo: Repo; issue: Issue; commentId: number; commentBody: string }
  | { __typename: 'Label'; name: string; color: string }
  | { __typename: 'LabelConnection'; nodes: { name: string; color: string }[] }
  | { __typename: '__Type'; name: string }
  | { __typename: '__Field'; name: string }
  | { __typename: '__EnumValue'; name: string }
  | { __typename: '__Schema' };

interface PageInfoValue {
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  startCursor: string | null;
  endCursor: string | null;
}

interface GqlContext {
  forge: Forge;
  viewerLogin: string;
  errors: { message: string }[];
}

// -- argument resolution -----------------------------------------------------------

function resolveValue(node: ValueNode, variables: JsonObject): unknown {
  switch (node.kind) {
    case 'variable': {
      const value = variables[node.name];
      return value === undefined ? null : value;
    }
    case 'string':
    case 'int':
    case 'float':
    case 'boolean':
      return node.value;
    case 'null':
      return null;
    case 'enum':
      return node.value;
    case 'list':
      return node.values.map((entry) => resolveValue(entry, variables));
    case 'object': {
      const out: JsonObject = {};
      for (const field of node.fields) out[field.name] = resolveValue(field.value, variables);
      return out;
    }
  }
}

function resolveArgs(field: FieldNode, variables: JsonObject): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const arg of field.args) args[arg.name] = resolveValue(arg.value, variables);
  return args;
}

// -- cursors -----------------------------------------------------------------------

function cursorForNumber(number: number): string {
  return Buffer.from(`cursor:${number}`, 'utf8').toString('base64');
}

function numberFromCursor(cursor: string): number | null {
  try {
    const text = Buffer.from(cursor, 'base64').toString('utf8');
    const match = /^cursor:(\d+)$/.exec(text);
    return match === null ? null : Number.parseInt(match[1] as string, 10);
  } catch {
    return null;
  }
}

// -- line stats (shared shape with REST /pulls/{n}/files) --------------------------

function splitLines(text: string): string[] {
  if (text === '') return [];
  const parts = text.split('\n');
  if (parts.length > 0 && parts[parts.length - 1] === '') parts.pop();
  return parts;
}

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

// -- connections ---------------------------------------------------------------------

function issueConnection(ctx: GqlContext, repo: Repo, args: Record<string, unknown>): Runtime {
  let list = [...repo.issues];
  const states = args.states;
  if (Array.isArray(states)) {
    const wanted = new Set(
      states.map((state) => (state === 'OPEN' ? 'open' : state === 'CLOSED' ? 'closed' : String(state).toLowerCase())),
    );
    list = list.filter((issue) => wanted.has(issue.state));
  }
  const orderBy = isObject(args.orderBy) ? args.orderBy : null;
  const direction = orderBy !== null && orderBy.direction === 'DESC' ? -1 : 1;
  list.sort((a, b) => direction * (a.number - b.number));
  const after = typeof args.after === 'string' && args.after !== '' ? numberFromCursor(args.after) : null;
  if (after !== null) {
    list = direction === 1 ? list.filter((issue) => issue.number > after) : list.filter((issue) => issue.number < after);
  }
  const first = typeof args.first === 'number' && Number.isFinite(args.first) ? Math.max(0, Math.floor(args.first)) : list.length;
  const page = list.slice(0, first);
  const pageInfo: PageInfoValue = {
    hasNextPage: list.length > page.length,
    hasPreviousPage: after !== null,
    startCursor: page.length === 0 ? null : cursorForNumber((page[0] as Issue).number),
    endCursor: page.length === 0 ? null : cursorForNumber((page[page.length - 1] as Issue).number),
  };
  return { __typename: 'IssueConnection', repo, issues: page, pageInfo };
}

function pullFiles(ctx: GqlContext, repo: Repo, pull: Pull, first: number | null): Runtime {
  try {
    if (pull.headSha === null || pull.baseSha === null) return { __typename: 'PullRequestChangedFileConnection', files: [] };
    const mergeBase = ctx.forge.objects.mergeBase(pull.headSha, pull.baseSha);
    const baseTree = mergeBase === null ? null : (ctx.forge.objects.readCommit(mergeBase)?.tree ?? null);
    const headCommit = ctx.forge.objects.readCommit(pull.headSha);
    if (headCommit === undefined) return { __typename: 'PullRequestChangedFileConnection', files: [] };
    const headTree = headCommit?.tree ?? null;
  const diffs = ctx.forge.objects.diffTrees(baseTree, headTree);
  const files = diffs.map((diff) => {
    const oldBlob = diff.oldSha === null ? null : (ctx.forge.objects.readBlob(diff.oldSha) ?? null);
    const newBlob = diff.newSha === null ? null : (ctx.forge.objects.readBlob(diff.newSha) ?? null);
    const stats = lineStats(oldBlob === null ? '' : oldBlob.toString('utf8'), newBlob === null ? '' : newBlob.toString('utf8'));
    return {
      path: diff.path,
      additions: stats.additions,
      deletions: stats.deletions,
      changeType: diff.oldSha === null ? 'ADDED' : diff.newSha === null ? 'DELETED' : 'MODIFIED',
    };
  });
  const page = first === null ? files : files.slice(0, first);
  return { __typename: 'PullRequestChangedFileConnection', files: page };
  } catch {
    return { __typename: 'PullRequestChangedFileConnection', files: [] };
  }
}

function mergeableOf(ctx: GqlContext, repo: Repo, pull: Pull): string {
  try {
    if (pull.headSha === null || pull.baseSha === null) return 'UNKNOWN';
    const headTip = repo.refs.get(`refs/heads/${pull.headRef}`);
    const baseTip = repo.refs.get(`refs/heads/${pull.baseRef}`);
    if (headTip === undefined || baseTip === undefined) return 'UNKNOWN';
    const headCommit = ctx.forge.objects.readCommit(headTip);
    const baseCommit = ctx.forge.objects.readCommit(baseTip);
    if (headCommit === undefined || baseCommit === undefined) return 'UNKNOWN';
    const mergeBase = ctx.forge.objects.mergeBase(headTip, baseTip);
    const baseTree = mergeBase === null ? null : (ctx.forge.objects.readCommit(mergeBase)?.tree ?? null);
    const { conflicts } = ctx.forge.objects.threeWayMerge(baseTree, headCommit.tree, baseCommit.tree);
    return conflicts.length > 0 ? 'CONFLICTING' : 'MERGEABLE';
  } catch {
    return 'UNKNOWN';
  }
}

// -- field resolution ----------------------------------------------------------------

/** Resolve one field on a runtime value; null when the field is null/absent. */
function resolveField(ctx: GqlContext, parent: Runtime, field: FieldNode, args: Record<string, unknown>): unknown {
  const typeName = parent.__typename;
  switch (typeName) {
    case 'Query': {
      if (field.name === 'viewer') return { __typename: 'Viewer', login: ctx.viewerLogin } satisfies Runtime;
      if (field.name === 'repository') {
        const owner = args.owner;
        const name = args.name;
        if (typeof owner !== 'string' || typeof name !== 'string') return null;
        const repo = ctx.forge.getRepo(owner, name);
        return repo === undefined ? null : ({ __typename: 'Repository', repo } satisfies Runtime);
      }
      if (field.name === 'node') {
        if (typeof args.id !== 'string') return null;
        const decoded = decodeGlobalId(args.id);
        if (decoded === null) return null;
        if (decoded.kind === 'Repository') {
          const repoId = Number.parseInt(decoded.parts[0] as string, 10);
          const repo = findRepoById(ctx.forge, repoId);
          return repo === undefined ? null : ({ __typename: 'Repository', repo } satisfies Runtime);
        }
        if (decoded.kind === 'Issue') {
          const repoId = Number.parseInt(decoded.parts[0] as string, 10);
          const number = Number.parseInt(decoded.parts[1] as string, 10);
          const repo = findRepoById(ctx.forge, repoId);
          const issue = repo === undefined ? undefined : ctx.forge.getIssue(repo, number);
          return issue === undefined || repo === undefined
            ? null
            : ({ __typename: 'Issue', repo, issue } satisfies Runtime);
        }
        return null;
      }
      if (field.name === '__type') {
        if (typeof args.name !== 'string') return null;
        return TYPE_FIELDS[args.name] !== undefined || TYPE_ENUM_VALUES[args.name] !== undefined
          ? ({ __typename: '__Type', name: args.name } satisfies Runtime)
          : null;
      }
      if (field.name === '__schema') return { __typename: '__Schema' } satisfies Runtime;
      return null;
    }
    case 'Mutation': {
      if (field.name === 'createIssue') return mutateCreateIssue(ctx, args);
      if (field.name === 'closeIssue') return mutateCloseIssue(ctx, args);
      if (field.name === 'createPullRequest') return mutateCreatePullRequest(ctx, args);
      if (field.name === 'mergePullRequest') return mutateMergePullRequest(ctx, args);
      if (field.name === 'addComment') return mutateAddComment(ctx, args);
      return null;
    }
    case 'Viewer':
      if (field.name === 'login') return parent.login;
      return null;
    case 'User':
      if (field.name === 'login') return parent.login;
      if (field.name === 'id') return 1000;
      return null;
    case 'Repository': {
      const repo = parent.repo;
      switch (field.name) {
        case 'id':
          return repoGlobalId(repo);
        case 'databaseId':
          return repo.id;
        case 'name':
          return repo.name;
        case 'nameWithOwner':
          return `${repo.owner}/${repo.name}`;
        case 'owner':
          return { __typename: 'User', login: repo.owner } satisfies Runtime;
        case 'defaultBranchRef': {
          const tip = repo.refs.get(`refs/heads/${repo.defaultBranch}`);
          return tip === undefined ? null : ({ __typename: 'Ref', name: repo.defaultBranch } satisfies Runtime);
        }
        case 'viewerPermission':
          return 'ADMIN';
        case 'hasIssuesEnabled':
          return true;
        case 'description':
          return repo.description;
        case 'private':
        case 'isPrivate':
          return repo.private;
        case 'url':
          return `https://github.com/${repo.owner}/${repo.name}`;
        case 'resourcePath':
          return `/${repo.owner}/${repo.name}`;
        case 'createdAt':
        case 'updatedAt':
          return FIXED_TIME;
        case 'issue': {
          const number = args.number;
          if (typeof number !== 'number') return null;
          const issue = ctx.forge.getIssue(repo, number);
          return issue === undefined ? null : ({ __typename: 'Issue', repo, issue } satisfies Runtime);
        }
        case 'issues':
          return issueConnection(ctx, repo, args);
        case 'pullRequests': {
          const first = typeof args.first === 'number' ? Math.max(0, Math.floor(args.first)) : repo.pulls.length;
          const page = [...repo.pulls].sort((a, b) => a.number - b.number).slice(0, first);
          return {
            __typename: 'PullRequestConnection',
            nodes: page,
            repo,
            pageInfo: {
              hasNextPage: repo.pulls.length > page.length,
              hasPreviousPage: false,
              startCursor: page.length === 0 ? null : cursorForNumber((page[0] as Pull).number),
              endCursor: page.length === 0 ? null : cursorForNumber((page[page.length - 1] as Pull).number),
            },
          } satisfies Runtime;
        }
        case 'pullRequest': {
          const number = args.number;
          if (typeof number !== 'number') return null;
          const pull = ctx.forge.getPull(repo, number);
          return pull === undefined ? null : ({ __typename: 'PullRequest', repo, pull } satisfies Runtime);
        }
        default:
          return null;
      }
    }
    case 'Ref':
      if (field.name === 'name') return parent.name;
      if (field.name === 'target') return null;
      return null;
    case 'Issue': {
      const { repo, issue } = parent;
      switch (field.name) {
        case 'id':
          return issueGlobalId(repo, issue);
        case 'number':
          return issue.number;
        case 'title':
          return issue.title;
        case 'body':
          return issue.body;
        case 'state':
          return issue.state === 'open' ? 'OPEN' : 'CLOSED';
        case 'author':
          return { __typename: 'User', login: ctx.viewerLogin } satisfies Runtime;
        case 'createdAt':
        case 'updatedAt':
          return FIXED_TIME;
        case 'comments':
          return {
            __typename: 'CommentConnection',
            repo,
            issue,
            nodes: issue.comments.map((comment) => ({ id: comment.id, body: comment.body })),
          } satisfies Runtime;
        case 'labels':
          return {
            __typename: 'LabelConnection',
            nodes: issue.labels.map((name) => ({ name, color: repo.labels.get(name)?.color ?? 'ededed' })),
          } satisfies Runtime;
        case 'url':
          return `https://github.com/${repo.owner}/${repo.name}/issues/${issue.number}`;
        case 'resourcePath':
          return `/${repo.owner}/${repo.name}/issues/${issue.number}`;
        case 'databaseId':
          return issue.number;
        case 'repository':
          return { __typename: 'Repository', repo } satisfies Runtime;
        default:
          return null;
      }
    }
    case 'IssueConnection': {
      if (field.name === 'nodes') return parent.issues.map((issue) => ({ __typename: 'Issue', repo: parent.repo, issue }) satisfies Runtime);
      if (field.name === 'pageInfo') return { __typename: 'PageInfo', value: parent.pageInfo } satisfies Runtime;
      if (field.name === 'totalCount') return parent.issues.length;
      if (field.name === 'edges') return parent.issues.map((issue) => ({ node: { __typename: 'Issue', repo: parent.repo, issue } }));
      return null;
    }
    case 'PageInfo': {
      if (field.name === 'hasNextPage') return parent.value.hasNextPage;
      if (field.name === 'hasPreviousPage') return parent.value.hasPreviousPage;
      if (field.name === 'startCursor') return parent.value.startCursor;
      if (field.name === 'endCursor') return parent.value.endCursor;
      return null;
    }
    case 'PullRequest': {
      const { repo, pull } = parent;
      switch (field.name) {
        case 'id':
          return pullGlobalId(repo, pull);
        case 'number':
          return pull.number;
        case 'title':
          return pull.title;
        case 'body':
          return pull.body;
        case 'state':
          return pull.merged ? 'MERGED' : pull.state === 'open' ? 'OPEN' : 'CLOSED';
        case 'merged':
          return pull.merged;
        case 'mergeable':
          return mergeableOf(ctx, repo, pull);
        case 'headRefName':
          return pull.headRef;
        case 'baseRefName':
          return pull.baseRef;
        case 'headRefOid':
          return pull.headSha;
        case 'baseRefOid':
          return pull.baseSha;
        case 'files': {
          const first = typeof args.first === 'number' ? Math.max(0, Math.floor(args.first)) : null;
          return pullFiles(ctx, repo, pull, first);
        }
        case 'author':
          return { __typename: 'User', login: ctx.viewerLogin } satisfies Runtime;
        case 'createdAt':
        case 'updatedAt':
          return FIXED_TIME;
        case 'url':
          return `https://github.com/${repo.owner}/${repo.name}/pull/${pull.number}`;
        case 'resourcePath':
          return `/${repo.owner}/${repo.name}/pull/${pull.number}`;
        case 'databaseId':
          return pull.number;
        default:
          return null;
      }
    }
    case 'PullRequestConnection': {
      if (field.name === 'nodes') return parent.nodes.map((pull) => ({ __typename: 'PullRequest', repo: parent.repo, pull }) satisfies Runtime);
      if (field.name === 'pageInfo') return { __typename: 'PageInfo', value: parent.pageInfo } satisfies Runtime;
      if (field.name === 'totalCount') return parent.nodes.length;
      return null;
    }
    case 'PullRequestChangedFileConnection': {
      if (field.name === 'nodes') return parent.files.map((file) => ({ __typename: 'PullRequestChangedFile', ...file }) satisfies Runtime);
      if (field.name === 'pageInfo') {
        return {
          __typename: 'PageInfo',
          value: { hasNextPage: false, hasPreviousPage: false, startCursor: null, endCursor: null },
        } satisfies Runtime;
      }
      if (field.name === 'totalCount') return parent.files.length;
      return null;
    }
    case 'PullRequestChangedFile': {
      if (field.name === 'path') return parent.path;
      if (field.name === 'additions') return parent.additions;
      if (field.name === 'deletions') return parent.deletions;
      if (field.name === 'changeType') return parent.changeType;
      return null;
    }
    case 'Comment': {
      if (field.name === 'id') return commentGlobalId(parent.repo, parent.issue, parent.id);
      if (field.name === 'body') return parent.body;
      if (field.name === 'author') return { __typename: 'User', login: ctx.viewerLogin } satisfies Runtime;
      return null;
    }
    case 'CreateIssuePayload':
    case 'CloseIssuePayload': {
      if (field.name === 'issue') return { __typename: 'Issue', repo: parent.repo, issue: parent.issue } satisfies Runtime;
      return null;
    }
    case 'CreatePullRequestPayload':
    case 'MergePullRequestPayload': {
      if (field.name === 'pullRequest') return { __typename: 'PullRequest', repo: parent.repo, pull: parent.pull } satisfies Runtime;
      return null;
    }
    case 'AddCommentPayload': {
      if (field.name === 'commentEdge') {
        return {
          __typename: 'IssueCommentEdge',
          repo: parent.repo,
          issue: parent.issue,
          commentId: parent.commentId,
          commentBody: parent.commentBody,
        } satisfies Runtime;
      }
      return null;
    }
    case 'IssueCommentEdge': {
      if (field.name === 'node') {
        return {
          __typename: 'Comment',
          repo: parent.repo,
          issue: parent.issue,
          id: parent.commentId,
          body: parent.commentBody,
        } satisfies Runtime;
      }
      return null;
    }
    case 'CommentConnection': {
      if (field.name === 'nodes') {
        return parent.nodes.map((node) => ({ __typename: 'Comment', repo: parent.repo, issue: parent.issue, ...node }) satisfies Runtime);
      }
      if (field.name === 'pageInfo') {
        return {
          __typename: 'PageInfo',
          value: { hasNextPage: false, hasPreviousPage: false, startCursor: null, endCursor: null },
        } satisfies Runtime;
      }
      if (field.name === 'totalCount') return parent.nodes.length;
      return null;
    }
    case 'Label': {
      if (field.name === 'name') return parent.name;
      if (field.name === 'color') return parent.color;
      return null;
    }
    case 'LabelConnection': {
      if (field.name === 'nodes') return parent.nodes.map((node) => ({ __typename: 'Label', ...node }) satisfies Runtime);
      return null;
    }
    case '__Type': {
      const name = parent.name;
      if (field.name === 'name') return name;
      if (field.name === 'kind') return TYPE_ENUM_VALUES[name] !== undefined ? 'ENUM' : 'OBJECT';
      if (field.name === 'description') return null;
      if (field.name === 'fields') {
        const list = TYPE_FIELDS[name];
        return list === undefined ? null : list.map((entry) => ({ __typename: '__Field', name: entry }) satisfies Runtime);
      }
      if (field.name === 'enumValues') {
        const list = TYPE_ENUM_VALUES[name];
        return list === undefined ? null : list.map((entry) => ({ __typename: '__EnumValue', name: entry }) satisfies Runtime);
      }
      return null;
    }
    case '__Field':
      if (field.name === 'name') return parent.name;
      if (field.name === 'description') return null;
      return null;
    case '__EnumValue':
      if (field.name === 'name') return parent.name;
      if (field.name === 'description') return null;
      return null;
    case '__Schema': {
      if (field.name === 'types') {
        return [...Object.keys(TYPE_FIELDS), ...Object.keys(TYPE_ENUM_VALUES)].map((name) => ({ __typename: '__Type', name }) satisfies Runtime);
      }
      if (field.name === 'queryType') return { __typename: '__Type', name: 'Query' } satisfies Runtime;
      if (field.name === 'mutationType') return { __typename: '__Type', name: 'Mutation' } satisfies Runtime;
      return null;
    }
    default:
      return null;
  }
}

function findRepoById(forge: Forge, id: number): Repo | undefined {
  return forge.getRepoById(id);
}

// -- mutations ---------------------------------------------------------------------

function inputOf(args: Record<string, unknown>): JsonObject | null {
  return isObject(args.input) ? args.input : null;
}

function mutateCreateIssue(ctx: GqlContext, args: Record<string, unknown>): unknown {
  const input = inputOf(args);
  if (input === null) {
    ctx.errors.push({ message: 'createIssue requires an input object' });
    return null;
  }
  const decoded = typeof input.repositoryId === 'string' ? decodeGlobalId(input.repositoryId) : null;
  if (decoded === null || decoded.kind !== 'Repository') {
    ctx.errors.push({ message: 'createIssue: repositoryId is not a repository global id' });
    return null;
  }
  const repo = findRepoById(ctx.forge, Number.parseInt(decoded.parts[0] as string, 10));
  if (repo === undefined) {
    ctx.errors.push({ message: 'createIssue: repository not found' });
    return null;
  }
  if (typeof input.title !== 'string' || input.title === '') {
    ctx.errors.push({ message: 'createIssue: title is required' });
    return null;
  }
  const issue = ctx.forge.createIssue(repo, input.title, typeof input.body === 'string' ? input.body : '', []);
  return { __typename: 'CreateIssuePayload', repo, issue } satisfies Runtime;
}

function mutateCloseIssue(ctx: GqlContext, args: Record<string, unknown>): unknown {
  const input = inputOf(args);
  if (input === null) {
    ctx.errors.push({ message: 'closeIssue requires an input object' });
    return null;
  }
  const decoded = typeof input.issueId === 'string' ? decodeGlobalId(input.issueId) : null;
  if (decoded === null || decoded.kind !== 'Issue') {
    ctx.errors.push({ message: 'closeIssue: issueId is not an issue global id' });
    return null;
  }
  const repo = findRepoById(ctx.forge, Number.parseInt(decoded.parts[0] as string, 10));
  const issue = repo === undefined ? undefined : ctx.forge.getIssue(repo, Number.parseInt(decoded.parts[1] as string, 10));
  if (repo === undefined || issue === undefined) {
    ctx.errors.push({ message: 'closeIssue: issue not found' });
    return null;
  }
  issue.state = 'closed';
  return { __typename: 'CloseIssuePayload', repo, issue } satisfies Runtime;
}

function mutateCreatePullRequest(ctx: GqlContext, args: Record<string, unknown>): unknown {
  const input = inputOf(args);
  if (input === null) {
    ctx.errors.push({ message: 'createPullRequest requires an input object' });
    return null;
  }
  const decoded = typeof input.repositoryId === 'string' ? decodeGlobalId(input.repositoryId) : null;
  if (decoded === null || decoded.kind !== 'Repository') {
    ctx.errors.push({ message: 'createPullRequest: repositoryId is not a repository global id' });
    return null;
  }
  const repo = findRepoById(ctx.forge, Number.parseInt(decoded.parts[0] as string, 10));
  if (repo === undefined) {
    ctx.errors.push({ message: 'createPullRequest: repository not found' });
    return null;
  }
  const head = input.headRefName;
  const base = input.baseRefName;
  if (typeof head !== 'string' || typeof base !== 'string' || typeof input.title !== 'string' || input.title === '') {
    ctx.errors.push({ message: 'createPullRequest: title, headRefName and baseRefName are required' });
    return null;
  }
  const headSha = ctx.forge.resolveRef(repo, head);
  const baseSha = ctx.forge.resolveRef(repo, base);
  if (headSha === null || baseSha === null) {
    ctx.errors.push({ message: 'createPullRequest: head or base branch not found' });
    return null;
  }
  const pull = ctx.forge.createPull(repo, input.title, typeof input.body === 'string' ? input.body : '', head, base, headSha, baseSha);
  return { __typename: 'CreatePullRequestPayload', repo, pull } satisfies Runtime;
}

function mutateMergePullRequest(ctx: GqlContext, args: Record<string, unknown>): unknown {
  const input = inputOf(args);
  if (input === null) {
    ctx.errors.push({ message: 'mergePullRequest requires an input object' });
    return null;
  }
  const decoded = typeof input.pullRequestId === 'string' ? decodeGlobalId(input.pullRequestId) : null;
  if (decoded === null || decoded.kind !== 'PullRequest') {
    ctx.errors.push({ message: 'mergePullRequest: pullRequestId is not a pull request global id' });
    return null;
  }
  const repo = findRepoById(ctx.forge, Number.parseInt(decoded.parts[0] as string, 10));
  const pull = repo === undefined ? undefined : ctx.forge.getPull(repo, Number.parseInt(decoded.parts[1] as string, 10));
  if (repo === undefined || pull === undefined) {
    ctx.errors.push({ message: 'mergePullRequest: pull request not found' });
    return null;
  }
  const result = ctx.forge.squashMerge(repo, pull);
  if (!result.ok) {
    ctx.errors.push({ message: `mergePullRequest: cannot merge (${result.reason})` });
    return null;
  }
  return { __typename: 'MergePullRequestPayload', repo, pull } satisfies Runtime;
}

function mutateAddComment(ctx: GqlContext, args: Record<string, unknown>): unknown {
  const input = inputOf(args);
  if (input === null) {
    ctx.errors.push({ message: 'addComment requires an input object' });
    return null;
  }
  const decoded = typeof input.subjectId === 'string' ? decodeGlobalId(input.subjectId) : null;
  if (decoded === null || decoded.kind !== 'Issue') {
    ctx.errors.push({ message: 'addComment: subjectId is not an issue global id' });
    return null;
  }
  const repo = findRepoById(ctx.forge, Number.parseInt(decoded.parts[0] as string, 10));
  const issue = repo === undefined ? undefined : ctx.forge.getIssue(repo, Number.parseInt(decoded.parts[1] as string, 10));
  if (repo === undefined || issue === undefined) {
    ctx.errors.push({ message: 'addComment: issue not found' });
    return null;
  }
  if (typeof input.body !== 'string' || input.body === '') {
    ctx.errors.push({ message: 'addComment: body is required' });
    return null;
  }
  const comment = ctx.forge.addComment(repo, issue, input.body);
  return {
    __typename: 'AddCommentPayload',
    repo,
    issue,
    commentId: comment.id,
    commentBody: comment.body,
  } satisfies Runtime;
}

// -- executor ----------------------------------------------------------------------

function isRuntime(value: unknown): value is Runtime {
  return isObject(value) && typeof value.__typename === 'string';
}

function executeSelectionSet(
  ctx: GqlContext,
  selections: SelectionNode[],
  parentType: string,
  parent: Runtime,
  variables: JsonObject,
): JsonObject {
  const out: JsonObject = {};
  const fields = FIELDS[parentType];
  for (const selection of selections) {
    if (selection.kind === 'inlineFragment') {
      if (selection.typeCondition !== null && selection.typeCondition !== parentType) continue;
      const nested = executeSelectionSet(ctx, selection.selections, parentType, parent, variables);
      for (const [key, value] of Object.entries(nested)) out[key] = value;
      continue;
    }
    const field = selection;
    const responseKey = field.alias ?? field.name;
    if (field.name === '__typename') {
      out[responseKey] = parentType;
      continue;
    }
    if (fields === undefined || !fields.has(field.name)) {
      ctx.errors.push({ message: `Cannot query field "${field.name}" on type "${parentType}".` });
      out[responseKey] = null;
      continue;
    }
    const knownArgs = FIELD_ARGS[parentType]?.[field.name];
    let badArg: string | null = null;
    for (const arg of field.args) {
      if (knownArgs !== undefined && !knownArgs.has(arg.name)) {
        badArg = arg.name;
        break;
      }
    }
    if (badArg !== null) {
      ctx.errors.push({ message: `Unknown argument "${badArg}" on field "${parentType}.${field.name}".` });
      out[responseKey] = null;
      continue;
    }
    const args = resolveArgs(field, variables);
    const value = resolveField(ctx, parent, field, args);
    out[responseKey] = completeValue(ctx, field, value, variables);
  }
  return out;
}

function completeValue(ctx: GqlContext, field: FieldNode, value: unknown, variables: JsonObject): unknown {
  if (value === null || value === undefined) return null;
  if (Array.isArray(value)) {
    if (field.selections === null) return value;
    return value.map((entry) => {
      if (!isRuntime(entry)) return entry;
      return executeSelectionSet(ctx, field.selections as SelectionNode[], entry.__typename, entry, variables);
    });
  }
  if (isRuntime(value)) {
    if (field.selections === null) {
      ctx.errors.push({ message: `Field "${field.name}" of type "${value.__typename}" must have a selection of subfields.` });
      return null;
    }
    return executeSelectionSet(ctx, field.selections, value.__typename, value, variables);
  }
  // A scalar with sub-selections is a validation error.
  if (field.selections !== null) {
    ctx.errors.push({ message: `Field "${field.name}" must not have a selection since type is a scalar.` });
    return null;
  }
  return value;
}

function executeOperation(ctx: GqlContext, operation: OperationNode, variables: JsonObject): JsonObject {
  const rootType = operation.operation === 'mutation' ? 'Mutation' : 'Query';
  const root: Runtime = rootType === 'Mutation' ? { __typename: 'Mutation' } : { __typename: 'Query' };
  const data = executeSelectionSet(ctx, operation.selections, rootType, root, variables);
  if (ctx.errors.length > 0) return { data: null, errors: ctx.errors };
  return { data };
}

function executeOne(forge: Forge, viewerLogin: string, body: unknown): JsonObject {
  const ctx: GqlContext = { forge, viewerLogin, errors: [] };
  if (!isObject(body) || typeof body.query !== 'string') {
    return { data: null, errors: [{ message: 'A GraphQL query string is required' }] };
  }
  const variables = isObject(body.variables) ? body.variables : {};
  let operations: OperationNode[];
  try {
    operations = parseDocument(body.query);
  } catch (err) {
    return { data: null, errors: [{ message: err instanceof Error ? err.message : 'GraphQL syntax error' }] };
  }
  const operationName = typeof body.operationName === 'string' ? body.operationName : null;
  let operation: OperationNode | undefined;
  if (operationName !== null) {
    operation = operations.find((entry) => entry.name === operationName);
    if (operation === undefined) {
      return { data: null, errors: [{ message: `Unknown operation named "${operationName}".` }] };
    }
  } else {
    operation = operations[0];
  }
  if (operation === undefined) return { data: null, errors: [{ message: 'No operation in the GraphQL document' }] };
  if (operation.operation === 'subscription') {
    return { data: null, errors: [{ message: 'Subscriptions are not supported' }] };
  }
  // Variable defaults, for variables the caller did not send.
  for (const definition of operation.variableDefinitions) {
    if (!(definition.name in variables) && definition.defaultValue !== null) {
      variables[definition.name] = resolveValue(definition.defaultValue, variables);
    }
  }
  try {
    return executeOperation(ctx, operation, variables);
  } catch (err) {
    return { data: null, errors: [{ message: err instanceof Error ? err.message : 'GraphQL execution failed' }] };
  }
}

/** Run one operation or a JSON batch of operations. Always HTTP 200 on the wire. */
export function executeGraphQL(forge: Forge, viewerLogin: string, body: unknown): { status: number; body: unknown } {
  if (Array.isArray(body)) {
    return { status: 200, body: body.map((entry) => executeOne(forge, viewerLogin, entry)) };
  }
  return { status: 200, body: executeOne(forge, viewerLogin, body) };
}
