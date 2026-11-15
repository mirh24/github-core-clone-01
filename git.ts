// A minimal git object store: blobs, trees and commits, all deterministic.
//
// Every object is content-addressed by its SHA-1, exactly as git hashes it:
//   blob   -> sha1("blob <len>\0" + bytes)
//   tree   -> sha1("tree <len>\0" + entries), each entry "<mode> <name>\0" + 20 raw bytes
//   commit -> sha1("commit <len>\0" + "tree <sha>\n" + "parent <sha>\n"* + fixed
//             author/committer lines + "\n" + message)
//
// Author and committer are fixed constants (epoch 0), never the wall clock, so the
// same requests always build the same objects and the same state export byte for byte.
// This store is the foundation for M1 (contents/branches/commits over REST) and M2
// (packfiles and smart HTTP, which read the same objects).

import { createHash } from 'node:crypto';

export function sha1(data: Buffer): string {
  return createHash('sha1').update(data).digest('hex');
}

function hashObject(type: GitObjectType, data: Buffer): string {
  const header = Buffer.from(`${type} ${data.length}\0`, 'utf8');
  return sha1(Buffer.concat([header, data]));
}

export type GitObjectType = 'blob' | 'tree' | 'commit' | 'tag';

export interface GitObject {
  type: GitObjectType;
  data: Buffer;
}

export interface TreeEntry {
  mode: string;
  name: string;
  sha: string;
  isTree: boolean;
}

export interface TagData {
  object: string;
  type: GitObjectType;
  tag: string;
  tagger: string;
  message: string;
}

export interface TreeDiffEntry {
  path: string;
  oldSha: string | null;
  newSha: string | null;
}

export interface CommitData {
  tree: string;
  parents: string[];
  message: string;
}

/** Fixed identity for every commit this forge writes: deterministic, no clock. */
const AUTHOR_LINE = 'author Muse <muse@make.fun> 0 +0000';
const COMMITTER_LINE = 'committer Muse <muse@make.fun> 0 +0000';

export class ObjectStore {
  readonly objects = new Map<string, GitObject>();

  put(type: GitObjectType, data: Buffer): string {
    const sha = hashObject(type, data);
    if (!this.objects.has(sha)) this.objects.set(sha, { type, data });
    return sha;
  }

  get(sha: string): GitObject | undefined {
    return this.objects.get(sha.toLowerCase());
  }

  // -- blobs -----------------------------------------------------------------

  writeBlob(content: Buffer): string {
    return this.put('blob', content);
  }

  readBlob(sha: string): Buffer | undefined {
    const object = this.get(sha);
    return object !== undefined && object.type === 'blob' ? object.data : undefined;
  }

  // -- trees -----------------------------------------------------------------

  /** Serialize entries git-style: sorted with directories as name + '/', mode + space + name + NUL + 20 raw sha bytes. */
  writeTree(entries: readonly TreeEntry[]): string {
    const sorted = [...entries].sort((a, b) => {
      const ka = a.name + (a.isTree ? '/' : '');
      const kb = b.name + (b.isTree ? '/' : '');
      return ka < kb ? -1 : ka > kb ? 1 : 0;
    });
    const parts: Buffer[] = [];
    for (const entry of sorted) {
      parts.push(Buffer.from(`${entry.mode} ${entry.name}\0`, 'utf8'));
      parts.push(Buffer.from(entry.sha, 'hex'));
    }
    return this.put('tree', Buffer.concat(parts));
  }

  readTree(sha: string): TreeEntry[] | undefined {
    const object = this.get(sha);
    if (object === undefined || object.type !== 'tree') return undefined;
    const data = object.data;
    const entries: TreeEntry[] = [];
    let offset = 0;
    while (offset < data.length) {
      const space = data.indexOf(0x20, offset);
      const nul = data.indexOf(0x00, space + 1);
      if (space === -1 || nul === -1 || nul + 21 > data.length) return undefined;
      const mode = data.subarray(offset, space).toString('utf8');
      const name = data.subarray(space + 1, nul).toString('utf8');
      const sha = data.subarray(nul + 1, nul + 21).toString('hex');
      entries.push({ mode, name, sha, isTree: mode === '40000' });
      offset = nul + 21;
    }
    return entries;
  }

  /** All files under a tree, as full paths -> blob sha. */
  flattenTree(treeSha: string, prefix = '', out = new Map<string, string>()): Map<string, string> {
    const entries = this.readTree(treeSha);
    if (entries === undefined) return out;
    for (const entry of entries) {
      const path = prefix === '' ? entry.name : `${prefix}/${entry.name}`;
      if (entry.isTree) this.flattenTree(entry.sha, path, out);
      else out.set(path, entry.sha);
    }
    return out;
  }

  /** Build nested trees from a flat path -> blob sha map. */
  buildTree(files: ReadonlyMap<string, string>): string {
    const direct: TreeEntry[] = [];
    const subdirs = new Map<string, Map<string, string>>();
    for (const [path, blobSha] of files) {
      const slash = path.indexOf('/');
      if (slash === -1) {
        direct.push({ mode: '100644', name: path, sha: blobSha, isTree: false });
      } else {
        const dir = path.slice(0, slash);
        const rest = path.slice(slash + 1);
        let sub = subdirs.get(dir);
        if (sub === undefined) {
          sub = new Map();
          subdirs.set(dir, sub);
        }
        sub.set(rest, blobSha);
      }
    }
    for (const [dir, sub] of subdirs) {
      direct.push({ mode: '40000', name: dir, sha: this.buildTree(sub), isTree: true });
    }
    return this.writeTree(direct);
  }

  /** A new root tree with `path` set to `blobSha` (or removed when null). */
  setPath(rootTreeSha: string | null, path: string, blobSha: string | null): string {
    const files = rootTreeSha === null ? new Map<string, string>() : this.flattenTree(rootTreeSha);
    if (blobSha === null) files.delete(path);
    else files.set(path, blobSha);
    return this.buildTree(files);
  }

  /** Read one path under a tree: a file, a directory listing, or null when absent. */
  readPath(
    rootTreeSha: string,
    path: string,
  ): { kind: 'file'; sha: string } | { kind: 'dir'; entries: TreeEntry[] } | null {
    if (path === '') {
      const entries = this.readTree(rootTreeSha);
      return entries === undefined ? null : { kind: 'dir', entries };
    }
    const segments = path.split('/');
    let treeSha = rootTreeSha;
    for (let i = 0; i < segments.length; i += 1) {
      const entries = this.readTree(treeSha);
      if (entries === undefined) return null;
      const entry = entries.find((candidate) => candidate.name === segments[i]);
      if (entry === undefined) return null;
      if (i === segments.length - 1) {
        if (entry.isTree) {
          const children = this.readTree(entry.sha);
          return children === undefined ? null : { kind: 'dir', entries: children };
        }
        return { kind: 'file', sha: entry.sha };
      }
      if (!entry.isTree) return null;
      treeSha = entry.sha;
    }
    return null;
  }

  // -- commits ---------------------------------------------------------------

  writeCommit(treeSha: string, parents: readonly string[], message: string): string {
    const text =
      `tree ${treeSha}\n` +
      parents.map((parent) => `parent ${parent}\n`).join('') +
      `${AUTHOR_LINE}\n${COMMITTER_LINE}\n\n${message}`;
    return this.put('commit', Buffer.from(text, 'utf8'));
  }

  readCommit(sha: string): CommitData | undefined {
    const object = this.get(sha);
    if (object === undefined || object.type !== 'commit') return undefined;
    const text = object.data.toString('utf8');
    const split = text.indexOf('\n\n');
    const header = split === -1 ? text : text.slice(0, split);
    const message = split === -1 ? '' : text.slice(split + 2);
    const tree = /^tree ([0-9a-f]{40})$/m.exec(header)?.[1];
    if (tree === undefined) return undefined;
    const parents: string[] = [];
    for (const match of header.matchAll(/^parent ([0-9a-f]{40})$/gm)) {
      if (match[1] !== undefined) parents.push(match[1]);
    }
    return { tree, parents, message };
  }

  /** Commit shas reachable from `fromSha`, newest first (linear histories: the chain). */
  walkCommits(fromSha: string): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    const stack: string[] = [fromSha];
    while (stack.length > 0) {
      const sha = stack.pop() as string;
      if (seen.has(sha)) continue;
      seen.add(sha);
      const commit = this.readCommit(sha);
      if (commit === undefined) continue;
      out.push(sha);
      for (let i = commit.parents.length - 1; i >= 0; i -= 1) {
        stack.push(commit.parents[i] as string);
      }
    }
    return out;
  }

  // -- tags ------------------------------------------------------------------

  writeTag(data: Buffer): string {
    return this.put('tag', data);
  }

  readTag(sha: string): TagData | undefined {
    const object = this.get(sha);
    if (object === undefined || object.type !== 'tag') return undefined;
    const text = object.data.toString('utf8');
    const split = text.indexOf('\n\n');
    const header = split === -1 ? text : text.slice(0, split);
    const message = split === -1 ? '' : text.slice(split + 2);
    let target = '';
    let type: GitObjectType = 'commit';
    let tag = '';
    let tagger = '';
    for (const line of header.split('\n')) {
      if (line.startsWith('object ')) target = line.slice(7);
      else if (line.startsWith('type ')) {
        const value = line.slice(5);
        if (value === 'blob' || value === 'tree' || value === 'commit' || value === 'tag') type = value;
      } else if (line.startsWith('tag ')) tag = line.slice(4);
      else if (line.startsWith('tagger ')) tagger = line.slice(7);
    }
    if (!/^[0-9a-f]{40}$/.test(target)) return undefined;
    return { object: target, type, tag, tagger, message };
  }

  /** If `sha` names an annotated tag, the commit it ultimately points at (tags may chain). */
  peelTag(sha: string): string | null {
    let current = sha;
    for (let hops = 0; hops < 16; hops += 1) {
      const tag = this.readTag(current);
      if (tag === undefined) return null;
      current = tag.object;
      if (this.readCommit(current) !== undefined) return current;
    }
    return null;
  }

  /**
   * Every object reachable from `roots`: commit -> tree + parents, tag -> its
   * target, tree -> entries (gitlinks skipped), blob -> nothing. Missing
   * objects are skipped, never an error: the store is closed under every ref
   * tip, but a defensive walk must not throw on a gap.
   */
  collectReachable(roots: readonly string[]): Set<string> {
    const seen = new Set<string>();
    const queue: string[] = [];
    for (const root of roots) {
      if (/^[0-9a-f]{40}$/i.test(root)) queue.push(root.toLowerCase());
    }
    while (queue.length > 0) {
      const oid = queue.pop() as string;
      if (seen.has(oid)) continue;
      const object = this.get(oid);
      if (object === undefined) continue;
      seen.add(oid);
      if (object.type === 'commit') {
        const commit = this.readCommit(oid);
        if (commit === undefined) continue;
        queue.push(commit.tree);
        for (const parent of commit.parents) queue.push(parent);
      } else if (object.type === 'tag') {
        const tag = this.readTag(oid);
        if (tag !== undefined) queue.push(tag.object);
      } else if (object.type === 'tree') {
        const entries = this.readTree(oid);
        if (entries === undefined) continue;
        for (const entry of entries) {
          if (entry.mode !== '160000') queue.push(entry.sha);
        }
      }
    }
    return seen;
  }

  /** True when `ancestor` is reachable from `descendant` by following commit parents. */
  isAncestor(ancestor: string, descendant: string): boolean {
    if (ancestor.toLowerCase() === descendant.toLowerCase()) return true;
    return this.walkCommits(descendant).includes(ancestor.toLowerCase());
  }

  /** The merge base of two commits: the first commit in a newest-first walk of `b` that is also an ancestor of `a` (null when the histories share nothing). */
  mergeBase(aSha: string, bSha: string): string | null {
    const ancestorsA = new Set(this.walkCommits(aSha));
    for (const sha of this.walkCommits(bSha)) {
      if (ancestorsA.has(sha)) return sha;
    }
    return null;
  }

  /** The paths whose blob differs between two trees (null tree = empty). */
  diffTrees(baseTreeSha: string | null, headTreeSha: string | null): TreeDiffEntry[] {
    const base = baseTreeSha === null ? new Map<string, string>() : this.flattenTree(baseTreeSha);
    const head = headTreeSha === null ? new Map<string, string>() : this.flattenTree(headTreeSha);
    const paths = new Set([...base.keys(), ...head.keys()]);
    const out: TreeDiffEntry[] = [];
    for (const path of [...paths].sort()) {
      const oldSha = base.get(path) ?? null;
      const newSha = head.get(path) ?? null;
      if (oldSha !== newSha) out.push({ path, oldSha, newSha });
    }
    return out;
  }

  /**
   * A three-way merge of file maps: `base` is the merge base, `head` the branch
   * being merged, `other` the branch it merges into. A path lands in `files`
   * when both sides agree, or when only one side changed it; it lands in
   * `conflicts` when both sides changed it differently since the base.
   */
  threeWayMerge(
    baseTreeSha: string | null,
    headTreeSha: string | null,
    otherTreeSha: string | null,
  ): { files: Map<string, string>; conflicts: string[] } {
    const base = baseTreeSha === null ? new Map<string, string>() : this.flattenTree(baseTreeSha);
    const head = headTreeSha === null ? new Map<string, string>() : this.flattenTree(headTreeSha);
    const other = otherTreeSha === null ? new Map<string, string>() : this.flattenTree(otherTreeSha);
    const files = new Map<string, string>();
    const conflicts: string[] = [];
    const paths = new Set([...base.keys(), ...head.keys(), ...other.keys()]);
    for (const path of paths) {
      const b = base.get(path) ?? null;
      const h = head.get(path) ?? null;
      const o = other.get(path) ?? null;
      if (h === o) {
        if (h !== null) files.set(path, h);
      } else if (h === b) {
        if (o !== null) files.set(path, o);
      } else if (o === b) {
        if (h !== null) files.set(path, h);
      } else {
        conflicts.push(path);
      }
    }
    return { files, conflicts };
  }
}
