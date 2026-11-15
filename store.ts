// The forge's store: repositories, refs, issues, labels and comments over the
// git object store (git.ts). In-memory only: the entry is a fresh server each
// trial and the checks create everything they need through the API, so there is
// nothing to persist. Every id and number is a deterministic counter (creation
// order); no clock, no randomness anywhere in the store.

import { ObjectStore } from './git.ts';

export interface Label {
  name: string;
  color: string;
}

export interface Comment {
  id: number;
  body: string;
}

export interface Issue {
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  labels: string[];
  comments: Comment[];
}

export interface Pull {
  number: number;
  title: string;
  body: string;
  /** Branch name as given at creation (e.g. "feature-x"), not a full ref. */
  headRef: string;
  baseRef: string;
  /** Branch tips at creation time. */
  headSha: string;
  baseSha: string;
  state: 'open' | 'closed';
  merged: boolean;
  mergedAt: string | null;
  mergeCommitSha: string | null;
  /** Creation order, for stable list ordering. */
  seq: number;
}

export interface Repo {
  id: number;
  owner: string;
  name: string;
  private: boolean;
  description: string;
  defaultBranch: string;
  /** Full ref name (refs/heads/<branch>) -> commit sha. */
  refs: Map<string, string>;
  issues: Issue[];
  nextIssue: number;
  labels: Map<string, Label>;
  nextCommentId: number;
  pulls: Pull[];
  nextPull: number;
  /** Creation order, for stable list ordering. */
  seq: number;
}

/** Fixed merged_at for every squash merge: deterministic, never the wall clock. */
export const MERGED_AT = '1970-01-01T00:00:00Z';

export const DEFAULT_BRANCH = 'main';

export class Forge {
  readonly objects = new ObjectStore();
  private readonly owners = new Map<string, Map<string, Repo>>();
  private nextRepoId = 1;
  private nextRepoSeq = 1;

  getRepo(owner: string, name: string): Repo | undefined {
    return this.owners.get(owner)?.get(name);
  }

  /** Find a repository by its numeric id, across owners. */
  getRepoById(id: number): Repo | undefined {
    for (const byName of this.owners.values()) {
      for (const repo of byName.values()) {
        if (repo.id === id) return repo;
      }
    }
    return undefined;
  }

  createRepo(owner: string, name: string, isPrivate: boolean, description: string): Repo | undefined {
    let byName = this.owners.get(owner);
    if (byName === undefined) {
      byName = new Map();
      this.owners.set(owner, byName);
    }
    if (byName.has(name)) return undefined;
    const repo: Repo = {
      id: this.nextRepoId,
      owner,
      name,
      private: isPrivate,
      description,
      defaultBranch: DEFAULT_BRANCH,
      refs: new Map(),
      issues: [],
      nextIssue: 1,
      labels: new Map(),
      nextCommentId: 1,
      pulls: [],
      nextPull: 1,
      seq: this.nextRepoSeq,
    };
    this.nextRepoId += 1;
    this.nextRepoSeq += 1;
    byName.set(name, repo);
    return repo;
  }

  deleteRepo(owner: string, name: string): boolean {
    const byName = this.owners.get(owner);
    if (byName === undefined) return false;
    return byName.delete(name);
  }

  /** The viewer's repositories, in creation order. */
  listRepos(owner: string): Repo[] {
    const byName = this.owners.get(owner);
    if (byName === undefined) return [];
    return [...byName.values()].sort((a, b) => a.seq - b.seq);
  }

  /** Resolve a ref name, a branch name or a commit sha to a commit sha, or null. */
  resolveRef(repo: Repo, ref: string | undefined): string | null {
    if (ref === undefined || ref === '') {
      return repo.refs.get(`refs/heads/${repo.defaultBranch}`) ?? null;
    }
    const asRef = repo.refs.get(ref);
    if (asRef !== undefined) return asRef;
    const asBranch = repo.refs.get(`refs/heads/${ref}`);
    if (asBranch !== undefined) return asBranch;
    if (/^[0-9a-f]{40}$/i.test(ref) && this.objects.readCommit(ref) !== undefined) return ref.toLowerCase();
    return null;
  }

  listBranches(repo: Repo): { name: string; sha: string }[] {
    const out: { name: string; sha: string }[] = [];
    for (const [ref, sha] of repo.refs) {
      if (ref.startsWith('refs/heads/')) out.push({ name: ref.slice('refs/heads/'.length), sha });
    }
    out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return out;
  }

  /** Commit one file's bytes on `branch`; creates the branch ref when the repo is unborn. */
  commitFile(
    repo: Repo,
    branch: string,
    path: string,
    content: Buffer,
    message: string,
  ): { commitSha: string; blobSha: string } | null {
    const ref = `refs/heads/${branch}`;
    const tip = repo.refs.get(ref);
    if (tip === undefined && repo.refs.size > 0) return null;
    const blobSha = this.objects.writeBlob(content);
    const baseTree = tip === undefined ? null : (this.objects.readCommit(tip)?.tree ?? null);
    const newTree = this.objects.setPath(baseTree, path, blobSha);
    const commitSha = this.objects.writeCommit(newTree, tip === undefined ? [] : [tip], message);
    repo.refs.set(ref, commitSha);
    return { commitSha, blobSha };
  }

  /** Remove one file on `branch`, committing the removal. */
  deleteFile(repo: Repo, branch: string, path: string, message: string): { commitSha: string } | null {
    const ref = `refs/heads/${branch}`;
    const tip = repo.refs.get(ref);
    if (tip === undefined) return null;
    const commit = this.objects.readCommit(tip);
    if (commit === undefined) return null;
    const newTree = this.objects.setPath(commit.tree, path, null);
    const commitSha = this.objects.writeCommit(newTree, [tip], message);
    repo.refs.set(ref, commitSha);
    return { commitSha };
  }

  /** The blob sha at `path` on `branch`, or null when the path is not a file there. */
  fileShaAt(repo: Repo, branch: string, path: string): string | null {
    const tip = repo.refs.get(`refs/heads/${branch}`);
    if (tip === undefined) return null;
    const commit = this.objects.readCommit(tip);
    if (commit === undefined) return null;
    const found = this.objects.readPath(commit.tree, path);
    return found !== null && found.kind === 'file' ? found.sha : null;
  }

  createIssue(repo: Repo, title: string, body: string, labels: string[]): Issue {
    const issue: Issue = {
      number: repo.nextIssue,
      title,
      body,
      state: 'open',
      labels: [...labels],
      comments: [],
    };
    repo.nextIssue += 1;
    repo.issues.push(issue);
    return issue;
  }

  getIssue(repo: Repo, number: number): Issue | undefined {
    return repo.issues.find((issue) => issue.number === number);
  }

  addComment(repo: Repo, issue: Issue, body: string): Comment {
    const comment: Comment = { id: repo.nextCommentId, body };
    repo.nextCommentId += 1;
    issue.comments.push(comment);
    return comment;
  }

  createPull(repo: Repo, title: string, body: string, headRef: string, baseRef: string, headSha: string, baseSha: string): Pull {
    const pull: Pull = {
      number: repo.nextPull,
      title,
      body,
      headRef,
      baseRef,
      headSha,
      baseSha,
      state: 'open',
      merged: false,
      mergedAt: null,
      mergeCommitSha: null,
      seq: repo.nextPull,
    };
    repo.nextPull += 1;
    repo.pulls.push(pull);
    return pull;
  }

  getPull(repo: Repo, number: number): Pull | undefined {
    return repo.pulls.find((pull) => pull.number === number);
  }

  /**
   * Squash-merge an open pull request onto its base branch: one new commit on
   * the base branch whose tree is the three-way merge of head into base. The
   * merge commit has the base tip as its only parent (a squash, not a merge
   * commit with two parents). Returns the merge commit sha, or the reason the
   * merge cannot happen: already merged, a conflict, or a missing ref.
   */
  squashMerge(
    repo: Repo,
    pull: Pull,
  ): { ok: true; commitSha: string } | { ok: false; reason: 'already-merged' | 'conflict' | 'missing-ref' } {
    if (pull.state === 'closed') return { ok: false, reason: 'already-merged' };
    const headTip = repo.refs.get(`refs/heads/${pull.headRef}`);
    const baseTip = repo.refs.get(`refs/heads/${pull.baseRef}`);
    if (headTip === undefined || baseTip === undefined) return { ok: false, reason: 'missing-ref' };
    const headCommit = this.objects.readCommit(headTip);
    const baseCommit = this.objects.readCommit(baseTip);
    if (headCommit === undefined || baseCommit === undefined) return { ok: false, reason: 'missing-ref' };
    const mergeBase = this.objects.mergeBase(headTip, baseTip);
    const baseTree = mergeBase === null ? null : (this.objects.readCommit(mergeBase)?.tree ?? null);
    const { files, conflicts } = this.objects.threeWayMerge(baseTree, headCommit.tree, baseCommit.tree);
    if (conflicts.length > 0) return { ok: false, reason: 'conflict' };
    const mergedTree = this.objects.buildTree(files);
    const commitSha = this.objects.writeCommit(mergedTree, [baseTip], `Merge pull request #${pull.number}: ${pull.title}`);
    repo.refs.set(`refs/heads/${pull.baseRef}`, commitSha);
    pull.state = 'closed';
    pull.merged = true;
    pull.mergedAt = MERGED_AT;
    pull.mergeCommitSha = commitSha;
    return { ok: true, commitSha };
  }
}
