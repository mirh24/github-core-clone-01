// Git smart-HTTP (protocol v2 + v0) and packfile encode/decode for the forge.
//
// This module speaks the wire the kit's git client speaks (see
// github-core-clone-01-kit/sdk/conformance/git-client.ts for the exact
// expectations) on top of the object store in git.ts:
//
//   - info/refs advertisements: v2 (`version 2` + capability lines) for
//     git-upload-pack when the client sends `Git-Protocol: version=2`, v0
//     (`# service=...` preamble + ref lines) otherwise and always for
//     git-receive-pack.
//   - v2 ls-refs (peel / symrefs / unborn / ref-prefix, server-side filtering).
//   - v2 and v0 fetch: want/have negotiation, shallow/deepen, packfile
//     generation (plain objects, zlib, SHA-1 trailer) framed in sideband.
//   - receive-pack: command list + packfile parse (OFS/REF delta application),
//     per-ref fast-forward validation, report-status in sideband band 1.
//
// Determinism: object order in a generated pack is sorted by oid; no clock,
// no randomness anywhere. A missing repository answers HTTP 404 on info/refs;
// a malformed request body answers HTTP 400 and never crashes the server.

import { createHash } from 'node:crypto';
import { deflateSync, inflateSync } from 'node:zlib';
import {
  ObjectStore,
  type GitObject,
  type GitObjectType,
  sha1,
} from './git.ts';
import type { Forge, Repo } from './store.ts';

export const ZERO_OID = '0'.repeat(40);
const AGENT = 'makefun-forge/1.0';
const PKT_LINE_MAX = 65520;
const PKT_LINE_MAX_PAYLOAD = PKT_LINE_MAX - 4;

export class GitHttpError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'GitHttpError';
    this.status = status;
  }
}

// -- pkt-line ------------------------------------------------------------------

export function pktLine(payload: Buffer | string): Buffer {
  const data = typeof payload === 'string' ? Buffer.from(payload, 'utf8') : payload;
  if (data.length > PKT_LINE_MAX_PAYLOAD) {
    throw new GitHttpError(400, `pkt-line payload is ${data.length} bytes, over the maximum`);
  }
  const header = Buffer.allocUnsafe(4);
  header.write((data.length + 4).toString(16).padStart(4, '0'), 0, 4, 'ascii');
  return Buffer.concat([header, data]);
}

export const pktFlush = (): Buffer => Buffer.from('0000', 'ascii');
export const pktDelim = (): Buffer => Buffer.from('0001', 'ascii');

export function pktLines(items: ReadonlyArray<Buffer | string | 'flush' | 'delim'>): Buffer {
  const parts: Buffer[] = [];
  for (const item of items) {
    if (item === 'flush') parts.push(pktFlush());
    else if (item === 'delim') parts.push(pktDelim());
    else parts.push(pktLine(item));
  }
  return Buffer.concat(parts);
}

const HEX4 = /^[0-9a-fA-F]{4}$/;

type DecodedPacket =
  | { kind: 'data'; payload: Buffer }
  | { kind: 'flush' }
  | { kind: 'delim' }
  | { kind: 'response-end' };

/** Decode a whole buffer of pkt-lines. Throws GitHttpError(400) on bad framing. */
function decodePackets(bytes: Buffer): DecodedPacket[] {
  const packets: DecodedPacket[] = [];
  let offset = 0;
  while (offset < bytes.length) {
    if (offset + 4 > bytes.length) throw new GitHttpError(400, 'truncated pkt-line header');
    const header = bytes.subarray(offset, offset + 4).toString('ascii');
    if (!HEX4.test(header)) throw new GitHttpError(400, `invalid pkt-line length ${JSON.stringify(header)}`);
    const length = Number.parseInt(header, 16);
    if (length === 0) {
      packets.push({ kind: 'flush' });
      offset += 4;
      continue;
    }
    if (length === 1) {
      packets.push({ kind: 'delim' });
      offset += 4;
      continue;
    }
    if (length === 2) {
      packets.push({ kind: 'response-end' });
      offset += 4;
      continue;
    }
    if (length === 3 || length > PKT_LINE_MAX) throw new GitHttpError(400, `invalid pkt-line length ${length}`);
    if (offset + length > bytes.length) throw new GitHttpError(400, 'truncated pkt-line');
    packets.push({ kind: 'data', payload: bytes.subarray(offset + 4, offset + length) });
    offset += length;
  }
  return packets;
}

/** Split a request body into the pkt-line command section and the raw bytes after its flush. */
function splitCommandSection(body: Buffer): { packets: DecodedPacket[]; rest: Buffer } {
  let offset = 0;
  const packets: DecodedPacket[] = [];
  while (offset < body.length) {
    if (offset + 4 > body.length) throw new GitHttpError(400, 'truncated pkt-line header');
    const header = body.subarray(offset, offset + 4).toString('ascii');
    if (!HEX4.test(header)) throw new GitHttpError(400, `invalid pkt-line length ${JSON.stringify(header)}`);
    const length = Number.parseInt(header, 16);
    if (length === 0) {
      packets.push({ kind: 'flush' });
      return { packets, rest: body.subarray(offset + 4) };
    }
    if (length === 1) {
      packets.push({ kind: 'delim' });
      offset += 4;
      continue;
    }
    if (length === 2) {
      packets.push({ kind: 'response-end' });
      offset += 4;
      continue;
    }
    if (length === 3 || length > PKT_LINE_MAX) throw new GitHttpError(400, `invalid pkt-line length ${length}`);
    if (offset + length > body.length) throw new GitHttpError(400, 'truncated pkt-line');
    packets.push({ kind: 'data', payload: body.subarray(offset + 4, offset + length) });
    offset += length;
  }
  throw new GitHttpError(400, 'request has no terminating flush');
}

// -- packfile write --------------------------------------------------------------

const TYPE_CODE: Readonly<Record<GitObjectType, number>> = { commit: 1, tree: 2, blob: 3, tag: 4 };

function packObjectHeader(type: GitObjectType, size: number): Buffer {
  const code = TYPE_CODE[type];
  const bytes: number[] = [];
  let remaining = size;
  let first = (code << 4) | (remaining & 0x0f);
  remaining = Math.floor(remaining / 16);
  while (remaining > 0) {
    bytes.push(first | 0x80);
    first = remaining & 0x7f;
    remaining = Math.floor(remaining / 128);
  }
  bytes.push(first);
  return Buffer.from(bytes);
}

/** Build a version-2 packfile from plain objects (no deltas). Objects are emitted in oid order. */
export function buildPack(objects: ReadonlyArray<{ type: GitObjectType; data: Buffer }>): Buffer {
  // Hash each object once up front: hashing inside the comparator re-hashes both
  // sides on every comparison (O(n log n) hashes for an n-object pack).
  const withOid = objects.map((object) => ({
    object,
    oid: sha1(Buffer.concat([Buffer.from(`${object.type} ${object.data.length}\0`, 'utf8'), object.data])),
  }));
  withOid.sort((a, b) => (a.oid < b.oid ? -1 : a.oid > b.oid ? 1 : 0));
  const sorted = withOid.map((entry) => entry.object);
  const parts: Buffer[] = [];
  const header = Buffer.allocUnsafe(12);
  header.write('PACK', 0, 4, 'ascii');
  header.writeUInt32BE(2, 4);
  header.writeUInt32BE(sorted.length, 8);
  parts.push(header);
  for (const object of sorted) {
    parts.push(packObjectHeader(object.type, object.data.length));
    parts.push(deflateSync(object.data));
  }
  const body = Buffer.concat(parts);
  return Buffer.concat([body, createHash('sha1').update(body).digest()]);
}

// -- delta application -----------------------------------------------------------

function readDeltaVarint(delta: Buffer, offset: number): { value: number; next: number } {
  let value = 0;
  let shift = 0;
  let at = offset;
  for (;;) {
    if (at >= delta.length) throw new GitHttpError(400, 'delta size varint is truncated');
    const byte = delta[at];
    at += 1;
    value += (byte & 0x7f) * 2 ** shift;
    shift += 7;
    if (!(byte & 0x80)) break;
    if (shift > 63) throw new GitHttpError(400, 'delta size varint is too long');
  }
  return { value, next: at };
}

/** Apply a git delta (copy/insert instruction stream) to a base object's bytes. */
export function applyDelta(base: Buffer, delta: Buffer): Buffer {
  const source = readDeltaVarint(delta, 0);
  const target = readDeltaVarint(delta, source.next);
  if (source.value !== base.length) {
    throw new GitHttpError(400, `delta expects a ${source.value}-byte base, got ${base.length}`);
  }
  const out = Buffer.allocUnsafe(target.value);
  let written = 0;
  let offset = target.next;
  while (offset < delta.length) {
    const op = delta[offset];
    offset += 1;
    if (op & 0x80) {
      let copyOffset = 0;
      let copySize = 0;
      let shift = 0;
      for (let bit = 0; bit < 4; bit += 1) {
        if (op & (1 << bit)) {
          if (offset >= delta.length) throw new GitHttpError(400, 'delta copy offset is truncated');
          copyOffset += delta[offset] * 2 ** shift;
          offset += 1;
        }
        shift += 8;
      }
      shift = 0;
      for (let bit = 4; bit < 7; bit += 1) {
        if (op & (1 << bit)) {
          if (offset >= delta.length) throw new GitHttpError(400, 'delta copy size is truncated');
          copySize += delta[offset] * 2 ** shift;
          offset += 1;
        }
        shift += 8;
      }
      if (copySize === 0) copySize = 0x10000;
      if (copyOffset + copySize > base.length) throw new GitHttpError(400, 'delta copy is out of the base');
      if (written + copySize > target.value) throw new GitHttpError(400, 'delta writes past its target size');
      base.copy(out, written, copyOffset, copyOffset + copySize);
      written += copySize;
    } else if (op !== 0) {
      const literal = op & 0x7f;
      if (offset + literal > delta.length) throw new GitHttpError(400, 'delta insert is truncated');
      if (written + literal > target.value) throw new GitHttpError(400, 'delta writes past its target size');
      delta.copy(out, written, offset, offset + literal);
      offset += literal;
      written += literal;
    } else {
      throw new GitHttpError(400, 'delta opcode 0 is invalid');
    }
  }
  if (written !== target.value) throw new GitHttpError(400, `delta produced ${written} of ${target.value} bytes`);
  return out;
}

// -- packfile read ---------------------------------------------------------------

function readPackVarint(buffer: Buffer, start: number): { size: number; next: number; first: number } {
  if (start >= buffer.length) throw new GitHttpError(400, 'pack entry has no size header');
  let byte = buffer[start];
  const first = byte;
  let size = byte & 0x0f;
  let shift = 4;
  let offset = start + 1;
  while (byte & 0x80) {
    if (offset >= buffer.length) throw new GitHttpError(400, 'pack entry size varint is truncated');
    byte = buffer[offset];
    offset += 1;
    size += (byte & 0x7f) * 2 ** shift;
    shift += 7;
    if (shift > 63) throw new GitHttpError(400, 'pack entry size varint is too long');
  }
  return { size, next: offset, first };
}

function readOfsDeltaBase(buffer: Buffer, start: number): { baseOffset: number; next: number } {
  if (start >= buffer.length) throw new GitHttpError(400, 'ofs-delta has no base offset');
  let byte = buffer[start];
  let offset = start + 1;
  let value = byte & 0x7f;
  while (byte & 0x80) {
    if (offset >= buffer.length) throw new GitHttpError(400, 'ofs-delta base offset is truncated');
    byte = buffer[offset];
    offset += 1;
    value = (value + 1) * 128 + (byte & 0x7f);
  }
  return { baseOffset: value, next: offset };
}

const CODE_TYPE: Readonly<Record<number, GitObjectType>> = { 1: 'commit', 2: 'tree', 3: 'blob', 4: 'tag' };

/**
 * Parse a packfile (version 2 or 3) and resolve every object, including
 * OFS_DELTA and REF_DELTA (bases may be in the pack or, for a thin pack, in
 * `store`). Every object's SHA-1 is recomputed from its bytes; a mismatch, a
 * bad trailer or a truncated stream is an unpack error (HTTP 400).
 */
export function parsePack(pack: Buffer, store: ObjectStore): Map<string, GitObject> {
  if (pack.length < 32) throw new GitHttpError(400, `pack is ${pack.length} bytes, too short`);
  if (pack.subarray(0, 4).toString('ascii') !== 'PACK') throw new GitHttpError(400, 'pack does not start with PACK');
  const version = pack.readUInt32BE(4);
  if (version !== 2 && version !== 3) throw new GitHttpError(400, `unsupported pack version ${version}`);
  const declaredCount = pack.readUInt32BE(8);
  const trailerStart = pack.length - 20;
  const trailer = pack.subarray(trailerStart).toString('hex');
  const computedTrailer = createHash('sha1').update(pack.subarray(0, trailerStart)).digest('hex');
  if (trailer !== computedTrailer) throw new GitHttpError(400, 'pack trailer does not match its bytes');

  interface RawEntry {
    offset: number;
    typeCode: number;
    size: number;
    data: Buffer;
    baseOffset?: number;
    baseOid?: string;
    resolvedType?: GitObjectType;
    resolved?: Buffer;
    oid?: string;
  }

  const raw: RawEntry[] = [];
  const byOffset = new Map<number, RawEntry>();
  let offset = 12;
  for (let index = 0; index < declaredCount; index += 1) {
    const entryOffset = offset;
    const header = readPackVarint(pack, offset);
    offset = header.next;
    const typeCode = (header.first >> 4) & 0x07;
    const entry: RawEntry = { offset: entryOffset, typeCode, size: header.size, data: Buffer.alloc(0) };
    if (typeCode === 6) {
      const base = readOfsDeltaBase(pack, offset);
      offset = base.next;
      if (base.baseOffset <= 0 || base.baseOffset > entryOffset) {
        throw new GitHttpError(400, `ofs-delta at ${entryOffset} names an invalid base offset ${base.baseOffset}`);
      }
      entry.baseOffset = entryOffset - base.baseOffset;
    } else if (typeCode === 7) {
      if (offset + 20 > trailerStart) throw new GitHttpError(400, 'ref-delta base id is truncated');
      entry.baseOid = pack.subarray(offset, offset + 20).toString('hex');
      offset += 20;
    } else if (CODE_TYPE[typeCode] === undefined) {
      throw new GitHttpError(400, `pack entry ${index} has unknown type code ${typeCode}`);
    }
    let inflated: { buffer: Buffer; engine: { bytesWritten: number } };
    try {
      inflated = inflateSync(pack.subarray(offset, trailerStart), { info: true }) as unknown as {
        buffer: Buffer;
        engine: { bytesWritten: number };
      };
    } catch (err) {
      throw new GitHttpError(400, `zlib stream at offset ${offset} did not inflate: ${(err as Error).message}`);
    }
    if (inflated.buffer.length !== header.size) {
      throw new GitHttpError(
        400,
        `pack entry at ${entryOffset} declares ${header.size} bytes but inflated to ${inflated.buffer.length}`,
      );
    }
    entry.data = inflated.buffer;
    offset += inflated.engine.bytesWritten;
    raw.push(entry);
    byOffset.set(entryOffset, entry);
  }
  if (offset !== trailerStart) throw new GitHttpError(400, `pack has ${trailerStart - offset} stray bytes before its trailer`);

  const objects = new Map<string, GitObject>();
  const resolvedByOid = new Map<string, RawEntry>();
  const resolving = new Set<number>();

  const resolve = (entry: RawEntry): { type: GitObjectType; data: Buffer; oid: string } => {
    if (entry.resolved !== undefined && entry.resolvedType !== undefined && entry.oid !== undefined) {
      return { type: entry.resolvedType, data: entry.resolved, oid: entry.oid };
    }
    if (resolving.has(entry.offset)) throw new GitHttpError(400, `delta cycle at pack offset ${entry.offset}`);
    resolving.add(entry.offset);
    let type: GitObjectType;
    let data: Buffer;
    if (entry.typeCode === 6) {
      const baseEntry = byOffset.get(entry.baseOffset as number);
      if (baseEntry === undefined) throw new GitHttpError(400, `ofs-delta base at offset ${entry.baseOffset} is not in the pack`);
      const base = resolve(baseEntry);
      type = base.type;
      data = applyDelta(base.data, entry.data);
    } else if (entry.typeCode === 7) {
      const wanted = entry.baseOid as string;
      const baseEntry = resolvedByOid.get(wanted);
      if (baseEntry !== undefined) {
        const base = resolve(baseEntry);
        type = base.type;
        data = applyDelta(base.data, entry.data);
      } else {
        const external = store.get(wanted);
        if (external === undefined) throw new GitHttpError(400, `ref-delta base ${wanted} is neither in the pack nor in the store`);
        type = external.type;
        data = applyDelta(external.data, entry.data);
      }
    } else {
      type = CODE_TYPE[entry.typeCode] as GitObjectType;
      data = entry.data;
    }
    const header = Buffer.from(`${type} ${data.length}\0`, 'utf8');
    const oid = sha1(Buffer.concat([header, data]));
    entry.resolved = data;
    entry.resolvedType = type;
    entry.oid = oid;
    resolvedByOid.set(oid, entry);
    resolving.delete(entry.offset);
    return { type, data, oid };
  };

  for (const entry of raw) {
    const resolved = resolve(entry);
    if (!objects.has(resolved.oid)) objects.set(resolved.oid, { type: resolved.type, data: resolved.data });
  }
  return objects;
}

// -- sideband ------------------------------------------------------------------

/** Wrap bytes in sideband band 1 (data) pkt-lines, chunked under the pkt-line maximum. */
export function sidebandData(data: Buffer): Buffer {
  const parts: Buffer[] = [];
  let offset = 0;
  while (offset < data.length) {
    const chunk = data.subarray(offset, offset + PKT_LINE_MAX_PAYLOAD - 1);
    parts.push(pktLine(Buffer.concat([Buffer.from([1]), chunk])));
    offset += chunk.length;
  }
  return Buffer.concat(parts);
}

/** One sideband progress packet (band 2). */
function sidebandProgress(text: string): Buffer {
  return pktLine(Buffer.concat([Buffer.from([2]), Buffer.from(text, 'utf8')]));
}

// -- refs ------------------------------------------------------------------------

interface RefLine {
  name: string;
  oid: string;
}

/** Every ref in the repo, sorted by name. */
function listRefs(repo: Repo): RefLine[] {
  return [...repo.refs.entries()]
    .map(([name, oid]) => ({ name, oid }))
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

function headTarget(repo: Repo): string {
  return `refs/heads/${repo.defaultBranch}`;
}

/** The peeled commit for an annotated tag ref, or null when the ref is not one. */
function peeledCommit(objects: ObjectStore, oid: string): string | null {
  return objects.peelTag(oid);
}

// -- advertisements ---------------------------------------------------------------

export type GitService = 'git-upload-pack' | 'git-receive-pack';

export interface GitResponse {
  status: number;
  contentType: string;
  body: Buffer;
}

function v0AdvertisementWithStore(forge: Forge, repo: Repo, service: GitService): Buffer {
  const parts: Buffer[] = [];
  const preamble = service === 'git-upload-pack' ? '# service=git-upload-pack\n' : '# service=git-receive-pack\n';
  parts.push(pktLine(preamble));
  parts.push(pktFlush());
  const caps =
    service === 'git-upload-pack'
      ? `symref=HEAD:${headTarget(repo)} side-band-64k thin-pack ofs-delta multi_ack shallow`
      : `symref=HEAD:${headTarget(repo)} report-status side-band-64k delete-refs ofs-delta agent=${AGENT}`;
  const refs = listRefs(repo);
  if (refs.length === 0) {
    parts.push(pktLine(`${ZERO_OID} capabilities^{}\0${caps}\n`));
  } else {
    const head = refs.find((ref) => ref.name === headTarget(repo));
    const firstOid = head !== undefined ? head.oid : refs[0].oid;
    const firstName = head !== undefined ? 'HEAD' : refs[0].name;
    parts.push(pktLine(`${firstOid} ${firstName}\0${caps}\n`));
    // Every ref gets its own line, including refs/heads/<default>: the client
    // looks refs/heads/main up by name, and HEAD alone does not satisfy it.
    for (const ref of refs) {
      parts.push(pktLine(`${ref.oid} ${ref.name}\n`));
      const peeled = peeledCommit(forge.objects, ref.oid);
      if (peeled !== null) parts.push(pktLine(`${peeled} ${ref.name}^{}\n`));
    }
  }
  parts.push(pktFlush());
  return Buffer.concat(parts);
}

function v2Advertisement(): Buffer {
  return pktLines(['version 2\n', 'ls-refs=unborn\n', 'fetch=shallow\n', 'object-format=sha1\n', `agent=${AGENT}\n`, 'flush']);
}

/** GET /{owner}/{repo}.git/info/refs?service=... — 404 when the repo does not exist. */
export function handleInfoRefs(
  forge: Forge,
  owner: string,
  name: string,
  service: GitService,
  wantV2: boolean,
): GitResponse {
  const repo = forge.getRepo(owner, name);
  if (repo === undefined) throw new GitHttpError(404, 'Not Found');
  if (service === 'git-receive-pack' || !wantV2) {
    return {
      status: 200,
      contentType: `application/x-${service}-advertisement`,
      body: v0AdvertisementWithStore(forge, repo, service),
    };
  }
  return { status: 200, contentType: 'application/x-git-upload-pack-advertisement', body: v2Advertisement() };
}

// -- ls-refs (v2) ------------------------------------------------------------------

/**
 * POST /{owner}/{repo}.git/git-upload-pack with `command=ls-refs`. Returns the
 * ref lines the client asked for; ref-prefix filters server-side, so HEAD is
 * only reported when it matches a prefix (or none was given).
 */
export function handleLsRefs(forge: Forge, owner: string, name: string, body: Buffer): GitResponse {
  const repo = forge.getRepo(owner, name);
  if (repo === undefined) throw new GitHttpError(404, 'Not Found');
  const packets = decodePackets(body);
  const lines: string[] = [];
  let afterDelim = false;
  for (const packet of packets) {
    if (packet.kind === 'delim') {
      afterDelim = true;
      continue;
    }
    if (packet.kind !== 'data') continue;
    if (afterDelim) lines.push(packet.payload.toString('utf8').replace(/\n$/, ''));
  }
  let peel = false;
  let symrefs = false;
  let unborn = false;
  const prefixes: string[] = [];
  for (const line of lines) {
    if (line === 'peel') peel = true;
    else if (line === 'symrefs') symrefs = true;
    else if (line === 'unborn') unborn = true;
    else if (line.startsWith('ref-prefix ')) prefixes.push(line.slice('ref-prefix '.length));
  }
  const keep = (refName: string): boolean => prefixes.length === 0 || prefixes.some((prefix) => refName.startsWith(prefix));
  const out: Array<Buffer | string | 'flush'> = [];
  const refs = listRefs(repo);
  const headOid = repo.refs.get(headTarget(repo));
  if (refs.length === 0) {
    if (unborn && keep('HEAD')) {
      out.push(`unborn HEAD${symrefs ? ` symref-target:${headTarget(repo)}` : ''}\n`);
    }
  } else {
    if (keep('HEAD')) {
      const oid = headOid ?? ZERO_OID;
      out.push(`${oid} HEAD${symrefs ? ` symref-target:${headTarget(repo)}` : ''}\n`);
    }
    for (const ref of refs) {
      if (!keep(ref.name)) continue;
      let line = `${ref.oid} ${ref.name}`;
      if (peel) {
        const peeled = peeledCommit(forge.objects, ref.oid);
        if (peeled !== null) line += ` peeled:${peeled}`;
      }
      out.push(`${line}\n`);
    }
  }
  out.push('flush');
  return { status: 200, contentType: 'application/x-git-upload-pack-result', body: pktLines(out) };
}

// -- fetch -------------------------------------------------------------------------

interface FetchRequest {
  wants: string[];
  haves: string[];
  shallow: string[];
  depth: number | null;
  done: boolean;
}

function parseFetchRequest(body: Buffer): FetchRequest {
  const packets = decodePackets(body);
  const lines: string[] = [];
  let afterDelim = false;
  for (const packet of packets) {
    if (packet.kind === 'delim') {
      afterDelim = true;
      continue;
    }
    if (packet.kind !== 'data') continue;
    if (afterDelim) lines.push(packet.payload.toString('utf8').replace(/\n$/, ''));
  }
  const request: FetchRequest = { wants: [], haves: [], shallow: [], depth: null, done: false };
  for (const line of lines) {
    if (line.startsWith('want ')) request.wants.push(line.slice(5).trim());
    else if (line.startsWith('have ')) request.haves.push(line.slice(5).trim());
    else if (line === 'done') request.done = true;
    else if (line.startsWith('shallow ')) request.shallow.push(line.slice(8).trim());
    else if (line.startsWith('deepen ')) {
      const n = Number.parseInt(line.slice(7), 10);
      if (Number.isSafeInteger(n) && n >= 0) request.depth = n;
    }
  }
  return request;
}

/**
 * The objects a fetch must send, and the shallow boundary it must report.
 *
 * held: everything reachable from the client's haves, with the walk cut at the
 * client's current shallow boundaries (a boundary commit is held, its parents
 * are not). want: the closure of the wants, or only `depth` generations of
 * commits when a depth was asked. send = want - held. A commit in want whose
 * parents are not in want is a new shallow boundary; an old boundary whose
 * parents are now all in want is unshallow.
 */
function planFetch(
  objects: ObjectStore,
  request: FetchRequest,
): { send: GitObject[]; shallow: string[]; unshallow: string[]; acks: string[] } {
  const boundaries = new Set(request.shallow.map((oid) => oid.toLowerCase()));
  const isOid = (value: string): boolean => /^[0-9a-f]{40}$/i.test(value);

  // Held set: walk from the haves, stopping at boundary commits.
  const held = new Set<string>();
  {
    const queue: string[] = [];
    for (const have of request.haves) if (isOid(have)) queue.push(have.toLowerCase());
    while (queue.length > 0) {
      const oid = queue.pop() as string;
      if (held.has(oid)) continue;
      const object = objects.get(oid);
      if (object === undefined) continue;
      held.add(oid);
      if (object.type === 'commit') {
        const commit = objects.readCommit(oid);
        if (commit !== undefined) {
          for (const treeOid of objects.collectReachable([commit.tree])) held.add(treeOid);
          if (!boundaries.has(oid)) {
            for (const parent of commit.parents) queue.push(parent);
          }
        }
      } else if (object.type === 'tag') {
        const tag = objects.readTag(oid);
        if (tag !== undefined) queue.push(tag.object);
      }
    }
  }

  // Want set: full closure, or depth-limited generations of commits.
  const want = new Set<string>();
  const wantCommits = new Set<string>();
  {
    if (request.depth === null) {
      for (const oid of objects.collectReachable(request.wants.filter(isOid))) want.add(oid);
      for (const oid of want) {
        if (objects.get(oid)?.type === 'commit') wantCommits.add(oid);
      }
    } else {
      const depth = request.depth;
      const queue: Array<{ oid: string; gen: number }> = [];
      for (const wantOid of request.wants) if (isOid(wantOid)) queue.push({ oid: wantOid.toLowerCase(), gen: 0 });
      while (queue.length > 0) {
        const { oid, gen } = queue.shift() as { oid: string; gen: number };
        if (want.has(oid)) continue;
        const object = objects.get(oid);
        if (object === undefined) continue;
        if (object.type === 'commit') {
          if (gen >= depth) continue;
          want.add(oid);
          wantCommits.add(oid);
          const commit = objects.readCommit(oid);
          if (commit !== undefined) {
            for (const treeOid of objects.collectReachable([commit.tree])) want.add(treeOid);
            for (const parent of commit.parents) queue.push({ oid: parent, gen: gen + 1 });
          }
        } else if (object.type === 'tag') {
          want.add(oid);
          const tag = objects.readTag(oid);
          if (tag !== undefined) queue.push({ oid: tag.object, gen });
        } else {
          want.add(oid);
        }
      }
    }
  }

  const send: GitObject[] = [];
  for (const oid of want) {
    if (held.has(oid)) continue;
    const object = objects.get(oid);
    if (object !== undefined) send.push(object);
  }

  const shallow: string[] = [];
  for (const oid of wantCommits) {
    const commit = objects.readCommit(oid);
    if (commit === undefined) continue;
    if (commit.parents.some((parent) => !wantCommits.has(parent))) shallow.push(oid);
  }
  const unshallow: string[] = [];
  for (const boundary of boundaries) {
    if (!wantCommits.has(boundary)) continue;
    const commit = objects.readCommit(boundary);
    if (commit !== undefined && commit.parents.every((parent) => wantCommits.has(parent))) unshallow.push(boundary);
  }

  const acks: string[] = [];
  for (const have of request.haves) {
    if (isOid(have) && objects.get(have.toLowerCase()) !== undefined) acks.push(have.toLowerCase());
  }
  return { send, shallow, unshallow, acks };
}

function sidebandPack(pack: Buffer): Buffer {
  return pack.length === 0 ? Buffer.alloc(0) : sidebandData(pack);
}

/** POST git-upload-pack: a v2 fetch or an ls-refs command, or a v0 fetch. */
export function handleUploadPack(
  forge: Forge,
  owner: string,
  name: string,
  body: Buffer,
  wantV2: boolean,
): GitResponse {
  const repo = forge.getRepo(owner, name);
  if (repo === undefined) throw new GitHttpError(404, 'Not Found');
  const packets = decodePackets(body);
  const firstLine = packets.find((packet) => packet.kind === 'data')?.payload.toString('utf8').replace(/\n$/, '') ?? '';
  if (firstLine.startsWith('command=ls-refs')) return handleLsRefs(forge, owner, name, body);

  // A fetch: v2 when the client asked for it, v0 otherwise.
  const request = wantV2 ? parseFetchRequest(body) : parseV0FetchRequest(body);
  const plan = planFetch(forge.objects, request);
  const pack = buildPack(plan.send);

  if (wantV2) {
    const parts: Buffer[] = [];
    // The ack section is conditional, and real git is strict about it:
    //  - request carries `done` (fresh clone, or the kit client which always
    //    sends it): NO ack section — straight to shallow-info/packfile.
    //    Sending one makes real git die "expected 'packfile'".
    //  - request has haves but NO `done` (real git's incremental
    //    negotiation round): the client expects `acknowledgments` first —
    //    ACK for each common have, `ready` when a common base exists, then
    //    delim, then the packfile in the SAME response. Omitting it makes
    //    real git die "expected 'acknowledgments'".
    if (!request.done && request.haves.length > 0) {
      parts.push(pktLine('acknowledgments\n'));
      if (plan.acks.length === 0) parts.push(pktLine('NAK\n'));
      else {
        for (const ack of plan.acks) parts.push(pktLine(`ACK ${ack}\n`));
        parts.push(pktLine('ready\n'));
      }
      parts.push(pktDelim());
    }
    if (plan.shallow.length > 0 || plan.unshallow.length > 0) {
      parts.push(pktLine('shallow-info\n'));
      for (const oid of plan.shallow) parts.push(pktLine(`shallow ${oid}\n`));
      for (const oid of plan.unshallow) parts.push(pktLine(`unshallow ${oid}\n`));
      parts.push(pktDelim());
    }
    parts.push(pktLine('packfile\n'));
    parts.push(sidebandPack(pack));
    parts.push(pktFlush());
    return { status: 200, contentType: 'application/x-git-upload-pack-result', body: Buffer.concat(parts) };
  }

  // v0: plain ACK/NAK + shallow lines, then the sideband pack.
  const parts: Buffer[] = [];
  if (plan.acks.length === 0) parts.push(pktLine('NAK\n'));
  else for (const ack of plan.acks) parts.push(pktLine(`ACK ${ack}\n`));
  for (const oid of plan.shallow) parts.push(pktLine(`shallow ${oid}\n`));
  for (const oid of plan.unshallow) parts.push(pktLine(`unshallow ${oid}\n`));
  parts.push(sidebandPack(pack));
  parts.push(pktFlush());
  return { status: 200, contentType: 'application/x-git-upload-pack-result', body: Buffer.concat(parts) };
}

/** A v0 fetch request: want lines (first carries capabilities) up to a flush, then have/done lines. */
function parseV0FetchRequest(body: Buffer): FetchRequest {
  const packets = decodePackets(body);
  const request: FetchRequest = { wants: [], haves: [], shallow: [], depth: null, done: true };
  for (const packet of packets) {
    if (packet.kind !== 'data') continue;
    const line = packet.payload.toString('utf8').replace(/\n$/, '');
    if (line.startsWith('want ')) {
      const rest = line.slice(5);
      const space = rest.indexOf(' ');
      request.wants.push(space < 0 ? rest.trim() : rest.slice(0, space));
    } else if (line.startsWith('have ')) request.haves.push(line.slice(5).trim());
    else if (line === 'done') request.done = true;
    else if (line.startsWith('shallow ')) request.shallow.push(line.slice(8).trim());
    else if (line.startsWith('deepen ')) {
      const n = Number.parseInt(line.slice(7), 10);
      if (Number.isSafeInteger(n) && n >= 0) request.depth = n;
    }
  }
  return request;
}

// -- receive-pack (push) ---------------------------------------------------------

interface PushCommand {
  oldOid: string;
  newOid: string;
  ref: string;
}

interface RefReport {
  ref: string;
  ok: boolean;
  reason: string;
}

/**
 * POST git-receive-pack: a v0 command list + packfile. Objects are unpacked
 * and stored first; each ref update is then validated (create / delete /
 * fast-forward) and only the valid ones are applied. The report-status stream
 * travels in sideband band 1.
 */
export function handleReceivePack(forge: Forge, owner: string, name: string, body: Buffer): GitResponse {
  const repo = forge.getRepo(owner, name);
  if (repo === undefined) throw new GitHttpError(404, 'Not Found');
  const { packets, rest: packBytes } = splitCommandSection(body);
  const commands: PushCommand[] = [];
  let clientCapabilities = '';
  for (const packet of packets) {
    if (packet.kind !== 'data') continue;
    let line = packet.payload.toString('utf8').replace(/\n$/, '');
    const nul = line.indexOf('\0');
    if (nul >= 0) {
      if (commands.length === 0) clientCapabilities = line.slice(nul + 1);
      line = line.slice(0, nul);
    }
    const parts = line.split(' ');
    if (parts.length !== 3) throw new GitHttpError(400, `bad push command ${JSON.stringify(line.slice(0, 80))}`);
    const [oldOid, newOid, ref] = parts as [string, string, string];
    if (!/^[0-9a-f]{40}$/i.test(oldOid) || !/^[0-9a-f]{40}$/i.test(newOid) || !ref.startsWith('refs/')) {
      throw new GitHttpError(400, `bad push command ${JSON.stringify(line.slice(0, 80))}`);
    }
    commands.push({ oldOid: oldOid.toLowerCase(), newOid: newOid.toLowerCase(), ref });
  }
  if (commands.length === 0) throw new GitHttpError(400, 'push has no commands');

  const sideband = clientCapabilities.split(' ').includes('side-band-64k');
  const reportLines: string[] = [];
  let unpackOk = true;
  let unpackDetail = 'ok';

  // Unpack first: a pack is required whenever a command creates or updates a ref.
  const needsPack = commands.some((command) => command.newOid !== ZERO_OID);
  if (needsPack) {
    if (packBytes.length === 0) {
      unpackOk = false;
      unpackDetail = 'unpacker error';
    } else {
      try {
        const objects = parsePack(packBytes, forge.objects);
        for (const [oid, object] of objects) {
          if (forge.objects.get(oid) === undefined) forge.objects.put(object.type, object.data);
        }
      } catch (err) {
        unpackOk = false;
        unpackDetail = err instanceof GitHttpError ? err.message : (err as Error).message;
      }
    }
  }

  const reports: RefReport[] = [];
  if (unpackOk) {
    for (const command of commands) {
      const current = repo.refs.get(command.ref);
      if (command.newOid === ZERO_OID) {
        if (current === undefined) reports.push({ ref: command.ref, ok: false, reason: 'no such ref' });
        else if (command.oldOid !== ZERO_OID && command.oldOid !== current) {
          reports.push({ ref: command.ref, ok: false, reason: 'old value mismatch' });
        } else {
          repo.refs.delete(command.ref);
          reports.push({ ref: command.ref, ok: true, reason: '' });
        }
      } else if (command.oldOid === ZERO_OID) {
        if (current !== undefined) {
          reports.push({ ref: command.ref, ok: false, reason: 'already exists' });
        } else if (forge.objects.get(command.newOid) === undefined) {
          reports.push({ ref: command.ref, ok: false, reason: 'object not found' });
        } else {
          repo.refs.set(command.ref, command.newOid);
          reports.push({ ref: command.ref, ok: true, reason: '' });
        }
      } else {
        if (current === undefined) {
          reports.push({ ref: command.ref, ok: false, reason: 'no such ref' });
        } else if (command.oldOid !== current) {
          reports.push({ ref: command.ref, ok: false, reason: 'old value mismatch' });
        } else if (forge.objects.get(command.newOid) === undefined) {
          reports.push({ ref: command.ref, ok: false, reason: 'object not found' });
        } else {
          const target = forge.objects.get(command.newOid);
          if (target !== undefined && target.type === 'commit' && !forge.objects.isAncestor(command.oldOid, command.newOid)) {
            reports.push({ ref: command.ref, ok: false, reason: 'non-fast-forward' });
          } else {
            repo.refs.set(command.ref, command.newOid);
            reports.push({ ref: command.ref, ok: true, reason: '' });
          }
        }
      }
    }
  }

  reportLines.push(`unpack ${unpackDetail}\n`);
  for (const report of reports) {
    reportLines.push(report.ok ? `ok ${report.ref}\n` : `ng ${report.ref} ${report.reason}\n`);
  }
  const reportBody = pktLines([...reportLines, 'flush']);
  let responseBody: Buffer;
  if (sideband) {
    responseBody = Buffer.concat([sidebandData(reportBody), sidebandProgress('unpacking objects\n'), pktFlush()]);
  } else {
    responseBody = Buffer.concat([reportBody, pktFlush()]);
  }
  return { status: 200, contentType: 'application/x-git-receive-pack-result', body: responseBody };
}
