// Reads NumPy .npy / .npz files in the browser.
//
// The spectral-library-match viewer preloads two .npz artifacts written by
// _spectral_library_match_run.py - the anomaly pixels' spectra and the matched
// library spectra - so every hover and click in the chart is served locally
// with no api round-trip per pixel. This is the reader for them.
//
// Why hand-rolled: an .npz is just a ZIP of .npy members, both formats are
// small and stable, and the frontend deliberately carries no zip or ndarray
// dependency. See docs/07-frontend.md on the dependency-light rule.
//
// Limitations, all fine for the artifacts we actually produce:
//   - ZIP64 (>4 GB) is not handled.
//   - Big-endian and object arrays are rejected rather than mis-read.
//   - int64 comes back as BigInt64Array; the callers use int32.

export interface NpyArray {
  data: ArrayBufferView;
  shape: number[];
  dtype: string;
}

export type Npz = Record<string, NpyArray>;

type Ctor = new (buf: ArrayBuffer) => ArrayBufferView;

// NumPy dtype string -> TypedArray. Keyed without the byte-order character,
// which is checked separately.
const DTYPES: Record<string, Ctor> = {
  f4: Float32Array,
  f8: Float64Array,
  i1: Int8Array,
  i2: Int16Array,
  i4: Int32Array,
  i8: BigInt64Array as unknown as Ctor,
  u1: Uint8Array,
  u2: Uint16Array,
  u4: Uint32Array,
  u8: BigUint64Array as unknown as Ctor,
  b1: Uint8Array, // numpy bool_ is one byte per element
};

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

/** Parse a single .npy buffer. */
export function readNpy(buffer: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < NPY_MAGIC.length; i++) {
    if (bytes[i] !== NPY_MAGIC[i]) throw new Error("not a .npy file");
  }

  const view = new DataView(buffer);
  const major = bytes[6];
  // v1 stores the header length as uint16, v2+ as uint32.
  const headerLenSize = major === 1 ? 2 : 4;
  const headerLen =
    major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = 8 + headerLenSize;
  const header = new TextDecoder("latin1").decode(
    bytes.subarray(headerStart, headerStart + headerLen),
  );

  // The header is a Python dict literal, e.g.
  //   {'descr': '<f4', 'fortran_order': False, 'shape': (12, 165), }
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header)?.[1];
  const shapeRaw = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || shapeRaw === undefined) {
    throw new Error(`unparseable .npy header: ${header}`);
  }
  if (fortran === "True") {
    // Column-major would need a transpose; nothing we write produces it.
    throw new Error("fortran_order .npy is not supported");
  }

  const order = descr[0];
  if (order === ">") throw new Error(`big-endian .npy is not supported: ${descr}`);
  const kind = descr.slice(1); // 'f4', 'i4', 'u1', ...
  const Ctor = DTYPES[kind];
  if (!Ctor) throw new Error(`unsupported .npy dtype: ${descr}`);

  const shape = shapeRaw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));

  // Copy rather than view: the payload starts at an arbitrary offset, and a
  // TypedArray needs its byteOffset aligned to the element size.
  const payload = buffer.slice(headerStart + headerLen);
  return { data: new Ctor(payload), shape, dtype: descr };
}

/** Parse an .npz (a ZIP of .npy members) into {name: NpyArray}. */
export async function readNpz(buffer: ArrayBuffer): Promise<Npz> {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // Find the End Of Central Directory record by scanning backwards for its
  // signature. It is last in the file, followed only by an optional comment.
  let eocd = -1;
  for (let i = bytes.length - 22; i >= 0; i--) {
    if (view.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("not a .npz file (no ZIP end-of-central-directory)");

  const entryCount = view.getUint16(eocd + 10, true);
  let pos = view.getUint32(eocd + 16, true); // central directory offset

  const out: Npz = {};
  for (let n = 0; n < entryCount; n++) {
    if (view.getUint32(pos, true) !== 0x02014b50) {
      throw new Error("corrupt .npz central directory");
    }
    const method = view.getUint16(pos + 10, true);
    const compressedSize = view.getUint32(pos + 20, true);
    const nameLen = view.getUint16(pos + 28, true);
    const extraLen = view.getUint16(pos + 30, true);
    const commentLen = view.getUint16(pos + 32, true);
    const localOffset = view.getUint32(pos + 42, true);
    const name = new TextDecoder().decode(
      bytes.subarray(pos + 46, pos + 46 + nameLen),
    );

    // The local header repeats the name and extra fields, and its extra length
    // can differ from the central directory's - so read it, don't assume.
    const localNameLen = view.getUint16(localOffset + 26, true);
    const localExtraLen = view.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLen + localExtraLen;
    const raw = buffer.slice(dataStart, dataStart + compressedSize);

    let member: ArrayBuffer;
    if (method === 0) {
      member = raw; // stored - what np.savez produces
    } else if (method === 8) {
      // deflate - only if someone switches to np.savez_compressed
      member = await inflateRaw(raw);
    } else {
      throw new Error(`unsupported .npz compression method: ${method}`);
    }

    out[name.replace(/\.npy$/, "")] = readNpy(member);
    pos += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

async function inflateRaw(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buffer])
    .stream()
    .pipeThrough(new DecompressionStream("deflate-raw"));
  return new Response(stream).arrayBuffer();
}
