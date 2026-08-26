"""
Tests for the segmentation sharder's storage backends.

No AWS and no payloads: the S3 tests bypass `__init__` with `object.__new__`
so no boto3 client is ever constructed. Runs anywhere.

The `release_scene` tests are the important ones. Under `LocalSceneStorage`
a scene folder is the user's own data - on Google Drive, in practice - and
the reconstruction sharder's equivalent step is an unguarded
`shutil.rmtree(..., ignore_errors=True)`. If that behaviour ever leaks into
this path it destroys source scenes silently. These tests are the guard.
"""

import inspect
import os

import pytest

from app.utils.patch_generation.scene_storage import (
    LocalSceneStorage,
    S3SceneStorage,
    SceneStorage,
)

_METHODS = [
    "list_scenes",
    "fetch_scene",
    "fetch_metadata",
    "publish_shard",
    "release_scene",
    "shard_exists",
]


@pytest.fixture
def scene_root(tmp_path):
    """Two EnMAP scenes, one decoy directory and one loose file."""
    for name in ["ENMAP01-b", "ENMAP01-a"]:
        d = tmp_path / name
        d.mkdir()
        (d / "SPECTRAL_IMAGE.TIF").write_bytes(b"irreplaceable")
    (tmp_path / "PRISMA-x").mkdir()
    (tmp_path / "notes.txt").write_text("not a scene")
    return str(tmp_path)


@pytest.mark.parametrize("method", _METHODS)
def test_backends_match_the_protocol(method):
    """A backend that drifts from the protocol fails at the call site, so
    keep the signatures pinned here instead."""
    expected = str(inspect.signature(getattr(SceneStorage, method)))
    assert str(inspect.signature(getattr(LocalSceneStorage, method))) == expected
    assert str(inspect.signature(getattr(S3SceneStorage, method))) == expected


# --- LocalSceneStorage --------------------------------------------------


def test_lists_only_matching_directories(scene_root):
    assert LocalSceneStorage(scene_root).list_scenes() == ["ENMAP01-a", "ENMAP01-b"]


def test_fetch_returns_the_source_path_and_copies_nothing(scene_root):
    got = LocalSceneStorage(scene_root).fetch_scene("ENMAP01-a", dest_dir="/ignored")
    assert got == os.path.join(scene_root, "ENMAP01-a")
    assert os.path.isdir(got)


def test_local_metadata_fetch_is_the_same_no_op(scene_root):
    """Nothing is downloaded locally, so there is nothing cheaper to download."""
    storage = LocalSceneStorage(scene_root)
    assert storage.fetch_metadata("ENMAP01-a", "/ignored") == storage.fetch_scene(
        "ENMAP01-a", "/ignored"
    )


def test_s3_metadata_fetch_downloads_only_metadata(tmp_path):
    """The point of the method: splitting 212 scenes must not move ~64 GB to
    read ~14 MB of cover percentages."""
    s3 = _bare_s3()
    s3.bucket, s3.scene_prefix, s3.metadata_suffix = "b", "enmap/", "METADATA.XML"
    keys = [
        "enmap/SCENE/SCENE-SPECTRAL_IMAGE.TIF",
        "enmap/SCENE/SCENE-METADATA.XML",
        "enmap/SCENE/SCENE-QL_QUALITY_CLOUD.TIF",
    ]
    s3.client = type("C", (), {
        "list_objects": lambda self, **kw: {"Contents": [{"Key": k} for k in keys]},
        "download_file": lambda self, Bucket, Key, Filename: downloaded.append(Key),
    })()
    downloaded = []
    s3.fetch_metadata("SCENE", str(tmp_path))
    assert downloaded == ["enmap/SCENE/SCENE-METADATA.XML"]

    downloaded.clear()
    s3.fetch_scene("SCENE", str(tmp_path))
    assert downloaded == keys, "fetch_scene still takes everything"


def test_release_does_not_delete_the_source(scene_root):
    """THE critical one. Local scenes are the user's data, never a copy."""
    storage = LocalSceneStorage(scene_root)
    path = storage.fetch_scene("ENMAP01-a", dest_dir="/ignored")
    storage.release_scene(path)
    assert os.path.isdir(path)
    assert os.listdir(path) == ["SPECTRAL_IMAGE.TIF"]


def test_publish_copies_and_keeps_the_source(scene_root, tmp_path):
    out = tmp_path / "published"
    storage = LocalSceneStorage(scene_root, shard_dir=str(out))
    tar = tmp_path / "scene_a.tar"
    tar.write_bytes(b"x")
    storage.publish_shard(str(tar))
    assert (out / "scene_a.tar").exists()
    assert tar.exists(), "publish must copy, never move"


def test_shard_exists_round_trip(scene_root, tmp_path):
    out = tmp_path / "published"
    storage = LocalSceneStorage(scene_root, shard_dir=str(out))
    tar = tmp_path / "scene_a.tar"
    tar.write_bytes(b"x")
    assert storage.shard_exists("scene_a.tar") is False
    storage.publish_shard(str(tar))
    assert storage.shard_exists("scene_a.tar") is True
    assert storage.shard_exists("scene_b.tar") is False


def test_no_destination_never_skips_work(scene_root):
    """False, not True: a misconfigured run must redo work, not silently
    report success having produced nothing."""
    assert LocalSceneStorage(scene_root).shard_exists("anything.tar") is False


# --- S3SceneStorage -----------------------------------------------------


def _bare_s3(**overrides):
    """An instance without __init__, so no boto3 client is built.

    Every attribute the methods read must be set here — bypassing __init__
    means a newly added one is missing and surfaces as AttributeError in
    unrelated tests. Defaults mirror the real constructor.
    """
    s3 = object.__new__(S3SceneStorage)
    s3._fetched = set()
    s3.bucket = "bucket"
    s3.scene_prefix = "enmap/"
    s3.shard_prefix = None
    s3.metadata_suffix = "METADATA.XML"
    s3.scene_suffix = None
    for key, value in overrides.items():
        setattr(s3, key, value)
    return s3


def test_s3_lists_bare_ids_not_prefixes():
    """Both backends must speak the same vocabulary or the sharder would
    have to know which one it is holding."""
    s3 = _bare_s3()
    s3.bucket, s3.scene_prefix = "b", "enmap/"
    s3.paginator = type(
        "P", (), {"paginate": lambda self, **kw: [
            {"CommonPrefixes": [{"Prefix": "enmap/ENMAP01-b/"},
                                {"Prefix": "enmap/ENMAP01-a/"}]}]}
    )()
    assert s3.list_scenes() == ["ENMAP01-a", "ENMAP01-b"]


def test_local_lists_files_when_dirs_only_is_off(tmp_path):
    """PRISMA ships one .he5 per scene, not a folder of files."""
    (tmp_path / "PRS_a.he5").write_bytes(b"x")
    (tmp_path / "PRS_b.he5").write_bytes(b"x")
    (tmp_path / "PRS_c").mkdir()
    (tmp_path / "notes.txt").write_text("")
    storage = LocalSceneStorage(str(tmp_path), pattern="PRS_*", dirs_only=False)
    assert storage.list_scenes() == ["PRS_a.he5", "PRS_b.he5"]


def test_s3_lists_objects_when_scene_suffix_is_set(tmp_path):
    s3 = _bare_s3()
    s3.bucket, s3.scene_prefix, s3.scene_suffix = "b", "prisma/", ".he5"
    keys = ["prisma/PRS_b.he5", "prisma/PRS_a.he5", "prisma/notes.txt"]
    s3.paginator = type("P", (), {
        "paginate": lambda self, **kw: [{"Contents": [{"Key": k} for k in keys]}]
    })()
    assert s3.list_scenes() == ["PRS_a.he5", "PRS_b.he5"], "sorted, bare, .he5 only"


def test_s3_fetch_of_a_single_object_returns_the_file(tmp_path):
    s3 = _bare_s3()
    s3.bucket, s3.scene_prefix, s3.scene_suffix = "b", "prisma/", ".he5"
    pulled = []
    s3.client = type("C", (), {
        "download_file": lambda self, Bucket, Key, Filename: (
            pulled.append(Key), open(Filename, "wb").write(b"x")
        ),
    })()
    got = s3.fetch_scene("PRS_a.he5", str(tmp_path))
    assert pulled == ["prisma/PRS_a.he5"], "one object, not a folder listing"
    assert got == str(tmp_path / "PRS_a.he5"), "returns the file, not a directory"

    s3.release_scene(got)
    assert not os.path.exists(got), "release must remove a file scene too"


def test_s3_release_refuses_paths_it_did_not_fetch(tmp_path):
    """rmtree runs with ignore_errors=True, so a wrong target fails silently.
    Guarding the input is the only safe defence."""
    theirs = tmp_path / "not_mine"
    theirs.mkdir()
    (theirs / "f").write_text("")
    _bare_s3().release_scene(str(theirs))
    assert theirs.is_dir()


def test_s3_release_removes_what_it_fetched(tmp_path):
    mine = tmp_path / "downloaded"
    mine.mkdir()
    (mine / "f").write_text("")
    s3 = _bare_s3()
    s3._fetched.add(str(mine))
    s3.release_scene(str(mine))
    assert not mine.exists()
