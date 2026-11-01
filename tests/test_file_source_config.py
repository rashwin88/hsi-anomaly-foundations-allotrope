"""FileSourceConfig recognises every builder's source (known issue 13: AVIRIS-NG folders were rejected)."""

import pytest

from app.models.file_processing.file_categories import FileCategory
from app.models.file_processing.sources import FileSourceConfig


def test_aviris_ng_scene_folder_is_an_envi_folder(tmp_path):
    folder = tmp_path / "ang20180101t000000"
    folder.mkdir()
    (folder / "ang20180101t000000_corr_v2m2_img.hdr").write_text("ENVI\n")
    assert FileSourceConfig(source_path=str(folder)).file_category == FileCategory.ENVI_FOLDER


@pytest.mark.parametrize("name,cat", [("x.he5", FileCategory.HDFS), ("t.h5", FileCategory.HDFS),
                                      ("l.tif", FileCategory.TIF)])
def test_files_by_extension(name, cat):
    assert FileSourceConfig(source_path=name).file_category == cat


def test_a_folder_without_envi_headers_is_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unrecognized"):
        FileSourceConfig(source_path=str(tmp_path))
