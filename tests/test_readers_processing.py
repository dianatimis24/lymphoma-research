from pathlib import Path

import pandas as pd
import pytest
import yaml

from lymphdecoder import data
from lymphdecoder.features import dmr
from lymphdecoder.io import formats, readers
from lymphdecoder.processing import ProcessingConfig, process

DATA = Path(__file__).parent / "data"


def test_readers_are_registered_by_name():
    assert {"array", "wgbs_bed", "rrbs_methylkit", "matrix"} <= set(readers.READERS)
    assert isinstance(readers.make_reader("rrbs_methylkit"), readers.Reader)
    assert isinstance(readers.make_reader("matrix"), readers.MultiSampleReader)
    with pytest.raises(ValueError):
        readers.make_reader("unknown")


def test_new_format_only_needs_a_new_class(tmp_path):
    @readers.register("test_csv_percent")
    class CsvReader(readers.Reader):
        def read(self, path):
            df = pd.read_csv(path)
            return formats.finalize(df.rename(columns={"pos": "base", "meth": "freqC"}))

    try:
        path = tmp_path / "s.csv"
        pd.DataFrame({"chr": ["chr1"], "pos": [5], "meth": [40.0]}).to_csv(path, index=False)
        df = formats.convert(path, "test_csv_percent", tmp_path / "out.tsv")
        assert df["base"].tolist() == [5]
    finally:
        del readers.READERS["test_csv_percent"]


def test_registering_a_name_twice_is_rejected():
    with pytest.raises(ValueError):
        readers.register("array")(type("Dup", (readers.Reader,), {"read": lambda self, p: None}))


def test_dmr_methods_are_registered():
    finder = dmr.make_dmr_finder("naive", threshold=30.0)
    assert isinstance(finder, dmr.DMRFinder)
    assert finder.threshold == 30.0
    with pytest.raises(ValueError):
        dmr.make_dmr_finder("dss")


def make_project(tmp_path, extra_dataset=None):
    raw = tmp_path / "raw"
    (raw / "rrbs").mkdir(parents=True)
    (raw / "wgbs").mkdir()
    (raw / "rrbs" / "s1.txt").write_text((DATA / "rrbs_example.tsv").read_text())
    (raw / "rrbs" / "skip.txt").write_text((DATA / "rrbs_example.tsv").read_text())
    (raw / "wgbs" / "n1.bed").write_text((DATA / "wgbs_example.bed").read_text())
    pd.DataFrame({"chr": ["chr1"] * 2, "pos": [10525, 10848],
                  "FL002": [0.9, 0.5], "MCL001": [0.1, 0.2], "UNK001": [0.3, 0.3]}
                 ).to_csv(raw / "matrix.csv", index=False)
    (tmp_path / "panel.bed").write_text("chr1\t0\t200000\n")
    config = {
        "output": "processed", "panel": "panel.bed", "min_coverage": 10,
        "datasets": [
            {"name": "RRBS1", "reader": "rrbs_methylkit", "folder": "raw/rrbs", "pattern": "*.txt",
             "class": "DLBCL", "platform": "rrbs", "sample_type": "tissue", "exclude": ["skip"]},
            {"name": "WGBS1", "reader": "wgbs_bed", "folder": "raw/wgbs", "pattern": "*.bed",
             "class": "Normal", "platform": "wgbs", "sample_type": "plasma"},
            {"name": "EPIC1", "reader": "matrix", "file": "raw/matrix.csv",
             "reader_params": {"base_col": "pos", "sep": ",", "scale": "fraction"},
             "platform": "array_epic", "sample_type": "tissue", "exclude": ["UNK001"],
             "samples": {"FL002": "FL", "MCL001": "MCL", "UNK001": "unknown"}},
        ],
    }
    if extra_dataset:
        config["datasets"].append(extra_dataset)
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_processing_writes_samples_and_metadata(tmp_path):
    meta = process(ProcessingConfig.load(make_project(tmp_path)))
    assert sorted(meta["sample_id"]) == ["EPIC1/FL002", "EPIC1/MCL001", "RRBS1/s1", "WGBS1/n1"]
    loaded = data.load_metadata(tmp_path / "processed" / "samples.csv")
    assert loaded.loc["WGBS1/n1", "class"] == "Normal"
    assert loaded.loc["EPIC1/FL002", "platform"] == "array_epic"
    rrbs = formats.read(loaded.loc["RRBS1/s1", "file"])
    assert (rrbs["coverage"] >= 10).all()
    wgbs = formats.read(loaded.loc["WGBS1/n1", "file"])
    assert wgbs.empty
    fl = formats.read(loaded.loc["EPIC1/FL002", "file"])
    assert fl["freqC"].tolist() == pytest.approx([90.0, 50.0])


def test_processed_samples_are_not_converted_again(tmp_path):
    config = ProcessingConfig.load(make_project(tmp_path))
    process(config)
    out = tmp_path / "processed" / "RRBS1" / "s1.tsv"
    out.write_text("chr\tbase\tcoverage\tfreqC\n")
    process(config)
    assert out.read_text() == "chr\tbase\tcoverage\tfreqC\n"
    process(config, overwrite=True)
    assert len(formats.read(out)) > 0


@pytest.mark.parametrize("bad, message", [
    ({"name": "X", "reader": "nope", "folder": "a", "class": "c", "platform": "p",
      "sample_type": "t"}, "unknown reader"),
    ({"name": "X", "reader": "array", "platform": "p", "sample_type": "t"}, "exactly one"),
    ({"name": "X", "reader": "array", "folder": "a", "platform": "p", "sample_type": "t"},
     "needs a 'class'"),
    ({"name": "X", "reader": "matrix", "file": "f", "platform": "p", "sample_type": "t"},
     "needs 'samples'"),
    ({"name": "X", "reader": "array", "folder": "a", "class": "c", "platform": "p",
      "sample_type": "t", "colour": "red"}, "unknown keys"),
    ({"name": "RRBS1", "reader": "array", "folder": "a", "class": "c", "platform": "p",
      "sample_type": "t"}, "unique"),
])
def test_configuration_errors_are_reported(tmp_path, bad, message):
    with pytest.raises(ValueError, match=message):
        ProcessingConfig.load(make_project(tmp_path, bad))


def test_missing_input_paths_are_listed(tmp_path):
    config = ProcessingConfig.load(make_project(tmp_path))
    (tmp_path / "panel.bed").unlink()
    config.datasets[0].folder = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError) as err:
        process(config)
    assert "panel.bed" in str(err.value) and "nowhere" in str(err.value)


def test_demo_configuration_runs():
    root = Path(__file__).parent.parent
    config = ProcessingConfig.load(root / "configs" / "demo.yaml")
    config.check_paths()
