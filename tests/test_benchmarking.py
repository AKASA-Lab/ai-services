from lettuce_benchmark.benchmarking import (
    checkpoint_size_bytes,
    measure_classical_prediction,
    measure_training,
    parameter_count,
)


class FakeModel:
    def parameters(self):
        return [FakeParameter(3), FakeParameter(2)]


class FakeParameter:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class FakeClassifier:
    def predict(self, features):
        return [len(features)]


def test_training_checkpoint_and_parameters(tmp_path) -> None:
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"weights")
    result, duration = measure_training(lambda value: value + 1, 4)
    assert result == 5
    assert duration >= 0
    assert parameter_count(FakeModel()) == 5
    assert checkpoint_size_bytes(checkpoint) == 7


def test_classical_feature_and_prediction_timing_are_separate() -> None:
    measured = measure_classical_prediction(
        lambda images: [len(images)],
        FakeClassifier(),
        ["a", "b"],
    )
    assert measured["predictions"] == [1]
    assert measured["feature_extraction_seconds"] >= 0
    assert measured["classifier_prediction_seconds"] >= 0
