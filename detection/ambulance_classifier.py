from typing import Dict


class AmbulanceClassifier:
    """
    Lightweight classifier that determines whether a YOLO class label should be treated as an ambulance.
    """

    def __init__(self, model_names: Dict[int, str]) -> None:
        self.model_names = {int(k): v for k, v in enumerate(model_names.values())} if not isinstance(model_names, dict) else model_names
        self.positive_labels = {"ambulance", "emergency vehicle"}

    def is_ambulance_label(self, label: str) -> bool:
        """
        Returns True if the provided label represents an ambulance.
        """
        return label.strip().lower() in self.positive_labels

    def has_ambulance_class(self) -> bool:
        """
        Returns True if the loaded model contains an ambulance-related class name.
        """
        names = [str(v).lower() for v in self.model_names.values()]
        return any(name in self.positive_labels for name in names)
