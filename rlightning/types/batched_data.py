"""Batched data container for environment and policy data.

This module provides the BatchedData class for holding collections of
environment returns or policy responses, supporting both local data
and Ray ObjectRefs for distributed execution.
"""

from typing import Any, Dict, Iterator, List, Tuple, Union

from ray import ObjectRef


class BatchedData:
    """Container for batched data with associated identifiers.

    Holds either actual data or Ray ObjectRefs, maintaining the order
    of (id, data) pairs. Allows duplicate IDs.

    Attributes:
        is_future: True if any data element is a Ray ObjectRef.
    """

    def __init__(self, ids: List[Any], data: List[Union[Any, ObjectRef]]) -> None:
        """Initialize batched data container.

        Args:
            ids: List of identifiers for each data element.
            data: List of data elements or Ray ObjectRefs.

        Raises:
            ValueError: If lengths of ids and data don't match.
        """
        if len(ids) != len(data):
            raise ValueError("Length of ids and data must be the same.")

        self._ids: Tuple = tuple(ids)
        self._data: Tuple = tuple(data)

        self.is_future: bool = True if any(isinstance(d, ObjectRef) for d in data) else False

    def __iter__(self) -> Iterator[Tuple[Any, Union[Any, ObjectRef]]]:
        """Iterate over (id, data) pairs."""
        return zip(self._ids, self._data)

    def __getitem__(self, id_key: Any) -> Union[Any, ObjectRef]:
        """Get data by ID.

        Args:
            id_key: Identifier to look up.

        Returns:
            Data element associated with the ID.
        """
        return self._data[self._ids.index(id_key)]

    def items(self) -> Iterator[Tuple[Any, Union[Any, ObjectRef]]]:
        """Get (id, data) pairs iterator.

        Returns:
            Iterator of (id, data) pairs.
        """
        return zip(self._ids, self._data)

    def ids(self) -> Tuple:
        """Get all IDs in the batched data.

        Returns:
            Tuple of all identifiers.
        """
        return self._ids

    def values(self) -> Tuple:
        """Get all data values.

        Returns:
            Tuple of all data elements.
        """
        return self._data

    def __len__(self) -> int:
        """Get number of elements."""
        return len(self._ids)

    @staticmethod
    def from_dict(data_dict: Dict) -> "BatchedData":
        """Create BatchedData from a dictionary.

        Args:
            data_dict: Dictionary mapping IDs to data.

        Returns:
            BatchedData instance.
        """
        return BatchedData(list(data_dict.keys()), list(data_dict.values()))
