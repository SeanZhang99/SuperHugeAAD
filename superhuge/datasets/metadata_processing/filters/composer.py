from ..data import *


class MetaDataFilterComposer:
    """Base class for metadata filters."""

    def __init__(
        self,
        /,
        *fitlers: Callable[[MetaDataElement | None], MetaDataElement | None],
    ) -> None:
        self.fitlers = list(fitlers)

    def __call__(
        self, metadata_element: MetaDataElement | None
    ) -> MetaDataElement | None:
        """Filter metadata element."""
        for filter_func in self.fitlers:
            metadata_element = filter_func(metadata_element)
            if metadata_element is None:
                return None
        return metadata_element
