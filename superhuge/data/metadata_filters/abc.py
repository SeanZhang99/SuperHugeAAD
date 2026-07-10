"""
Revised MetadataFilter ABC with dual invocation modes.

Behaviour change: 2026/06/03.
- New-style subclasses override ``filter_element`` (per-element) and/or
  ``filter_batch`` (batch).  ``__call__`` acts as a unified dispatcher
  that works with both single elements and dicts.
- Legacy subclasses that override ``__call__`` with per-element logic
  continue to work without modification.  The ``apply`` method provides
  a batch-compatible entry point for all filter styles.

Forward-compatibility: if neither ``filter_element`` nor ``filter_batch``
is overridden, the filter automatically falls back to per-element mode
using whatever ``__call__`` provides (legacy pattern).
"""

from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar, overload, Any

from ..metadata_processing.data import Metadata, MetadataElement

MetadataElementType = TypeVar("MetadataElementType", bound=MetadataElement)


class MetadataFilter(ABC, Generic[MetadataElementType]):
    """Abstract metadata filter with dual invocation modes.

    Invocation modes
    ----------------
    **Per-element mode** — filter one element at a time::

        result = my_filter(element)        # __call__ dispatcher
        result = my_filter.filter_element(element)

    **Batch mode** — filter the whole metadata dict at once::

        result = my_filter(metadata_dict)  # __call__ dispatcher (new-style only)
        result = my_filter.apply(metadata_dict)  # works for ALL styles

    How to subclass
    ---------------
    Override **one** (or more) of the following:

    - ``filter_element(element)`` — per-element logic (new pattern).
    - ``filter_batch(metadata_dict)`` — batch logic (new pattern).
    - ``__call__(element)`` — per-element logic (legacy pattern).

    The unimplemented one auto-delegates to the other, so you only write
    the logic once.

    Legacy subclasses that override ``__call__`` with per-element logic
    continue to work without modification.  For legacy subclasses,
    ``filter(dict)`` is NOT supported — use ``filter.apply(dict)`` or
    ``filter.filter_batch(dict)`` instead.

    Forward-compatibility fallback
    -----------------------------
    If neither ``filter_element`` nor ``filter_batch`` is overridden (the
    legacy pattern), the filter automatically falls back to per-element
    mode: ``filter_batch`` iterates ``filter_element``, and
    ``filter_element`` delegates to the overridden ``__call__``.

    Enforcement
    -----------
    Every concrete (non-marker) subclass **must** override at least one
    of ``__call__``, ``filter_element``, or ``filter_batch``.  Violations
    raise ``TypeError`` at class-definition time.

    Marker base classes (e.g. ``ClassifyMetadataFilter``) set the class
    attribute ``_filter_marker = True`` to opt out of this check.
    """

    # Subclasses set _filter_marker = True to skip the override guard
    # (used by identification-only base classes like ClassifyMetadataFilter).
    _filter_marker: bool = False

    # ------------------------------------------------------------------
    # __init_subclass__ — enforce at least one override
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Skip the guard for marker base classes and MetadataFilter itself.
        if getattr(cls, "_filter_marker", False):
            return
        if cls is MetadataFilter:
            return
        # Check: at least one of __call__, filter_element, filter_batch
        # is overridden in *this* class's __dict__ (not inherited).
        call_overridden = "__call__" in cls.__dict__
        element_overridden = "filter_element" in cls.__dict__
        batch_overridden = "filter_batch" in cls.__dict__
        if not (call_overridden or element_overridden or batch_overridden):
            raise TypeError(
                f"{cls.__name__} must override at least one of "
                f"'__call__', 'filter_element', or 'filter_batch'. "
                f"If this is an identification-only base class, "
                f"set '_filter_marker = True'."
            )

    # ------------------------------------------------------------------
    # Core overridable methods — implement ONE (or both)
    # ------------------------------------------------------------------

    def filter_element(
        self,
        metadata_element: MetadataElementType | None,
    ) -> MetadataElementType | None:
        """Per-element filter logic.

        Return the (possibly modified) element, or ``None`` to discard it.

        **Default delegation order** (when NOT overridden by subclass):

        1. If ``filter_batch`` is overridden → wrap single element in a
           dict and delegate to ``filter_batch``.
        2. If ``__call__`` is overridden (legacy pattern) → delegate to
           ``self(element)``.
        3. Otherwise → pass-through (return element as-is).
        """
        if metadata_element is None:
            return None
        # Priority 1: batch-only subclass — delegate to filter_batch
        if type(self).filter_batch is not MetadataFilter.filter_batch:
            result = self.filter_batch({"__single__": metadata_element})
            return result.get("__single__")
        # Priority 2: legacy subclass — delegate to overridden __call__
        if type(self).__call__ is not MetadataFilter.__call__:
            out: dict[str, MetadataElementType] | MetadataElementType | None = self(
                metadata_element
            )
            assert isinstance(out, MetadataElement) or out is None
            return out
        # Priority 3: nothing overridden — pass-through
        return metadata_element

    def filter_batch(
        self,
        metadata: dict[str, MetadataElementType],
    ) -> dict[str, MetadataElementType]:
        """Batch filter logic — receive the full metadata dict, return filtered dict.

        Return a dict containing only the entries to keep (possibly modified).
        Omitting a key is equivalent to returning ``None`` for that element
        in per-element mode.

        **Default**: iterate ``filter_element`` for each entry.
        """
        out: dict[str, MetadataElementType] = {}
        for key, element in metadata.items():
            result = self.filter_element(element)
            if result is not None:
                out[key] = result
        return out

    # ------------------------------------------------------------------
    # apply — universal batch entry point (works for ALL filter styles)
    # ------------------------------------------------------------------

    def apply(
        self,
        metadata: dict[str, MetadataElementType],
    ) -> dict[str, MetadataElementType]:
        """Apply the filter to a metadata dict.

        This is the **recommended batch entry point**.  It works correctly
        for both legacy (per-element ``__call__``) and new-style
        (``filter_element`` / ``filter_batch``) filters.

        New-style filters can also call ``filter(dict)`` directly via the
        ``__call__`` dispatcher.  Legacy filters must use ``apply`` or
        ``filter_batch`` for batch processing.
        """
        return self.filter_batch(metadata)

    # ------------------------------------------------------------------
    # __call__ — unified dispatcher (new-style) / per-element (legacy)
    # ------------------------------------------------------------------

    # @overload
    # def __call__(
    #     self,
    #     x: MetadataElementType | None,
    #     /,
    # ) -> MetadataElementType | None: ...

    # @overload
    # def __call__(
    #     self,
    #     x: dict[str, MetadataElementType],
    #     /,
    # ) -> dict[str, MetadataElementType]: ...

    def __call__(self, input_data: Any, /) -> Any:  # type: ignore
        """Unified entry point for new-style filters; per-element for legacy.

        **New-style** subclasses (override ``filter_element`` / ``filter_batch``
        but NOT ``__call__``):
            ``filter(element)`` → ``filter_element(element)``
            ``filter(dict)``    → ``filter_batch(dict)``

        **Legacy** subclasses (override ``__call__`` directly):
            ``filter(element)`` → subclass's ``__call__(element)``  ✓
            ``filter(dict)``    → subclass's ``__call__(dict)``     ✗ (undefined)
            Use ``filter.apply(dict)`` instead for batch mode.
        """
        if isinstance(input_data, dict):
            return self.filter_batch(input_data)
        return self.filter_element(input_data)


# ======================================================================
# Usage examples
# ======================================================================

if __name__ == "__main__":

    # -- 0. Guard demonstration ------------------------------------------

    # This would raise TypeError at class-definition time:
    #
    #   class EmptyFilter(MetadataFilter):
    #       pass
    #   # TypeError: EmptyFilter must override at least one of
    #   # '__call__', 'filter_element', or 'filter_batch'.

    # Marker base classes are exempt (they set _filter_marker = True):
    class MyMarkerBase(MetadataFilter):
        _filter_marker = True  # opt out of the override guard

    # -- 1. Legacy subclass (existing pattern, no changes required) -----

    class MyLegacyFilter(MetadataFilter):
        """Per-element filter using the legacy __call__ override."""

        def __call__(self, metadata_element):
            if metadata_element is None:
                return None
            if metadata_element.dataset_id < 5:
                return None  # discard
            return metadata_element

    legacy = MyLegacyFilter()
    # Per-element call (same as before):
    # result = legacy(element)
    # Batch call (use apply, NOT legacy(dict)):
    # result = legacy.apply(metadata_dict)

    # -- 2. New-style per-element filter (override filter_element) -------

    class DurationFilter(MetadataFilter):
        """Drop elements whose signal_length < threshold."""

        def __init__(self, min_length: int) -> None:
            self.min_length = min_length

        def filter_element(self, metadata_element):
            if metadata_element is None:
                return None
            if metadata_element.signal_length < self.min_length:
                return None
            return metadata_element

    dur_filter = DurationFilter(min_length=5000)
    # Both call styles work via __call__ dispatcher:
    # result = dur_filter(element)       # → filter_element
    # result = dur_filter(metadata_dict) # → filter_batch (auto-iterates)
    # result = dur_filter.apply(metadata_dict)  # also works

    # -- 3. Batch filter (override filter_batch) ------------------------

    class DeduplicationFilter(MetadataFilter):
        """Keep only the first occurrence of each subject_id."""

        def filter_batch(self, metadata):
            seen: set[int] = set()
            out = {}
            for key, element in metadata.items():
                if element.subject_id not in seen:
                    seen.add(element.subject_id)
                    out[key] = element
            return out

    dedup = DeduplicationFilter()
    # result = dedup(metadata_dict)  # → filter_batch
    # result = dedup(element)        # → filter_element → wraps in dict → filter_batch
    # result = dedup.apply(metadata_dict)  # also works

    # -- 4. Dual-mode filter (override both for optimal performance) ----

    class NormalizationFilter(MetadataFilter):
        """Normalize signal_length relative to the batch maximum;
        per-element mode just centers around a reference."""

        def __init__(self, reference: int = 0) -> None:
            self.reference = reference

        def filter_element(self, metadata_element):
            if metadata_element is None:
                return None
            metadata_element.signal_length -= self.reference
            return metadata_element

        def filter_batch(self, metadata):
            if not metadata:
                return metadata
            max_len = max(el.signal_length for el in metadata.values())
            for element in metadata.values():
                element.signal_length = element.signal_length / max_len
            return metadata

    norm = NormalizationFilter()
    # result = norm(element)        # → filter_element
    # result = norm(metadata_dict)  # → filter_batch (efficient)
