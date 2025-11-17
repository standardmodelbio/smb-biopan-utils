"""MEDS EHR data formatter for inference.

This module formats patient event timelines from MEDS format into structured
text suitable for model inference. It provides utilities for extracting and
formatting demographics, conditions, measurements, observations, procedures,
drugs, and imaging paths.

UPDATED: Uses end_time as anchor point for time bins, fetching most recent events.
"""

# pyright: reportMissingTypeStubs=false

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import pandas as pd


# Define how the 'table' column in your data maps to the desired output tags.
TAG_TO_TABLES: DefaultDict[str, List[str]] = defaultdict(list)

# demographics
TAG_TO_TABLES["demographics"].extend(["person"])

# conditions
TAG_TO_TABLES["conditions"].extend(
    [
        "condition",
        "diagnosis",
        "condition_occurrence",
    ]
)

# observations
TAG_TO_TABLES["observations"].extend(
    [
        "observation",
        "ed_out",
        "icu_discharge",
        "ed_registration",
        "hospital_admission",
        "icu_admission",
        "hospital_discharge",
    ]
)

# measurements
TAG_TO_TABLES["measurements"].extend(
    [
        "measurement",
        "lab",
        "blood pressure sitting",
        "bmi",
        "bmi (kg/m2)",
        "blood pressure standing (1 min)",
        "egfr",
        "blood pressure",
        "height",
        "height (inches)",
        "weight",
        "blood pressure standing",
        "blood pressure lying",
        "subject_fluid_output",
        "blood pressure standing (3 mins)",
        "weight (lbs)",
    ]
)

# procedures
TAG_TO_TABLES["procedures"].extend(
    [
        "procedure",
        "device_exposure",
        "procedure_occurrence",
    ]
)

# drugs
TAG_TO_TABLES["drugs"].extend(
    [
        "drug_exposure",
        "medication",
        "infusion_start",
        "infusion_end",
    ]
)

# death
TAG_TO_TABLES["death"].extend(
    [
        "death",
        "meds_death",
    ]
)

# notes
TAG_TO_TABLES["notes"].extend(["note"])


def _get_tag_for_table(table_name: str) -> Optional[str]:
    """Get the tag associated with a table name."""
    for tag, tables in TAG_TO_TABLES.items():
        if table_name in tables:
            return tag
    return None


def _format_event_value(row: pd.Series) -> str:
    """Format a single event's value from numeric or text fields."""
    if pd.notna(row.get("numeric_value")):
        value = f"{row['numeric_value']:.2f}"
        return value
    elif pd.notna(row.get("text_value")):
        return str(row["text_value"])
    return ""


def format_events_chronological(
    events_df: Any,
    code_column: str,
    omop_table_name: str,
    max_values_per_code: int = 5,
) -> str:
    """
    Formats a dataframe of patient events into a structured string with
    custom tags. Groups by code and shows values in chronological order.

    Args:
        events_df: DataFrame containing patient events, pre-sorted by time.
        code_column: Column name containing the code/event type information.
        omop_table_name: Column name containing table/event type information.
        max_values_per_code: Maximum number of values to show per code (most recent).

    Returns:
        Formatted string with tagged sections (e.g., <conditions>...</conditions>).
    """
    if events_df is None or events_df.empty:
        return ""

    output_parts: Dict[str, List[str]] = {}

    for tag in TAG_TO_TABLES.keys():
        # Filter events for this tag
        tag_events_df = events_df[events_df[omop_table_name].isin(TAG_TO_TABLES[tag])]
        if tag_events_df.empty:
            continue

        aggregated_lines: List[str] = []

        # Group by code while preserving chronological order (sort=False keeps first-seen order)
        for code, group in tag_events_df.groupby(code_column, sort=False):
            # Special handling for birth
            if "birth" in str(code).lower():
                birth_date = group["time"].iloc[0].strftime("%Y-%m-%d")
                aggregated_lines.append(f"Birth: {birth_date}")
                continue

            # Collect all values in chronological order
            values_with_units: List[Tuple[str, str]] = []

            for idx, row in group.iterrows():
                value_str = _format_event_value(row)
                if value_str:
                    unit = row.get("unit")
                    unit_str = str(unit) if pd.notna(unit) else ""
                    values_with_units.append((value_str, unit_str))

            if values_with_units:
                # Take the last N values (most recent)
                recent_values = values_with_units[-max_values_per_code:]

                # Get the unit from the last value (most recent)
                last_unit = recent_values[-1][1] if recent_values else ""
                unit_suffix = f" ({last_unit})" if last_unit else ""

                # Extract just the values
                value_strings = [v[0] for v in recent_values]
                aggregated_lines.append(f"{code}{unit_suffix}: {', '.join(value_strings)}")
            else:
                # Code with no values
                aggregated_lines.append(str(code))

        if aggregated_lines:
            # Deduplicate while preserving original order
            seen_lines: Set[str] = set()
            unique_in_order: List[str] = []
            for line in aggregated_lines:
                if line not in seen_lines:
                    seen_lines.add(line)
                    unique_in_order.append(line)
            output_parts[tag] = unique_in_order

    # Format output by tag order
    formatted_parts: List[str] = []
    for tag in TAG_TO_TABLES.keys():
        if tag in output_parts and output_parts[tag]:
            formatted_parts.append(f"<{tag}>\n" + "\n".join(output_parts[tag]) + f"\n</{tag}>")

    return "\n".join(formatted_parts)


def format_events_by_date(
    events_df: Any,
    code_column: str,
    omop_table_name: str,
) -> str:
    """
    Formats events grouped by date with date headers.

    Args:
        events_df: DataFrame containing patient events, pre-sorted by time.
        code_column: Column name containing the code/event type information.
        omop_table_name: Column name containing table/event type information.

    Returns:
        Formatted string with date headers and tagged event sections.
    """
    if events_df is None or events_df.empty:
        return ""

    # Add date column for grouping
    events_df = events_df.copy()
    events_df["date"] = events_df["time"].dt.date

    output_lines: List[str] = []

    # Group by date
    for date, date_group in events_df.groupby("date", sort=True):
        date_str = date.strftime("%Y-%m-%d")
        output_lines.append(f"\n[{date_str}]")

        # Format events for this date
        date_content = format_events_chronological(date_group, code_column, omop_table_name)
        if date_content:
            output_lines.append(date_content)

    return "\n".join(output_lines)


def extract_imaging_paths(events_df: Any) -> List[str]:
    """
    Extract imaging file paths from an events dataframe.

    Looks for 'img_path', 'ct_path', or 'file_path' columns.

    Args:
        events_df: DataFrame containing patient events.

    Returns:
        List of unique imaging file paths in chronological order.
    """
    if events_df is None or events_df.empty:
        return []

    paths: List[str] = []

    # Column-based extraction
    for col in ("img_path", "ct_path", "file_path"):
        if col in events_df.columns:
            col_vals = events_df[col].dropna().astype(str).tolist()
            paths.extend(col_vals)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    return ordered


def prefix_image_tokens(content: str, num_tokens: int, token: str = "<image>") -> str:
    """
    Prefix image tokens to content for multimodal models.

    Args:
        content: Text content to prefix.
        num_tokens: Number of tokens to add.
        token: Token string to repeat (default: "<image>").

    Returns:
        Content with prefixed tokens.
    """
    if num_tokens <= 0:
        return content
    tokens = "".join([token for _ in range(num_tokens)])
    return f"{tokens}{content}"


def format_patient_history(
    patient_df: Any,
    omop_table_name: str = "table",
    code_column: str = "code",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    include_demographics: bool = True,
    time_bins: Optional[List[Tuple[int, int]]] = None,
) -> str:
    """
    Format patient history as a structured text string for inference.

    Args:
        patient_df: DataFrame of events for a single patient (MEDS format).
        omop_table_name: Column name containing table/event type information.
        code_column: Column name containing the code/event type information.
        start_time: Optional start time to filter events (lower bound, inclusive).
        end_time: Optional end time / anchor time for filtering and time bins.
        include_demographics: Whether to include demographics section.
        time_bins: Optional list of (days_back_start, days_back_end) tuples for binned history.
                   Events are organized into bins going backwards from end_time.
                   Each tuple defines a time window: (further_back, more_recent).
                   Example: [(90, 30), (30, 7), (7, 0)] creates bins going back from end_time.
                   Bins are output in REVERSE order (most recent first).
                   Timespan labels show absolute dates (e.g., "2023-10-01 - 2024-01-01").
                   If None, events are grouped by date with date headers (e.g., "[2024-01-01]").

    Returns:
        Formatted text string with tagged sections (demographics, conditions, etc.).
    """
    if patient_df.empty:
        return ""

    # Ensure strict chronological ordering
    patient_df = patient_df.sort_values(["time", omop_table_name]).reset_index(drop=True)

    # Apply time filtering
    filtered_df = patient_df.copy()

    if time_bins and end_time:
        # For time bins: keep everything before or at end_time (anchor point)
        filtered_df = filtered_df[filtered_df["time"] <= end_time]
        # Also apply start_time if provided as absolute lower bound
        if start_time is not None:
            filtered_df = filtered_df[filtered_df["time"] >= start_time]
    else:
        # For non-binned: apply start_time as lower bound, end_time as upper bound
        if start_time is not None:
            filtered_df = filtered_df[filtered_df["time"] >= start_time]
        if end_time is not None:
            filtered_df = filtered_df[filtered_df["time"] <= end_time]

    if filtered_df.empty:
        return ""

    output_parts: List[str] = []

    # Add demographics first if requested
    if include_demographics:
        person_rows = filtered_df[filtered_df[omop_table_name] == "person"]
        if not person_rows.empty:
            demographics_str = format_events_chronological(person_rows, code_column, omop_table_name)
            if demographics_str:
                output_parts.append(demographics_str)

    # Remove demographics from main timeline
    non_person_df = filtered_df[filtered_df[omop_table_name] != "person"]

    if non_person_df.empty:
        return "\n\n".join(output_parts)

    # If time bins are specified and we have an end_time (anchor), organize by bins
    if time_bins and end_time:
        # Validate time_bins
        for days_back_start, days_back_end in time_bins:
            if days_back_start < days_back_end:
                raise ValueError(
                    f"Invalid time bin: ({days_back_start}, {days_back_end}). "
                    f"First value (days_back_start) must be >= second value (days_back_end)"
                )

        # Process bins in REVERSE order to show most recent first
        for days_back_start, days_back_end in reversed(time_bins):
            bin_start_time = end_time - pd.Timedelta(days=days_back_start)
            bin_end_time = end_time - pd.Timedelta(days=days_back_end)

            bin_df = non_person_df[(non_person_df["time"] > bin_start_time) & (non_person_df["time"] <= bin_end_time)]

            if not bin_df.empty:
                # Create label with absolute dates
                bin_label = f"{bin_start_time.strftime('%Y-%m-%d')} - {bin_end_time.strftime('%Y-%m-%d')}"

                bin_content = format_events_chronological(bin_df, code_column, omop_table_name)
                if bin_content:
                    output_parts.append(f"<timespan>{bin_label}</timespan>\n{bin_content}")
    else:
        # Format all events grouped by date
        content = format_events_by_date(non_person_df, code_column, omop_table_name)
        if content:
            output_parts.append(content)

    return "\n\n".join(output_parts)


def process_ehr_info(
    df: Any,
    subject_id: Optional[Any] = None,
    omop_table_name: str = "table",
    code_column: str = "code",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    include_demographics: bool = True,
    time_bins: Optional[List[Tuple[int, int]]] = None,
    include_imaging: bool = False,
) -> str | Dict[str, Any]:
    """
    High-level API to format MEDS data for inference.

    Args:
        df: DataFrame in MEDS format (can contain multiple subjects).
        subject_id: Optional subject ID to filter to a specific patient.
                   If None, formats all subjects concatenated.
        omop_table_name: Column name containing table/event type information.
        code_column: Column name containing the code/event type information.
        start_time: Optional start time (absolute lower bound for events).
        end_time: Optional end time / anchor time for time bins.
        include_demographics: Whether to include demographics section.
        time_bins: Optional list of (days_back_start, days_back_end) tuples.
                   Events are fetched going backwards from end_time.
                   Output shows most recent bins first with absolute date labels.
                   If None, events are grouped by date with date headers.
        include_imaging: If True, returns dict with 'text' and 'images' keys.

    Returns:
        Formatted text string ready for model inference, or dict if include_imaging=True.

    Example:
        >>> import pandas as pd
        >>> df = pd.read_parquet("patient_data.parquet")
        >>>
        >>> # Simple format with date grouping (no time bins)
        >>> # Events grouped by date with headers like "[2024-01-01]"
        >>> text = process_ehr_info(df, subject_id="patient_123")
        >>>
        >>> # With time bins (most recent events first)
        >>> # This fetches events going back from end_time, showing most recent first
        >>> # Timespan labels will show absolute dates like "2023-12-25 - 2024-01-01"
        >>> text = process_ehr_info(
        ...     df,
        ...     subject_id="patient_123",
        ...     end_time=pd.Timestamp("2024-01-01"),  # anchor point
        ...     time_bins=[(90, 30), (30, 7), (7, 0)]  # output order: 7-0, 30-7, 90-30 days
        ... )
    """
    if df.empty:
        return "" if not include_imaging else {"text": "", "images": []}

    # Ensure the dataframe has required columns
    if "subject_id" not in df.columns and subject_id is None:
        # Assume single subject
        df = df.copy()
        df["subject_id"] = "default"

    # Filter to specific subject if requested
    if subject_id is not None:
        df = df[df["subject_id"] == subject_id]
        if df.empty:
            return "" if not include_imaging else {"text": "", "images": []}

    # If processing multiple subjects, format each separately
    if len(df["subject_id"].unique()) > 1:
        results = []
        for sid in df["subject_id"].unique():
            patient_df = df[df["subject_id"] == sid]
            text = format_patient_history(
                patient_df=patient_df,
                omop_table_name=omop_table_name,
                code_column=code_column,
                start_time=start_time,
                end_time=end_time,
                include_demographics=include_demographics,
                time_bins=time_bins,
            )
            if text:
                results.append(f"Subject: {sid}\n{text}")
        formatted_text = "\n\n" + ("=" * 80 + "\n\n").join(results)
    else:
        # Single subject
        formatted_text = format_patient_history(
            patient_df=df,
            omop_table_name=omop_table_name,
            code_column=code_column,
            start_time=start_time,
            end_time=end_time,
            include_demographics=include_demographics,
            time_bins=time_bins,
        )

    # Extract imaging paths if requested
    if include_imaging:
        imaging_paths = extract_imaging_paths(df)
        if imaging_paths:
            formatted_text = prefix_image_tokens(formatted_text, len(imaging_paths))
        return {"text": formatted_text, "images": imaging_paths}

    return formatted_text
