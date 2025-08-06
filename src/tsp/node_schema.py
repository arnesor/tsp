from enum import StrEnum

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class NodeType(StrEnum):
    """Represents types of nodes or sites/controls.

    Attributes:
        PERMANENT (str): Represents a permanent node type that remains constant.
        START (str): Represents a starting node type, indicating the beginning of a tour.
        END (str): Represents an ending node type, indicating the end of a tour.
        STARTEND (str): Represents a node type that serves as both the starting
            and ending point of a tour.
    """

    PERMANENT = "permanent"
    START = "start"
    END = "end"
    STARTEND = "startend"


class NodeInputModel(pa.DataFrameModel):
    """Schema for validating nodes or sites/controls input data."""

    name: Series[str] = pa.Field(str_length={"min_value": 1}, unique=True)
    node_type: Series[str] = pa.Field(isin=[n.value for n in NodeType])

    class Config:
        """Allow extra columns in the input data."""

        strict = False

    @pa.dataframe_check
    def _coords_valid(cls, df: DataFrame[pa.DataFrameModel]) -> Series[bool]:
        result = pd.Series([False] * len(df))

        # Check if both lon and lat columns exist and have valid values
        if "lon" in df.columns and "lat" in df.columns:
            lon_is_float = pd.api.types.is_float_dtype(df["lon"])
            lat_is_float = pd.api.types.is_float_dtype(df["lat"])

            if lon_is_float and lat_is_float:
                # Validate longitude and latitude ranges and non-null values
                lon_valid = (
                    df["lon"].notna() & (df["lon"] >= -180.0) & (df["lon"] <= 180.0)
                )
                lat_valid = (
                    df["lat"].notna() & (df["lat"] >= -90.0) & (df["lat"] <= 90.0)
                )
                has_valid_lon_lat = lon_valid & lat_valid
                result = result | has_valid_lon_lat

        # Check if both x and y columns exist and have valid (non-null) values
        if "x" in df.columns and "y" in df.columns:
            # For mixed data with None values, check if non-null values are integers
            x_non_null = df["x"].dropna()
            y_non_null = df["y"].dropna()

            # Check if non-null values are integers (or can be converted to integers)
            x_is_int = len(x_non_null) == 0 or all(
                isinstance(val, int | pd.Int64Dtype)
                or (isinstance(val, float) and val.is_integer())
                for val in x_non_null
            )
            y_is_int = len(y_non_null) == 0 or all(
                isinstance(val, int | pd.Int64Dtype)
                or (isinstance(val, float) and val.is_integer())
                for val in y_non_null
            )

            if x_is_int and y_is_int:
                has_valid_x_y = df["x"].notna() & df["y"].notna()
                result = result | has_valid_x_y

        return result
