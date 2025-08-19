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


def _validate_node_configuration(df: pd.DataFrame) -> bool:
    """Check that the dataframe has a valid node configuration.

    Valid configurations:
    1. Only permanent nodes
    2. One startend node and the rest permanent nodes
    3. One start node, one end node, and the rest permanent nodes

    Args:
        df: DataFrame with node_type column to validate

    Returns:
        bool: True if the node configuration is valid, False otherwise
    """
    node_types = df["node_type"].str.strip()

    start_count = (node_types == "start").sum()
    end_count = (node_types == "end").sum()
    startend_count = (node_types == "startend").sum()
    permanent_count = (node_types == "permanent").sum()

    total_nodes = len(df)

    if permanent_count == total_nodes:
        return True

    if startend_count == 1 and permanent_count == total_nodes - 1:
        return True

    return start_count == 1 and end_count == 1 and permanent_count == total_nodes - 2


class GeographicNodeModel(pa.DataFrameModel):
    """Schema for nodes with lat/lon coordinates."""

    name: Series[str] = pa.Field(str_length={"min_value": 1}, unique=True)
    node_type: Series[str] = pa.Field(isin=[n.value for n in NodeType])
    lat: Series[float] = pa.Field(ge=-90.0, le=90.0)
    lon: Series[float] = pa.Field(ge=-180.0, le=180.0)

    class Config:
        """Configuration for the schema."""

        strict = False
        coerce = True  # Enable automatic type coercion

    @pa.dataframe_check
    def _has_valid_node_configuration(cls, df: DataFrame["GeographicNodeModel"]) -> bool:
        """Check that the dataframe has a valid node configuration.

        Valid configurations:
        1. Only permanent nodes
        2. One startend node and the rest permanent nodes
        3. One start node, one end node, and the rest permanent nodes
        """
        return _validate_node_configuration(df)


class CartesianNodeModel(pa.DataFrameModel):
    """Schema for nodes with x/y coordinates."""

    name: Series[str] = pa.Field(str_length={"min_value": 1}, unique=True)
    node_type: Series[str] = pa.Field(isin=[n.value for n in NodeType])
    x: Series[float]
    y: Series[float]

    class Config:
        """Configuration for the schema."""

        strict = False
        coerce = True  # Enable automatic type coercion

    @pa.dataframe_check
    def _has_valid_node_configuration(cls, df: DataFrame["CartesianNodeModel"]) -> bool:
        """Check that the dataframe has a valid node configuration.

        Valid configurations:
        1. Only permanent nodes
        2. One startend node and the rest permanent nodes
        3. One start node, one end node, and the rest permanent nodes
        """
        return _validate_node_configuration(df)
