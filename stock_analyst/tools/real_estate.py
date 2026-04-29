from typing import Literal
from pydantic import BaseModel
from langchain.tools import tool
from pyscbwrapper import SCB


class RegionToolSchema(BaseModel):
    """Schema for the housing price index tool — expects a Swedish region name."""
    region: Literal[
        'Sweden',
        'Greater Stockholm',
        'Greater Gothenburg',
        'Greater Malmö',
        'Stockholm production county',
        'Eastern Central Sweden',
        'Småland with the islands',
        'South Sweden',
        'West Sweden',
        'Northern Central Sweden',
        'Central Norrland',
        'Upper Norrland',
    ]


@tool(args_schema=RegionToolSchema)
def housing_price_index_tool(region):
    """
    Tool to retrieve housing price index data from Sweden Statistics (SCB).

    Args:
        region (str): Predefined regions of Sweden in the SCB database.

    Returns:
        dataframe: The housing price index for the particular region.
    """
    try:
        print("\nUSING HOUSING_PRICE TOOL\n")
        scb = SCB("en", "BO", "BO0501A", "FastpiPSRegAr")
        scb.set_query(
            region=[region],
            year=["2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
        )
        data = scb.get_data()
        return data
    except Exception as e:
        return f"An error occurred while retrieving data: {str(e)}"
