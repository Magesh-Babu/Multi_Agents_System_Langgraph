from typing import Literal, List, Union, Annotated
from typing_extensions import TypedDict

MEMBERS = [
    "Fundamental_Analysis_Agent",
    "Sentiment_Analysis_Agent",
    "Technical_Analysis_Agent",
    "Risk_Assessment_Agent",
    "Real_Estate_Agent",
]


class Router(TypedDict):
    """Worker to route to next."""
    next_worker: List[Literal[
        "Fundamental_Analysis_Agent",
        "Sentiment_Analysis_Agent",
        "Technical_Analysis_Agent",
        "Risk_Assessment_Agent",
        "Real_Estate_Agent",
    ]]


class ConversationalResponse(TypedDict):
    """Respond in a conversational manner. Be kind and helpful."""
    response: Annotated[str, ..., "A conversational response to the user's query"]


class FinalResponse(TypedDict):
    """
    Represents the final output of a system, which can either be:
    Router type, determining the next worker to handle the process.
    ConversationalResponse type, providing a user-friendly response.
    """
    final_output: Union[Router, ConversationalResponse]
